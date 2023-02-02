import json


from pathlib import Path
import functools
import os
from collections import defaultdict
from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl

import torch.multiprocessing as mp
from torch.utils.data import DataLoader


from genslm.inference import GenSLM

from genslm.cmdline.inference import (
    InferenceConfig,
    InferenceSequenceDataset,
    OutputsCallback,
    LightningGenSLM,
)

node_rank = int(os.environ.get("NODE_RANK", 0))


class BulkInferenceConfig(InferenceConfig):
    """Configuration for bulk inference."""

    num_nodes_per_file: int = 1
    """Number of nodes to split each fasta file into."""

    file_lengths: Optional[Path]
    """Length of each fasta file in input directory. If not provided, will split files equally across all ranks."""


def main(config: BulkInferenceConfig) -> None:
    # Setup torch environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # Potential polaris fix for connection reset error
    mp.set_start_method("spawn")
    pl.seed_everything(42)

    # Load GenSLM model and inject into pytorch lightning
    model = GenSLM(config.model_id, config.model_cache_dir)
    # Set the default kwarg values once
    model.forward = functools.partial(
        model.forward,
        output_hidden_states=config.output_embeddings,
        output_attentions=config.output_attentions,
    )
    ptl_model = LightningGenSLM(model)

    # Create callback to save model outputs to disk
    outputs_callback = OutputsCallback(
        save_dir=config.output_path,
        layer_bounds=config.layer_bounds,
        mean_embedding_reduction=config.mean_embedding_reduction,
        output_embeddings=config.output_embeddings,
        output_attentions=config.output_attentions,
        output_logits=config.output_logits,
    )

    # Use pytorch lightning trainer to take advantage of distribution strategies
    trainer = pl.Trainer(
        gpus=-1,
        precision=config.precision,
        num_nodes=config.num_nodes_per_file,
        callbacks=[outputs_callback],
        strategy="ddp",
        logger=False,  # Avoid lightning_logs dir
        max_epochs=-1,  # Avoid warning
    )

    # This dataset loads each sequence from each fasta file into memory
    # as strings on each rank and then tokenizes on-the-fly.
    dataset = InferenceSequenceDataset(
        config.data_file, model.seq_length, model.tokenizer
    )
    # dataset = Subset(dataset, np.arange(512))  # for testing
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_data_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
    )

    if trainer.is_global_zero:
        print(f"Running inference with dataset length {len(dataloader)}")
        if config.output_embeddings:
            print("Generating embeddings values...")
        if config.output_attentions:
            print("Generating attention values...")
        if config.output_logits:
            print("Generating logit values...")

    trainer.predict(ptl_model, dataloaders=dataloader, return_predictions=False)

    if trainer.is_global_zero:
        print("Done")


if __name__ == "__main__":
    parser = ArgumentParser(
        (
            "This script runs inference on a GenSLM model in bulk.",
            "It takes same yaml file as `inference.py`",
            "but assumes the input is a directory of fasta files.",
        )
    )
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config = BulkInferenceConfig.from_yaml(args.config)

    if config.file_lengths:
        file_lengths = config.file_lengths.read_text().splitlines()
        file_lengths = {item.split()[0]: int(item.split()[1]) for item in file_lengths}
        file_lengths = {
            k: v for k, v in sorted(file_lengths.items(), key=lambda item: item[1])
        }

        process_files = defaultdict(list)
        for i, file in enumerate(file_lengths):
            if (config.data_file / file).exists():
                process_files[i % config.num_nodes].append(file)

    else:
        # Split files equally across all ranks
        files = list(config.data_file.glob("*.ffn"))
        files_per_node = len(files) // config.num_nodes
        process_files = [
            files[i * files_per_node : (i + 1) * files_per_node]
            for i in range(config.num_nodes_per_file)
        ]

    if node_rank == 0:
        json.dump(dict(process_files), open(f"process_files_{node_rank}.json", "w"))

    # Trick ptl into thinking we are single node
    os.environ["WORLD_SIZE"] = "4"
    os.environ["NODE_RANK"] = "0"

    for file in process_files[node_rank]:
        file_config = config.copy()
        file_config.data_file = config.data_file / file
        file_config.output_path = config.output_path / Path(file).stem
        file_config.output_path.mkdir(parents=True, exist_ok=True)
        main(file_config)

