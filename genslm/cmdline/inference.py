import functools
import os
import uuid
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from natsort import natsorted
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, Dataset  # Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from genslm.config import BaseSettings, path_validator
from genslm.inference import GenSLM
from genslm.utils import read_fasta_only_seq


class InferenceConfig(BaseSettings):
    # Input files
    model_id: str = "genslm_25M_patric"
    """The genslm model to load."""
    model_cache_dir: Path
    """The directory of the model weights."""
    data_file: Path
    """Data file to run inference on (HDF5)."""
    output_path: Path
    """Directory to write embeddings, attentions, logits to."""

    # Which outputs to generate
    output_embeddings: bool = True
    """Whether or not to generate and save embeddings."""
    mean_embedding_reduction: bool = False
    """Whether or not to average the embeddings over sequence length."""
    output_attentions: bool = False
    """Whether or not to generate and save attentions."""
    output_logits: bool = False
    """Whether or not to generate and save logits."""

    # Run time settings
    num_nodes: int = 1
    """Number of nodes to use for inference."""
    precision: int = 16
    """Model precision."""
    batch_size: int = 32
    """Batch size to use for inference."""
    num_data_workers: int = 4
    """Number of subprocesses to use for data loading."""
    prefetch_factor: int = 2
    """Number of batches loaded in advance by each worker."""
    pin_memory: bool = True
    """If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them."""

    # validators
    _data_file_exists = path_validator("data_file")
    _model_cache_dir_exists = path_validator("model_cache_dir")


class InferenceSequenceDataset(Dataset):
    """Dataset initialized from fasta files."""

    def __init__(
        self,
        fasta_path: Path,
        seq_length: int,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int = 3,
    ):

        # Read all fasta files into memory as strings
        self.sequences = self.read_sequences(fasta_path)
        # Quick transformation to group sequences by kmers
        self.sequences = [self.group_by_kmer(seq, kmer_size) for seq in self.sequences]

        # Define tokenizer function, but wait to tokenize
        # until a specific batch is requested
        self.tokenizer_fn = functools.partial(
            tokenizer,
            max_length=seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    @staticmethod
    def read_sequences(fasta_path: Path) -> List[str]:
        sequences = []
        if fasta_path.is_dir():
            fasta_files = natsorted(fasta_path.glob("*.fasta"))
            for fasta_file in tqdm(fasta_files, desc="Reading fasta files..."):
                sequences.extend(read_fasta_only_seq(fasta_file))
        else:
            sequences = read_fasta_only_seq(fasta_path)
        return sequences

    @staticmethod
    def group_by_kmer(seq: str, kmer: int) -> str:
        return " ".join(seq[i : i + kmer] for i in range(0, len(seq), kmer)).upper()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        batch_encoding = self.tokenizer_fn(seq)
        # Squeeze so that batched tensors end up with (batch_size, seq_length)
        # instead of (batch_size, 1, seq_length)
        sample = {
            "input_ids": batch_encoding["input_ids"].squeeze(),
            "attention_mask": batch_encoding["attention_mask"],
            "indices": torch.from_numpy(np.array([idx])),
            "seq_lens": torch.from_numpy(np.array([len(seq)])),
        }
        return sample


class OutputsCallback(Callback):
    def __init__(
        self,
        save_dir: Path = Path("./outputs"),
        mean_embedding_reduction: bool = False,
        output_embeddings: bool = True,
        output_attentions: bool = False,
        output_logits: bool = False,
    ) -> None:
        self.mean_embedding_reduction = mean_embedding_reduction
        self.output_attentions = output_attentions
        self.output_logits = output_logits
        self.output_embeddings = output_embeddings
        self.save_dir = save_dir
        # Embeddings: Key layer-id, value embedding array
        self.embeddings, self.attentions, self.indices = defaultdict(list), [], []
        save_dir.mkdir(exist_ok=True)

        self.h5s_open: Dict[int, h5py.File] = {}
        self.rank_label = uuid.uuid4()

    def on_predict_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.embeddings, self.attentions, self.indices = defaultdict(list), [], []

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # outputs.hidden_states: (layer, batch_size, sequence_length, hidden_size)
        if self.output_attentions:
            attend = torch.sum(outputs.attentions[0].detach().cpu().squeeze(), dim=0)
            self.attentions.append(attend)
        if self.output_logits:
            logits = outputs.logits.detach().cpu()
            self.logits.append(logits)
        if self.output_embeddings:

            for layer, embeddings in enumerate(outputs.hidden_states):

                # if self.mean_embedding_reduction:
                #     # Compute average over sequence length
                #     # TODO: Account for padding
                #     embed = embeddings.detach().mean(dim=1).cpu()
                # else:

                h5_file = self.h5s_open.get(layer)
                if h5_file is None:
                    name = (
                        self.save_dir / f"embeddings-layer-{layer}-{self.rank_label}.h5"
                    )
                    h5_file = h5py.File(name, "w")
                    h5_file.create_group("embeddings")
                    self.h5s_open[layer] = h5_file

                embed = embeddings.detach().cpu().numpy()
                # TODO: check +1 is correct for padding

                for i, (e, seq_len) in enumerate(zip(embed, batch["seq_lens"])):
                    h5_file["embeddings"].create_dataset(
                        f"{i}", data=e[1 : seq_len + 1]
                    )
                h5_file.flush()

                # self.embeddings[layer].append(embed)

        self.indices.append(batch["indices"].detach().cpu())

    def save_embeddings_h5(self, save_path: Path, data: np.ndarray) -> None:
        with h5py.File(save_path, "w") as f:
            grp = f.create_group("embeddings")
            counter = 0
            for batch in data:
                for example in batch:
                    grp.create_dataset(f"{counter}", data=example)
                    counter += 1
                f.flush()

    def on_predict_end_not_running(  # TODO: Remove this
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Save each ranks data to a unique file
        rank_label = uuid.uuid4()

        if self.output_logits:
            # TODO: figure out if cat is going to mess things up here
            self.logits = torch.cat(self.logits).numpy()
            np.save(self.save_dir / f"logits-{rank_label}.npy", self.logits)

        if self.output_embeddings:
            print(
                "Num layers in embeddings: ",
                len(self.embeddings),
                os.environ["RANK"],
                os.environ["NODE_RANK"],
                os.environ["GLOBAL_RANK"],
            )
            for layer, embed_ in self.embeddings.items():
                # embed = np.concatenate(embed_)
                self.save_embeddings_h5(
                    self.save_dir / f"embeddings-layer-{layer}-{rank_label}.h5", embed_
                )
                # np.save(
                #     self.save_dir / f"embeddings-layer-{layer}-{rank_label}.npy", embed
                # )

        if self.output_attentions:
            self.attentions = torch.stack(self.attentions).numpy()
            np.save(self.save_dir / f"attentions-{rank_label}.npy", self.attentions)

        # Save indices to combine the per-rank files into a single dataset
        self.indices = torch.cat(self.indices).numpy().squeeze()
        np.save(self.save_dir / f"indices-{rank_label}.npy", self.indices)


class LightningGenSLM(pl.LightningModule):
    """Lightning wrapper to facilitate distributed prediction."""

    def __init__(self, model: GenSLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Any:
        return self(batch["input_ids"], batch["attention_mask"])


def main(config: InferenceConfig) -> None:
    # Setup torch environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # Potential polaris fix for connection reset error
    mp.set_start_method("spawn")
    pl.seed_everything(0)

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
        mean_embedding_reduction=config.mean_embedding_reduction,
        output_embeddings=config.output_embeddings,
        output_attentions=config.output_attentions,
        output_logits=config.output_logits,
    )

    # Use pytorch lightning trainer to take advantage of distribution strategies
    trainer = pl.Trainer(
        gpus=-1,
        precision=config.precision,
        num_nodes=config.num_nodes,
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
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config = InferenceConfig.from_yaml(args.config)
    main(config)

    # TODO: Implement embedding padding removal
