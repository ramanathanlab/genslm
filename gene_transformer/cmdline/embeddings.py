import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pydantic import root_validator, validator
from torch.utils.data import DataLoader  # Subset

import gene_transformer
from gene_transformer.config import BaseSettings, WarmupLRSettings
from gene_transformer.dataset import FileBackedH5Dataset
from gene_transformer.model import DNATransformer
from gene_transformer.utils import (
    EmbeddingsCallback,
    LoadDeepSpeedStrategy,
    LoadPTCheckpointStrategy,
)


class InferenceConfig(BaseSettings):
    data_file: Path
    """Data file to run inference on (HDF5)."""
    embeddings_out_path: Path
    """Output path to write embeddings to (npy)."""
    model_config_json: Path
    """Huggingface json dict to load AutoConfig from."""
    load_pt_checkpoint: Optional[Path] = None
    """Checkpoint pt file to initialze model weights."""
    load_ds_checkpoint: Optional[Path] = None
    """DeepSpeed checkpoint file to initialze model weights."""
    precision: int = 16
    """Model precision."""
    num_nodes: int = 1
    """Number of nodes to use for inference."""
    batch_size: int = 32
    """Batch size to use for inference."""
    num_data_workers: int = 4
    """Number of subprocesses to use for data loading."""
    prefetch_factor: int = 2
    """Number of batches loaded in advance by each worker."""
    pin_memory: bool = True
    """If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them."""

    # Parameters needed to initialize DNATransformer (not used for inference)
    tokenizer_file: Path = (
        Path(gene_transformer.__file__).parent
        / "tokenizer_files"
        / "codon_wordlevel_100vocab.json"
    )
    learning_rate: float = 5e-5
    warm_up_lr: Optional[WarmupLRSettings] = None

    @root_validator
    def assert_checkpoint_file_specified(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        load_pt_checkpoint: Optional[Path] = values.get("load_pt_checkpoint")
        load_ds_checkpoint: Optional[Path] = values.get("load_ds_checkpoint")
        if load_pt_checkpoint is None and load_ds_checkpoint is None:
            raise ValueError(
                "At least one of load_pt_checkpoint or load_ds_checkpoint must be specified."
            )
        return values

    @validator("data_file")
    def data_file_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise FileNotFoundError(f"data_file path does not exist {v}.")
        return v

    @validator("load_pt_checkpoint")
    def load_pt_checkpoint_exists(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise FileNotFoundError(f"load_pt_checkpoint path does not exist {v}.")
        return v

    @validator("load_ds_checkpoint")
    def load_ds_checkpoint_exists(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise FileNotFoundError(f"load_ds_checkpoint path does not exist {v}.")
        return v

    @validator("embeddings_out_path")
    def assert_embeddings_out_path_npy(cls, v: Path) -> Path:
        if v.suffix != ".npy":
            raise ValueError("embeddings_out_path must have a .npy extension")
        return v


# class DNATransformer(pl.LightningModule):
#     def __init__(self, cfg: InferenceConfig) -> None:
#         # Loads from a hugging face JSON file
#         base_config = AutoConfig.from_pretrained(cfg.model_config_json)
#         self.model = AutoModelForCausalLM.from_config(base_config)

#     def forward(
#         self, batch: Dict[str, torch.Tensor], **kwargs: Dict[str, Any]
#     ) -> ModelOutput:
#         return self.model(
#             batch["input_ids"],
#             labels=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             **kwargs,
#         )

#     def predict_step(
#         self, batch: Dict[str, torch.Tensor], batch_idx: int
#     ) -> ModelOutput:
#         return self(batch, output_hidden_states=True)


def main(config: InferenceConfig) -> None:
    # Setup torch environment
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # torch.set_num_threads(config.num_data_workers)  # type: ignore[attr-defined]
    pl.seed_everything(0)

    if config.load_pt_checkpoint:
        model_strategy = LoadPTCheckpointStrategy(config.load_pt_checkpoint, cfg=config)
    else:
        model_strategy = LoadDeepSpeedStrategy(config.load_ds_checkpoint, cfg=config)

    model: DNATransformer = model_strategy.get_model(DNATransformer)

    embedding_callback = EmbeddingsCallback()
    trainer = pl.Trainer(
        gpus=-1,
        precision=config.precision,
        num_nodes=config.num_nodes,
        callbacks=[embedding_callback],
        strategy="ddp",
    )

    dataset = FileBackedH5Dataset(config.data_file)
    # dataset = Subset(dataset, np.arange(512))  # for testing
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_data_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
    )

    print(f"Running inference with dataset length {len(dataloader)}")
    trainer.predict(model, dataloaders=dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config = InferenceConfig.from_yaml(args.config)
    main(config)
