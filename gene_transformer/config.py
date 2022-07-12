import json
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseSettings as _BaseSettings

_T = TypeVar("_T")

PathLike = Union[str, Path]


class BaseSettings(_BaseSettings):
    def dump_yaml(self, cfg_path: PathLike) -> None:
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class WarmupLRSettings(BaseSettings):
    """Learning rate warm up settings."""

    min_lr: float = 5e-8
    """The starting learning rate."""
    num_steps: int = 50000
    """Steps to warm up for."""


class ModelSettings(BaseSettings):
    # logging settings
    wandb_active: bool = True
    wandb_project_name: str = "codon_transformer"
    checkpoint_dir: Path = Path("codon_transformer")
    node_local_path: Optional[Path] = None
    num_nodes: int = 1

    # data settings
    genome_level: bool = False
    tokenizer_file: Path = Path("tokenizer_files/codon_wordlevel_100vocab.json")
    train_file: Path
    val_file: Path
    test_file: Path
    kmer_size: int = 3
    small_subset: int = 0

    # blast settings
    enable_blast: bool = True
    blast_validation_file: Path = Path("blast_file.fasta")
    num_blast_seqs_per_gpu: int = 5
    blast_exe_path: Path = Path("blastn")  # Defaults to current conda environment

    # model settings
    model_name: str = "gpt2"
    batch_size: int = 8
    epochs: int = 5
    block_size: int = 512
    accumulate_grad_batches: int = 4
    learning_rate: float = 5e-5
    precision: int = 16
    warm_up_lr: Optional[WarmupLRSettings] = None
    load_from_checkpoint_dir: Optional[Path] = None
    deepspeed_cfg_file: Optional[Path] = None

    # generation settings
    num_test_seqs_per_gpu: int = 8
    custom_seq_name: str = "SyntheticSeq"

    # training ops
    num_data_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


if __name__ == "__main__":
    settings = ModelSettings(
        train_file=Path("train.fasta"),
        val_file=Path("val.fasta"),
        test_file=Path("test.fasta"),
    )
    settings.dump_yaml("settings_template.yaml")
