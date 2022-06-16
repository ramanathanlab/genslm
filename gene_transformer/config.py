import json
import yaml
from pathlib import Path
from typing import Type, TypeVar, Union, Optional
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


class ModelSettings(BaseSettings):
    # logging settings
    wandb_active: bool = True
    wandb_project_name: str = "codon_transformer"
    checkpoint_dir: Path = Path("codon_transformer")
    node_local_path: Optional[Path] = None
    num_nodes: int = 1

    # data settings
    alphabet_type: str = "codon"
    tokenizer_file: str = "tokenizer_files/codon_wordlevel_100vocab.json"
    train_file: str = "data/full_mdh_fasta/train.fasta"
    val_file: str = "data/full_mdh_fasta/val.fasta"
    test_file: str = "data/full_mdh_fasta/test.fasta"

    # blast settings
    enable_blast: bool = True
    blast_validation_file: str = "blast_file.fasta"
    num_blast_seqs_per_gpu: int = 5
    blast_exe_path: Path = Path("blastn")  # Defaults to current conda environment

    # model settings
    model_name: str = "gpt2"
    batch_size: int = 8
    epochs: int = 5
    block_size: int = 512
    accumulate_grad_batches: int = 4
    learning_rate: float = 5e-5
    load_from_checkpoint_dir: Optional[Path] = None

    # generation settings
    num_test_seqs_per_gpu: int = 8

    # training ops
    num_data_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


if __name__ == "__main__":
    settings = ModelSettings()
    settings.dump_yaml("settings_template.yaml")
