"""Configuration."""
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseSettings as _BaseSettings
from pydantic import root_validator, validator

import genslm

_T = TypeVar("_T")

PathLike = Union[str, Path]


def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def path_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


class BaseSettings(_BaseSettings):
    """Base settings to provide an easier interface to read/write YAML files."""

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


class CosineWithWarmupLRSettings(BaseSettings):
    """Learning rate scheduler settings to go into transformers.get_cosine_schedule_with_warmup.
    Note that the number of total training steps is taken from the full model config.
    From huggingface: Create a schedule with a learning rate that decreases following the values of the cosine function
    between the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0
     and the initial lr set in the optimizer.
    """

    num_warmup_steps: int = 150
    num_cycles: float = 0.5


class ReduceLROnPlateauSettings(BaseSettings):
    mode: str = "min"
    """LR will adjust based on minimizing/maximizing a metric"""
    factor: float = 0.1
    """Factor to decrease learning rate by upon plateau"""
    patience: int = 10
    """Number of epochs with no improvement after which learning rate will be reduced.
       For example, if patience = 2, then we will ignore the first 2 epochs with no
       improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn't
       improved then. Default: 10."""
    threshold: float = 1e-4
    """Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4."""
    eps: float = 1e-8
    """Minimal decay applied to lr. If the difference between new and old lr is
    smaller than eps, the update is ignored. Default: 1e-8."""


class ModelSettings(BaseSettings):
    """Settings for the DNATransformer model."""

    # logging settings
    wandb_active: bool = False
    """Whether to use wandb for logging."""
    wandb_project_name: str = "codon_transformer"
    """Wandb project name to log to."""
    wandb_entity_name: Optional[str] = None
    """Team name for wandb logging."""
    wandb_model_tag: Optional[str] = None
    """Model tag for wandb labeling and resuming."""
    checkpoint_dir: Optional[Path] = Path("codon_transformer")
    """Checkpoint directory to backup model weights."""
    load_pt_checkpoint: Optional[Path] = None
    """Checkpoint pt file to initialze model weights."""
    load_ds_checkpoint: Optional[Path] = None
    """DeepSpeed checkpoint file to initialze model weights."""
    node_local_path: Optional[Path] = None
    """A node local storage option to write temporary files to."""
    num_nodes: int = 1
    """The number of compute nodes used for training."""
    compute_throughput: bool = False
    """Flag for profiling - uses small subset to compute average throughput over 5 epochs after pinning."""
    profiling_path: Optional[Path] = None
    """Set to path if we want to run pytorch profiler"""
    enable_perplexity: bool = True
    """Enable logging of model perplexity"""
    log_every_n_steps: int = 50
    """Perform logging and perplexity checks every n steps"""
    val_check_interval: Union[Optional[int], Optional[float]] = None
    """Run validation set each n steps - pass an int for every n steps and a float for every percent of training"""
    limit_val_batches: Optional[int] = None
    """Limit validation batches to this many batches:
    total_val_samples = (num_ranks * mini_batch) * limit_val_batches"""
    check_val_every_n_epoch: int = 1
    """Run validation every n number of epochs"""
    checkpoint_every_n_train_steps: Optional[int] = None
    """Number of training steps to perform model checkpointing"""
    checkpoint_every_n_epochs: Optional[int] = None
    """Number of training epochs to perform model checkpointing"""
    deepspeed_flops_profile: bool = False
    """Flag to set whether or not to run deepspeed profiling on training"""

    # data settings
    tokenizer_file: Path = (
        Path(genslm.__file__).parent
        / "tokenizer_files"
        / "codon_wordlevel_69vocab.json"
    )
    """Path to the tokenizer file."""
    train_file: Path
    """Path to the training data."""
    val_file: Path
    """Path to the validation data."""
    test_file: Path
    """Path to the testing data."""
    kmer_size: int = 3
    """Size of kmer to use for tokenization."""
    small_subset: int = 0
    """Subset of data files to use during training. Uses the full dataset by default."""

    # blast settings
    enable_blast: bool = False
    """Whether or not to run BLAST during validation steps."""
    blast_validation_file: Path = Path("blast_file.fasta")
    """Path to fasta file to BLAST against."""
    num_blast_seqs_per_gpu: int = 5
    """Number of BLAST jobs per GPU/rank."""
    blast_exe_path: Path = Path("blastn")
    """Path to the BLAST executable, defaults to current conda environment."""

    # model settings
    random_seed: int = 0
    """Random seed for PL seed everything call"""
    model_config_json: Path
    """Huggingface json dict to load AutoConfig from."""
    batch_size: int = 8
    """Training micro-batch size."""
    epochs: int = 5
    """Number of training epochs."""
    max_steps: int = -1
    """Max number of training steps. -1 means that max number of epochs is used instead."""
    gradient_clip_value: float = 0.0
    """clip gradients' global norm to <=value using gradient_clip_algorithm='norm' by default. 0 means no clipping."""
    block_size: int = 512
    """Block size to specify sequence length passed to the transformer."""
    accumulate_grad_batches: int = 1
    """Number of batches to accumulate before gradient updates."""
    learning_rate: float = 5e-5
    """Learning rate to use for training."""
    precision: int = 16
    """Training precision."""
    warm_up_lr: Optional[WarmupLRSettings] = None
    """If specified, will use a learning rate warmup scheduler."""
    lr_plateau: Optional[ReduceLROnPlateauSettings] = None
    """If specified, will use a LR plateau scheduler."""
    lr_cosine_with_warmup: Optional[CosineWithWarmupLRSettings] = None
    """If specified, will use a cosine with warmup LR scheduler."""
    deepspeed_cfg_file: Optional[Path] = None
    """The deepspeed configuration file (currently unused)."""
    deepspeed_stage: int = 3
    """Which deepspeed stage to use"""
    offload_parameters: Optional[bool] = False
    """Whether or not to offload parameters using DeepSpeed to the CPU"""
    offload_optimizer: Optional[bool] = False
    """Whether or not to offload optimizer using DeepSpeed to the CPU"""
    offload_device: Optional[str] = "cpu"
    """The device to offload parameters using DeepSpeed - defaults to cpu"""
    nvme_path: Optional[str] = "/local/scratch"
    """The path to the nvme drive"""
    partition_activations: Optional[bool] = False
    """Whether or not activations are being checkpointed"""

    # generation settings
    num_test_seqs_per_gpu: int = 0
    """Number of sequences to generate per GPU when testing."""
    custom_seq_name: str = "SyntheticSeq"
    """Custum sequence name to write into fasta files for generate sequences."""

    # training ops (see PyTorch DataLoader for details.)
    num_data_workers: int = 4
    """Number of subprocesses to use for data loading."""
    prefetch_factor: int = 4
    """Number of batches loaded in advance by each worker."""
    pin_memory: bool = True
    """If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them."""
    persistent_workers: bool = True
    """If True, the data loader will not shutdown the worker processes after a dataset has been consumed once."""

    @validator("node_local_path")
    def resolve_node_local_path(cls, v: Optional[Path]) -> Optional[Path]:
        # Check if node local path is stored in environment variable
        # Example: v = Path("$PSCRATCH") => str(v)[1:] == "PSCRATCH"
        return None if v is None else Path(os.environ.get(str(v)[1:], v))

    @root_validator
    def warn_checkpoint_load(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        load_pt_checkpoint = values.get("load_pt_checkpoint")
        load_ds_checkpoint = values.get("load_ds_checkpoint")
        if load_pt_checkpoint is not None and load_ds_checkpoint is not None:
            warnings.warn(
                "Both load_pt_checkpoint and load_ds_checkpoint are "
                "specified in the configuration. Loading from load_pt_checkpoint."
            )
        return values

    @root_validator
    def warn_checkpoint_steps(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint_every_n_train_steps = values.get("checkpoint_every_n_train_steps")
        checkpoint_every_n_epochs = values.get("checkpoint_every_n_epochs")
        if (
            checkpoint_every_n_train_steps is not None
            and checkpoint_every_n_epochs is not None
        ):
            warnings.warn(
                "Both checkpoint_every_n_train_steps and checkpoint_every_n_epochs are "
                "specified in the configuration. Using checkpoint_every_n_train_steps."
            )
            values["checkpoint_every_n_epochs"] = None
        elif (
            checkpoint_every_n_train_steps is None and checkpoint_every_n_epochs is None
        ):
            warnings.warn(
                "Both checkpoint_every_n_train_steps and checkpoint_every_n_epochs are "
                "missing in the configuration. PLease specify one of these to log checkpoints."
            )
        return values


def throughput_config(cfg: ModelSettings) -> ModelSettings:
    new_config = cfg.copy()
    new_config.enable_perplexity = False
    new_config.checkpoint_dir = None
    new_config.epochs = 6
    new_config.check_val_every_n_epoch = 7
    new_config.limit_val_batches = 0
    # Select size of subset to use, more ranks require more data to compute stats.
    new_config.small_subset = 32 * new_config.batch_size * new_config.num_nodes
    return new_config


if __name__ == "__main__":
    settings = ModelSettings(
        train_file=Path("train.fasta"),
        val_file=Path("val.fasta"),
        test_file=Path("test.fasta"),
        model_config_json=Path("model.json"),
    )
    settings.dump_yaml("settings_template.yaml")
