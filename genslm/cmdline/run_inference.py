import functools
import hashlib
import os
import time
import uuid
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    layers: List[int] = [-1]
    """Which layers to generate data for, last only by default."""
    output_embeddings: bool = True
    """Whether or not to generate and save embeddings."""
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
        self.raw_sequences = self.read_sequences(fasta_path)
        # Quick transformation to group sequences by kmers
        self.sequences = [
            self.group_by_kmer(seq, kmer_size) for seq in self.raw_sequences
        ]

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
        raw_seq = self.raw_sequences[idx]
        seq = self.sequences[idx]
        batch_encoding = self.tokenizer_fn(seq)
        # Squeeze so that batched tensors end up with (batch_size, seq_length)
        # instead of (batch_size, 1, seq_length)
        sample = {
            "input_ids": batch_encoding["input_ids"].squeeze(),
            "attention_mask": batch_encoding["attention_mask"],
            "indices": torch.from_numpy(np.array([idx])),
            "seq_lens": batch_encoding["attention_mask"].sum(1),
            # Need raw string for hashing
            "na_hash": hashlib.md5(raw_seq.encode("utf-8")).hexdigest(),
        }
        return sample


def _read_average_embedding_process_fn(
    chunk_idxs: Tuple[int, int],
    h5_file_path: Path,
    hidden_dim: int,
    model_seq_len: int,
) -> np.ndarray:
    num_embs = chunk_idxs[1] - chunk_idxs[0]
    embs = np.empty(shape=(num_embs, hidden_dim), dtype=np.float32)
    emb = np.zeros((model_seq_len, hidden_dim), dtype=np.float32)
    with h5py.File(h5_file_path, "r") as f:
        group = f["embeddings"]
        for i, idx in enumerate(map(str, range(*chunk_idxs))):
            seqlen = group[idx].shape[0]
            f[f"embeddings/{idx}"].read_direct(emb, dest_sel=np.s_[:seqlen])
            embs[i] = emb[:seqlen].mean(axis=0)
    return embs


def read_average_embeddings(
    h5_file_path: Path,
    hidden_dim: int = 512,
    seq_len: int = 2048,
    num_workers: int = 4,
    return_md5: bool = False,
) -> Dict[str, np.ndarray]:
    """Read average embeddings from an HDF5 file.

    Parameters
    ----------
    h5_file_path : Path
        path to h5 file
    hidden_dim : int, optional
        hidden dimension of model that generated embeddings, by default 512
    seq_len : int, optional
        sequence length of the model, by default 2048
    num_workers : int, optional
        number of workers to use, by default 4

    Returns
    -------
    Dict[str, np.ndarray]
        embeddings averaged into hidden_dim under the 'embeddings' key, and if specified, the hashes under 'na-hashes'
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    out_data = {}

    with h5py.File(h5_file_path, "r") as h5_file:
        total_embeddings = len(h5_file["embeddings"])
        if return_md5:
            out_data["na-hashes"] = h5_file["na-hashes"][...]

    chunk_size = max(1, total_embeddings // num_workers)
    chunk_idxs = [
        (i, min(i + chunk_size, total_embeddings))
        for i in range(0, total_embeddings, chunk_size)
    ]

    read_func = functools.partial(
        _read_average_embedding_process_fn,
        h5_file_path=h5_file_path,
        hidden_dim=hidden_dim,
        model_seq_len=seq_len,
    )
    out_array = np.empty(shape=(total_embeddings, hidden_dim), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk_emb, chunk_range in zip(
            executor.map(read_func, chunk_idxs), chunk_idxs
        ):
            out_array[chunk_range[0] : chunk_range[1]] = chunk_emb

    out_data["embeddings"] = out_array
    return out_data


def _read_full_embeddings_process_fn(
    chunk_idxs: Tuple[int, int],
    h5_file_path: Path,
    hidden_dim: int,
    model_seq_len: int,
) -> np.ndarray:
    num_embs = chunk_idxs[1] - chunk_idxs[0]
    embs = np.zeros(shape=(num_embs, model_seq_len, hidden_dim), dtype=np.float32)
    emb = np.zeros((model_seq_len, hidden_dim), dtype=np.float32)
    with h5py.File(h5_file_path, "r") as f:
        group = f["embeddings"]
        for i, idx in enumerate(map(str, range(*chunk_idxs))):
            seqlen = group[idx].shape[0]
            emb[:] = 0  # reset
            f[f"embeddings/{idx}"].read_direct(emb, dest_sel=np.s_[:seqlen])
            embs[i] = emb
    return embs


def read_full_embeddings(
    h5_file_path: Path,
    hidden_dim: int = 512,
    seq_len: int = 2048,
    num_workers: int = 4,
    return_md5: bool = False,
) -> Dict[str, np.ndarray]:
    """Read token level embeddings from an HDF5 file.

        Parameters
        ----------
        h5_file_path : Path
            path to h5 file
        hidden_dim : int, optional
            hidden dimension of the model that generated embeddings, by default 512
        seq_len : int, optional
            sequence length of the model, by default 2048
        num_workers : int, optional
            number of workers to use, by default 4

        Returns
        -------
    Dict[str, np.ndarray]
            token level embeddings under the 'embeddings' key, and if specified, the hashes under 'na-hashes'
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    out_data = {}
    with h5py.File(h5_file_path, "r") as h5_file:
        total_embeddings = len(h5_file["embeddings"])
        if return_md5:
            out_data["na-hashes"] = h5_file["na-hashes"][...]

    chunk_size = max(1, total_embeddings // num_workers)
    chunk_idxs = [
        (i, min(i + chunk_size, total_embeddings))
        for i in range(0, total_embeddings, chunk_size)
    ]

    read_func = functools.partial(
        _read_full_embeddings_process_fn,
        h5_file_path=h5_file_path,
        hidden_dim=hidden_dim,
        model_seq_len=seq_len,
    )

    out_array = np.empty(
        shape=(total_embeddings, seq_len, hidden_dim), dtype=np.float32
    )
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk_emb, chunk_range in zip(
            executor.map(read_func, chunk_idxs), chunk_idxs
        ):
            out_array[chunk_range[0] : chunk_range[1]] = chunk_emb

    out_data["embeddings"] = out_array
    return out_data


class OutputsCallback(Callback):
    def __init__(
        self,
        save_dir: Path = Path("./outputs"),
        layers: List[int] = [-1],
        output_embeddings: bool = True,
        output_attentions: bool = False,
        output_logits: bool = False,
    ) -> None:
        self.rank_label = uuid.uuid4()

        self.output_attentions = output_attentions
        self.output_logits = output_logits
        self.output_embeddings = output_embeddings
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.layers = layers

        # Embeddings: Key layer-id, value embedding array
        self.attentions, self.indices, self.na_hashes = [], [], []

        self.h5embeddings_open: Dict[int, h5py.File] = {}
        self.h5logit_file = None

        self.h5_kwargs = {
            # "compression": "gzip",
            # "compression_opts": 4, Compression is too slow for current impl
            # "fletcher32": True,
        }

        self.io_time = 0

    def on_predict_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Plus one for initial embedding layer
        num_hidden_layers = pl_module.model.model.config.num_hidden_layers + 1

        for ind in range(len(self.layers)):
            layer_num = self.layers[ind]
            if layer_num < 0:
                # e.g -1 turns into model_layers + -1 (e.g. 12 + -1 = 11 last layer for 0 indexed arrays)
                self.layers[ind] = num_hidden_layers + layer_num

        if self.output_logits:
            self.h5logit_file = h5py.File(
                self.save_dir / f"logits-{self.rank_label}.h5", "w"
            )
            self.h5logit_file.create_group("logits")

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
        seq_lens = batch["seq_lens"].detach().cpu().numpy().reshape(-1)
        fasta_inds = batch["indices"].detach().cpu().numpy().reshape(-1)

        if self.output_attentions:
            attend = torch.sum(outputs.attentions[0].detach().cpu().squeeze(), dim=0)
            self.attentions.append(attend)

        if self.output_logits:
            start = time.time()
            logits = outputs.logits.detach().cpu().numpy()
            for logit, seq_len, fasta_ind in zip(logits, seq_lens, fasta_inds):
                self.h5logit_file["logits"].create_dataset(
                    f"{fasta_ind}", data=logit[:seq_len], **self.h5_kwargs
                )
            self.io_time += time.time() - start

        if self.output_embeddings:
            start = time.time()
            for layer, embeddings in enumerate(outputs.hidden_states):
                # User specified list of layers to take
                if layer not in self.layers:
                    continue

                h5_file = self.h5embeddings_open.get(layer)
                if h5_file is None:
                    name = (
                        self.save_dir / f"embeddings-layer-{layer}-{self.rank_label}.h5"
                    )
                    h5_file = h5py.File(name, "w")
                    h5_file.create_group("embeddings")
                    self.h5embeddings_open[layer] = h5_file

                embed = embeddings.detach().cpu().numpy()
                for emb, seq_len, fasta_ind in zip(embed, seq_lens, fasta_inds):
                    h5_file["embeddings"].create_dataset(
                        f"{fasta_ind}", data=emb[:seq_len], **self.h5_kwargs
                    )

                h5_file.flush()
            self.io_time += time.time() - start
        self.na_hashes.extend(batch["na_hash"])
        self.indices.append(batch["indices"].detach().cpu())

    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        self.indices = torch.cat(self.indices).numpy().reshape(-1)

        if self.output_logits:
            start = time.time()
            self.h5logit_file.create_dataset(
                "fasta-indices", data=self.indices, **self.h5_kwargs
            )
            print(self.na_hashes, flush=True)
            self.h5logit_file.create_dataset(
                "na-hashes", data=self.na_hashes, **self.h5_kwargs
            )
            self.h5logit_file.close()
            self.io_time += time.time() - start

        if self.output_embeddings:
            start = time.time()
            # Write indices to h5 files to map embeddings back to fasta file
            for h5_file in self.h5embeddings_open.values():
                h5_file.create_dataset(
                    "fasta-indices", data=self.indices, **self.h5_kwargs
                )
                h5_file.create_dataset(
                    "na-hashes", data=self.na_hashes, **self.h5_kwargs
                )

            # Close all h5 files
            for h5_file in self.h5embeddings_open.values():
                h5_file.close()
            self.io_time += time.time() - start

        print("IO time:\t", self.io_time)


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
