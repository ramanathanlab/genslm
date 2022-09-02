import functools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
from Bio import SeqIO  # type: ignore[import]
from natsort import natsorted
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from gene_transformer.config import PathLike


def group_by_kmer(s: SeqIO.SeqRecord, n: int) -> str:
    seq = str(s.seq).upper()  # need to make sure it's in upper case
    return " ".join(seq[i : i + n] for i in range(0, len(seq), n))


def _write_fasta_file(seq: SeqIO.SeqRecord, output_file: Path) -> None:
    SeqIO.write(seq, str(output_file), "fasta")


def write_individual_fasta_files(fasta_file: Path, output_dir: Path, num_workers: int = 1) -> None:
    output_dir.mkdir(exist_ok=True)
    seqs = list(SeqIO.parse(fasta_file, "fasta"))
    output_files = [output_dir / f"sequence-{i}.fasta" for i in range(len(seqs))]
    print(f"Number of sequences: {len(seqs)}")
    chunksize = max(1, len(seqs) // num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in executor.map(_write_fasta_file, seqs, output_files, chunksize=chunksize):
            pass


class FastaDataset(Dataset):
    def __init__(
        self,
        fasta_dir: PathLike,
        block_size: int,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int = 3,
        small_subset: int = 0,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.kmer_size = kmer_size

        self.files = natsorted(Path(fasta_dir).glob("*.fasta"))

        # default of zero will not call this logic
        if small_subset:
            self.files = self.files[:small_subset]

        # Cache the samples in memory
        self.samples: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # tokenize on the fly
        try:
            return self.samples[idx]
        except KeyError:
            sequence = list(SeqIO.parse(self.files[idx], "fasta"))[0]
            batch_encoding = self.tokenizer(
                group_by_kmer(sequence, self.kmer_size),
                max_length=self.block_size,
                padding="max_length",
                return_tensors="pt",
            )
            # Squeeze so that batched tensors end up with (batch_size, seq_length)
            # instead of (batch_size, 1, seq_length)
            sample = {
                "input_ids": batch_encoding["input_ids"].squeeze(),
                "attention_mask": batch_encoding["attention_mask"],
            }
            self.samples[idx] = sample
            return sample


class H5Dataset(Dataset):
    def __init__(
        self,
        file_path: PathLike,
        block_size: int,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int = 3,
        small_subset: int = 0,
    ) -> None:
        self.file_path = file_path
        self.block_size = block_size
        self.kmer_size = kmer_size
        self.tokenizer = tokenizer

        with h5py.File(file_path, "r") as f:
            # fetch all samples from the dataset
            self.input_ids = f["input_ids"][...]
            self.attn_masks = f["attention_mask"][...]

        if small_subset:
            self.input_ids = self.input_ids[:small_subset]
            self.attn_masks = self.attn_masks[:small_subset]

        print(f"{self.input_ids.shape=}")
        print(f"{self.attn_masks.shape=}")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # TODO: If it takes too much memory to read the whole dataset
        #       into memory, then we can open the h5py in the getitem
        #       and cache the required idx values like in the other Dataset class.
        return {
            "input_ids": torch.tensor(self.input_ids[idx]).long(),
            "attention_mask": torch.tensor(self.attn_masks[idx]).long(),
        }

    @staticmethod
    def preprocess(
        fasta_path: PathLike,
        output_file: PathLike,
        tokenizer: PreTrainedTokenizerFast,
        block_size: int = 2048,
        kmer_size: int = 3,
    ) -> None:
        fields = defaultdict(list)
        sequences = list(SeqIO.parse(fasta_path, "fasta"))
        print(f"File: {fasta_path}, num sequences: {len(sequences)}")
        for seq_record in sequences:
            batch_encoding = tokenizer(
                group_by_kmer(seq_record, kmer_size),
                max_length=block_size,
                padding="max_length",
                return_tensors="np",
            )

            for field in ["input_ids", "attention_mask"]:
                fields[field].append(batch_encoding[field].astype(np.int8))
            fields["id"].append(seq_record.id)
            fields["description"].append(seq_record.description)
            fields["sequence"].append(str(seq_record.seq).upper())

        # Gather model input into numpy arrays
        for key in ["input_ids", "attention_mask"]:
            fields[key] = np.concatenate(fields[key])

        # Write to HDF5 file
        with h5py.File(output_file, "w") as f:
            str_dtype = h5py.string_dtype(encoding="utf-8")
            create_dataset = functools.partial(
                f.create_dataset,
                fletcher32=True,
                chunks=True,
                compression="gzip",
                compression_opts=6,
            )
            create_dataset("input_ids", data=fields["input_ids"], dtype="i8")
            create_dataset("attention_mask", data=fields["attention_mask"], dtype="i8")
            create_dataset("id", data=fields["id"], dtype=str_dtype)
            create_dataset("description", data=fields["description"], dtype=str_dtype)
            create_dataset("sequence", data=fields["sequence"], dtype=str_dtype)

    @staticmethod
    def concatenate_virtual_h5(input_files: List[str], output_file: Path, fields: Optional[List[str]] = None) -> None:
        """Concatenate HDF5 files into a virtual HDF5 file.
        Concatenates a list :obj:`input_files` of HDF5 files containing
        the same format into a single virtual dataset.
        Parameters
        ----------
        input_files : List[str]
            List of HDF5 file names to concatenate.
        output_file : Path
            Name of output virtual HDF5 file.
        fields : Optional[List[str]], default=None
            Which dataset fields to concatenate. Will concatenate all fields by default.
        """

        # Open first file to get dataset shape and dtype
        # Assumes uniform number of data points per file
        h5_file = h5py.File(input_files[0], "r")

        if not fields:
            fields = list(h5_file.keys())

        if not fields:
            raise ValueError("No fields found in HDF5 file.")

        lengths = []
        for file in input_files:
            with h5py.File(file, "r") as f:
                lengths.append(f[fields[0]].shape[0])
        total_length = sum(lengths)

        # Helper function to output concatenated shape
        def concat_shape(shape: Tuple[int]) -> Tuple[int]:
            return (total_length, *shape[1:])

        # Create a virtual layout for each input field
        layouts = {
            field: h5py.VirtualLayout(
                shape=concat_shape(h5_file[field].shape),
                dtype=h5_file[field].dtype,
            )
            for field in fields
        }

        with h5py.File(output_file, "w", libver="latest") as f:
            for field in fields:
                for i, filename in enumerate(input_files):
                    shape = h5_file[field].shape
                    vsource = h5py.VirtualSource(filename, field, shape=(lengths[i], *shape[1:]))
                    start_idx = sum(lengths[:i])
                    end_idx = sum(lengths[: i + 1])
                    layouts[field][start_idx:end_idx, ...] = vsource

                f.create_virtual_dataset(field, layouts[field])

        h5_file.close()
