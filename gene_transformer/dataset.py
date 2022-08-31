import functools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

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


# TODO: Add h5py to requirements.txt and update docker
class H5Dataset(Dataset):
    def __init__(
        self,
        file_path: str,
        block_size: int,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int = 3,
    ) -> None:
        self.file_path = file_path
        self.block_size = block_size
        self.kmer_size = kmer_size
        self.tokenizer = tokenizer

        with h5py.File(file_path, "r") as f:
            # fetch all samples from the dataset
            self.input_ids = f["input_ids"][...]
            self.attn_masks = f["attention_mask"][...]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # TODO: If it takes too much memory to read the whole dataset
        #       into memory, then we can open the h5py in the getitem
        #       and cache the required idx values like in the other Dataset class.
        return {
            "input_ids": torch.tensor(self.input_ids[idx].astype("int32")).long(),
            "attention_mask": torch.tensor(self.attn_masks[idx].astype("int32")).long(),
        }

    @staticmethod
    def preprocess(
        fasta_path: PathLike,
        output_file: PathLike,
        tokenizer: PreTrainedTokenizerFast,
        block_size: int = 2048,
        kmer_size: int = 3,
    ) -> None:
        # TODO: Instead of passing in a tokenizer, it may be easier to simply define one here.
        fasta_path = Path(fasta_path)
        # TODO: Handle the case were fasta_path is directory of fastas, or single file
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
            # Squeeze so that batched tensors end up with (batch_size, seq_length)
            # instead of (batch_size, 1, seq_length)
            fields["input_ids"].append(batch_encoding["input_ids"].squeeze().astype(np.int8))
            fields["attention_mask"].append(batch_encoding["attention_mask"])

            fields["id"].append(np.array(seq_record.id, dtype=object))
            fields["description"].append(np.array(seq_record.description, dtype=object))
            fields["sequence"].append(np.array(str(seq_record.seq).upper(), dtype=object))
            # TODO: Add other fields?
        print(np.concatenate(fields["input_ids"]).shape)
        print(np.concatenate(fields["attention_masks"]).shape)
        exit()
        # Gather into numpy arrays
        fields = {key: np.concatenate(fields[key]) for key in fields}

        # TODO: Some of these arrays (id, description, attention_mask) may be ragged

        # Write to HDF5 file
        with h5py.File(output_file, "w") as f:
            # TODO: Experiment with smaller compression ratio
            create_dataset = functools.partial(
                f.create_dataset,
                fletcher32=True,
                chunks=True,
                compression="gzip",
                compression_opts=9,
            )
            create_dataset("input_ids", data=fields["input_ids"], dtype="i8")
            # TODO: Which is the best type to use?
            create_dataset("attention_mask", data=fields["attention_mask"], dtype="i8")
            # TODO: Test/debug: https://docs.h5py.org/en/stable/strings.html
            create_dataset("id", data=fields["id"], dtype="S")
            create_dataset("description", data=fields["description"], dtype="S")
            create_dataset("sequence", data=fields["sequence"], dtype="S")
