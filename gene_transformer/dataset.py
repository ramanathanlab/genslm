import functools
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def write_individual_fasta_files(
    fasta_file: Path, output_dir: Path, num_workers: int = 1
) -> None:
    output_dir.mkdir(exist_ok=True)
    seqs = list(SeqIO.parse(fasta_file, "fasta"))
    output_files = [output_dir / f"sequence-{i}.fasta" for i in range(len(seqs))]
    print(f"Number of sequences: {len(seqs)}")
    chunksize = max(1, len(seqs) // num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in executor.map(
            _write_fasta_file, seqs, output_files, chunksize=chunksize
        ):
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


class H5PreprocessMixin:
    @staticmethod
    def train_val_test_split(seqs, train_pct, val_pct):
        np.random.seed(42)
        train_pct, val_pct = 0.8, 0.1
        shuffled_inds = np.arange(len(seqs))
        np.random.shuffle(shuffled_inds)
        train_ind = round(len(seqs) * train_pct)
        val_ind = train_ind + round(len(seqs) * val_pct)
        train_split = [seqs[i] for i in shuffled_inds[:train_ind]]
        val_split = [seqs[i] for i in shuffled_inds[train_ind:val_ind]]
        test_split = [seqs[i] for i in shuffled_inds[val_ind:]]
        return train_split, val_split, test_split

    @staticmethod
    def preprocess(
        fasta_file: PathLike,
        output_file: PathLike,
        tokenizer: PreTrainedTokenizerFast,
        block_size: int = 2048,
        kmer_size: int = 3,
        train_val_test_split: Optional[Dict[str, float]] = None,
    ) -> None:
        if train_val_test_split is not None:
            if sum(train_val_test_split.values()) != 1:
                raise ValueError(
                    f"Train test val split percentages {train_val_test_split} do not add up to 100%"
                )

        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        print(f"File: {fasta_file}, num sequences: {len(sequences)}")

        sequence_splits = {}

        if train_val_test_split is not None:
            train_percentage = train_val_test_split["train"]
            val_percentage = train_val_test_split["val"]
            train_split, val_split, test_split = H5PreprocessMixin.train_val_test_split(
                sequences, train_percentage, val_percentage
            )

            sequence_splits["train"] = train_split
            sequence_splits["val"] = val_split
            sequence_splits["test"] = test_split

        else:
            sequence_splits["all"] = sequences

        for split_name, split_sequences in sequence_splits.items():
            fields = defaultdict(list)
            for seq_record in split_sequences:
                batch_encoding = tokenizer(
                    group_by_kmer(seq_record, kmer_size),
                    max_length=block_size,
                    padding="max_length",
                    return_tensors="np",
                    truncation=True,
                )
                for field in ["input_ids", "attention_mask"]:
                    fields[field].append(batch_encoding[field].astype(np.int8))
                fields["id"].append(seq_record.id)
                fields["description"].append(seq_record.description)
                fields["sequence"].append(str(seq_record.seq).upper())

            # Gather model input into numpy arrays
            for key in ["input_ids", "attention_mask"]:
                if len(fields[key]) == 0:
                    warnings.warn(
                        f"{fasta_file} led to empty input array with key: {key}"
                    )
                    return
                fields[key] = np.concatenate(fields[key])

            # Write to HDF5 file
            local_output_file = Path(output_file)
            if split_name != "all":
                local_output_file = (
                    local_output_file.parent / split_name / local_output_file.name
                )

            with h5py.File(local_output_file, "w") as f:
                str_dtype = h5py.string_dtype(encoding="utf-8")
                create_dataset = functools.partial(
                    f.create_dataset,
                    fletcher32=True,
                    chunks=True,
                    compression="gzip",
                    compression_opts=6,
                )
                create_dataset("input_ids", data=fields["input_ids"], dtype="i8")
                create_dataset(
                    "attention_mask", data=fields["attention_mask"], dtype="i8"
                )
                create_dataset("id", data=fields["id"], dtype=str_dtype)
                create_dataset(
                    "description", data=fields["description"], dtype=str_dtype
                )
                create_dataset("sequence", data=fields["sequence"], dtype=str_dtype)

    @staticmethod
    def get_num_samples_in_file(file: Path, field: str) -> int:
        with h5py.File(file, "r") as f:
            return f[field].shape[0]

    @staticmethod
    def get_num_samples(
        input_files: List[Path], field: str, num_workers: int = 1
    ) -> List[int]:
        lengths = []
        func = functools.partial(H5PreprocessMixin.get_num_samples_in_file, field=field)
        chunksize = max(1, len(input_files) // num_workers)
        with ProcessPoolExecutor() as pool:
            for length in pool.map(func, input_files, chunksize=chunksize):
                lengths.append(length)
        return lengths

    @staticmethod
    def concatenate_virtual_h5(
        input_files: List[Path],
        output_file: Path,
        fields: Optional[List[str]] = None,
        num_workers: int = 1,
    ) -> None:
        """Concatenate HDF5 files into a virtual HDF5 file.
        Concatenates a list :obj:`input_files` of HDF5 files containing
        the same format into a single virtual dataset.
        Parameters
        ----------
        input_files : List[Path]
            List of HDF5 file names to concatenate.
        output_file : Path
            Name of output virtual HDF5 file.
        fields : Optional[List[str]], default=None
            Which dataset fields to concatenate. Will concatenate all fields by default.
        num_workers : int, default=1
            Number of process to use for reading the data lengths from each file.
        """

        # Open first file to get dataset shape and dtype
        # Assumes uniform number of data points per file
        h5_file = h5py.File(input_files[0], "r")

        if not fields:
            fields = list(h5_file.keys())

        if not fields:
            raise ValueError("No fields found in HDF5 file.")

        # for file in input_files:
        #     with h5py.File(file, "r") as f:
        #         lengths.append(f[fields[0]].shape[0])
        # print(f"Total lengths {total_length}, num files: {len(input_files)}")
        lengths = H5PreprocessMixin.get_num_samples(input_files, fields[0], num_workers)

        total_length = sum(lengths)
        print(f"Total sequences: {total_length}")

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

        with h5py.File(output_file, "w") as f:
            for field in fields:
                for i, filename in enumerate(input_files):
                    shape = h5_file[field].shape
                    vsource = h5py.VirtualSource(
                        filename, field, shape=(lengths[i], *shape[1:])
                    )
                    start_idx = sum(lengths[:i])
                    end_idx = sum(lengths[: i + 1])
                    layouts[field][start_idx:end_idx, ...] = vsource

                f.create_virtual_dataset(field, layouts[field])

        h5_file.close()


class H5Dataset(Dataset, H5PreprocessMixin):
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

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx]).long(),
            "attention_mask": torch.tensor(self.attn_masks[idx]).long(),
        }


class CachingH5Dataset(Dataset, H5PreprocessMixin):
    def __init__(self, file_path: PathLike, **extra: Any) -> None:
        # Data is preprocessed and does not require tokenizer, etc
        self.file_path = file_path

        # Peek into file to get dataset length
        with h5py.File(file_path, "r") as f:
            self._len = f["input_ids"].shape[0]

        # Cache the samples in memory
        self.samples: Dict[int, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._len)

    def get_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"]).long(),
            "attention_mask": torch.tensor(sample["attention_mask"]).long(),
        }

    def cache_sample_from_h5(self, idx: int) -> None:
        # Accessing self.h5_file may raise AttributeError
        self.samples[idx] = {
            key: self.h5_file[key][idx][...] for key in ["input_ids", "attention_mask"]
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            return self.get_sample(idx)
        except KeyError:
            pass

        try:
            self.cache_sample_from_h5(idx)
        except AttributeError:
            # Need to open the H5 file in the getitem worker process
            self.h5_file = h5py.File(self.file_path, "r")
            self.cache_sample_from_h5(idx)

        return self.get_sample(idx)
