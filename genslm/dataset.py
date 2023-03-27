import functools
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from Bio import SeqIO  # type: ignore[import]
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerFast

from genslm.config import PathLike


# TODO: Remove dependecy for BioPython
# NOTE: Legacy H5 conversion code
def group_by_kmer(s: SeqIO.SeqRecord, n: int) -> str:
    seq = str(s.seq).upper()  # need to make sure it's in upper case
    return " ".join(seq[i : i + n] for i in range(0, len(seq), n))


class H5PreprocessMixin:
    @staticmethod
    def train_val_test_split(
        seqs: List[str], train_pct: float, val_pct: float
    ) -> Dict[str, List[str]]:
        np.random.seed(42)
        shuffled_inds = np.arange(len(seqs))
        np.random.shuffle(shuffled_inds)
        train_ind = round(len(seqs) * train_pct)
        val_ind = train_ind + round(len(seqs) * val_pct)
        split = {
            "train": [seqs[i] for i in shuffled_inds[:train_ind]],
            "val": [seqs[i] for i in shuffled_inds[train_ind:val_ind]],
            "test": [seqs[i] for i in shuffled_inds[val_ind:]],
        }
        return split

    @staticmethod
    def write_h5(ouput_file: PathLike, data: Dict[str, np.ndarray]) -> None:
        with h5py.File(ouput_file, "w") as f:
            str_dtype = h5py.string_dtype(encoding="utf-8")
            create_dataset = functools.partial(
                f.create_dataset,
                chunks=True,
                compression="gzip",
                compression_opts=6,
            )
            create_dataset("input_ids", data=data["input_ids"], dtype="i8")
            create_dataset("attention_mask", data=data["attention_mask"], dtype="i8")
            create_dataset("id", data=data["id"], dtype=str_dtype)
            create_dataset("description", data=data["description"], dtype=str_dtype)
            create_dataset("sequence", data=data["sequence"], dtype=str_dtype)

    @staticmethod
    def preprocess(
        fasta_file: PathLike,
        output_file: PathLike,
        tokenizer: PreTrainedTokenizerFast,
        block_size: int = 2048,
        kmer_size: int = 3,
        train_val_test_split: Optional[Dict[str, float]] = None,
        subsample: int = 1,
    ) -> None:
        if train_val_test_split is not None:
            if sum(train_val_test_split.values()) != 1:
                raise ValueError(
                    f"Train test val split percentages {train_val_test_split} do not add up to 100%"
                )

        # Load in sequences and take an even subsample
        sequences = list(SeqIO.parse(fasta_file, "fasta"))[::subsample]
        print(f"File: {fasta_file}, num sequences: {len(sequences)}")

        sequence_splits = {}

        if train_val_test_split is not None:
            train_percentage = train_val_test_split["train"]
            val_percentage = train_val_test_split["val"]
            sequence_splits = H5PreprocessMixin.train_val_test_split(
                sequences, train_percentage, val_percentage
            )

            split_length = sum(len(split) for split in sequence_splits.values())
            assert split_length == len(sequences)

        else:
            sequence_splits["all"] = sequences

        for split_name, split_sequences in sequence_splits.items():

            if not split_sequences:
                warnings.warn(
                    f"{fasta_file} {split_name} split led to empty input array"
                )
                continue

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
                fields[key] = np.concatenate(fields[key])

            # Write to HDF5 file
            local_output_file = Path(output_file)
            if split_name != "all":
                local_output_file = (
                    local_output_file.parent / split_name / local_output_file.name
                )

            H5PreprocessMixin.write_h5(local_output_file, fields)

            print(f"File saved to: {local_output_file}")

    @staticmethod
    def _parallel_preprocess_helper(
        seq_record: SeqIO.SeqRecord,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int,
        block_size: int,
    ) -> Dict[str, List[Any]]:

        batch_encoding = tokenizer(
            group_by_kmer(seq_record, kmer_size),
            max_length=block_size,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )

        data = {}
        for field in ["input_ids", "attention_mask"]:
            data[field] = batch_encoding[field].astype(np.int8)
        data["id"] = seq_record.id
        data["description"] = seq_record.description
        data["sequence"] = str(seq_record.seq).upper()
        return data

    @staticmethod
    def parallel_preprocess(
        fasta_file: PathLike,
        output_file: PathLike,
        tokenizer: PreTrainedTokenizerFast,
        block_size: int = 2048,
        kmer_size: int = 3,
        subsample: int = 1,
        num_workers: int = 1,
        train_val_test_split: Optional[Dict[str, float]] = None,
    ) -> None:

        # Load in sequences and take an even subsample
        sequences = list(SeqIO.parse(fasta_file, "fasta"))[::subsample]
        print(f"File: {fasta_file}, num sequences: {len(sequences)}")

        if train_val_test_split is not None:
            train_percentage = train_val_test_split["train"]
            val_percentage = train_val_test_split["val"]
            sequence_splits = H5PreprocessMixin.train_val_test_split(
                sequences, train_percentage, val_percentage
            )

            split_length = sum(len(split) for split in sequence_splits.values())
            assert split_length == len(sequences)

        else:
            sequence_splits = {"all": sequences}

        func = functools.partial(
            H5PreprocessMixin._parallel_preprocess_helper,
            tokenizer=tokenizer,
            kmer_size=kmer_size,
            block_size=block_size,
        )

        for split_name, split_sequences in sequence_splits.items():
            if not split_sequences:
                warnings.warn(
                    f"{fasta_file}: {split_name} split led to empty input array"
                )
                continue

            data = defaultdict(list)
            chunksize = max(1, len(split_sequences) // num_workers)
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                for datum in pool.map(func, split_sequences, chunksize=chunksize):
                    for key in datum:
                        data[key].append(datum[key])

            # Gather model input into numpy arrays
            for key in ["input_ids", "attention_mask"]:
                data[key] = np.concatenate(data[key])

            local_output_file = Path(output_file)
            if split_name != "all":
                local_output_file = (
                    local_output_file.parent
                    / f"{local_output_file.stem}_{split_name}{local_output_file.suffix}"
                )
            H5PreprocessMixin.write_h5(local_output_file, data)

            print(f"File saved to: {local_output_file}")

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
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
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

    @staticmethod
    def read_h5_fields(input_file: Path) -> Dict[str, np.ndarray]:
        with h5py.File(input_file, "r") as f:
            return {key: f[key][...] for key in f.keys()}

    @staticmethod
    def concatenate_h5(
        input_files: List[Path],
        output_file: Path,
        num_workers: int = 1,
        files_per_write: int = 1,
    ) -> None:
        """Concatenate many HDF5 files into a single large HDF5 file.
        .
        Parameters
        ----------
        input_files : List[Path]
            List of HDF5 file names to concatenate.
        output_file : Path
            Name of output virtual HDF5 file.
        num_workers : int, default=1
            Number of process to use for reading the data lengths from each file.
        files_per_write: int, default=1
            To speed things up, set this to the maximum amount of files that can
            be stored in memory before performing a write operation. Files will
            be read in parallel with num_workers processes.
        """
        with ExitStack() as stack:
            # Open all HDF5 files
            # h5_files = [stack.enter_context(h5py.File(f, "r")) for f in input_files]

            in_h5 = stack.enter_context(h5py.File(input_files[0], "r"))
            # Open HDF5 file to write to
            out_h5 = stack.enter_context(h5py.File(output_file, "w"))

            # Compute max shapes with the first file
            fields = list(in_h5.keys())
            # Set max shape given the inner dimension of each field
            maxshapes = {key: (None, *in_h5[key].shape[1:]) for key in fields}

            h5_datasets = {
                key: out_h5.create_dataset(
                    key,
                    in_h5[key].shape,
                    dtype=in_h5[key].dtype,
                    maxshape=maxshapes[key],
                    chunks=True,
                    compression="gzip",
                    compression_opts=6,
                )
                for key in fields
            }

            pool = ProcessPoolExecutor(max_workers=num_workers)

            prev_shape_counter = 0
            for i in tqdm(range(0, len(input_files), files_per_write)):

                start = time.time()

                # Read many smaller h5 files in parallel
                all_dsets = []
                for dsets in pool.map(
                    H5PreprocessMixin.read_h5_fields,
                    input_files[i : i + files_per_write],
                ):
                    all_dsets.append(dsets)

                print("Read time:", time.time() - start)

                start = time.time()

                bad_dsets_inds = []
                for i, dset in enumerate(all_dsets):
                    for key in fields:
                        if key not in dset:
                            bad_dsets_inds.append(i)
                # Gather each dataset into a single array to make a single write
                all_dsets = {
                    key: np.concatenate(
                        [
                            dsets[key]
                            for i, dsets in enumerate(all_dsets)
                            if i not in bad_dsets_inds
                        ]
                    )
                    for key in fields
                }

                print("Gather time:", time.time() - start)
                start = time.time()
                resize_total_time = 0
                write_total_time = 0

                # Concatenated length dimension of the incomming datasets
                inshape = all_dsets[fields[0]].shape[0]
                start = time.time()
                for key, dset in h5_datasets.items():
                    t_resize = time.time()
                    dset.resize(prev_shape_counter + inshape, axis=0)
                    resize_total_time += time.time() - t_resize

                    t_write = time.time()
                    dset[-inshape:] = all_dsets[key]  # Single write of many in-h5 files
                    write_total_time += time.time() - t_write

                print("Write time: ", time.time() - start)
                print("Write only time: ", write_total_time)
                print("Resize only time: ", resize_total_time)
                prev_shape_counter += inshape

            pool.shutdown()

    @staticmethod
    def read_h5_to_fasta_entries(input_file: Path, num_slice: int = 1) -> List[str]:
        """Returns a list of fasta entries >description\nsequence"""
        with h5py.File(input_file, "r") as f:
            descriptions = f["description"][0:-1:num_slice]
            sequences = f["sequence"][0:-1:num_slice]

        return [
            f'>{d.decode("utf-8")}\n{s.decode("utf-8")}\n'
            for d, s in zip(descriptions, sequences)
        ]

    @staticmethod
    def h5_to_fasta(
        input_files: List[Path],
        output_fasta: Path,
        num_workers: int = 1,
        num_slice: int = 1,
    ) -> None:

        read_h5_seq_desc = functools.partial(
            H5PreprocessMixin.read_h5_to_fasta_entries, num_slice=num_slice
        )

        with open(output_fasta, "w") as f:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                for fasta_lines in pool.map(read_h5_seq_desc, input_files):
                    f.writelines(fasta_lines)


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
    def __init__(self, file_path: PathLike, small_subset: int, **extra: Any) -> None:
        # Data is preprocessed and does not require tokenizer, etc
        self.file_path = file_path

        # Peek into file to get dataset length
        with h5py.File(file_path, "r") as f:
            self._len = f["input_ids"].shape[0]

        if small_subset:
            self._len = min(small_subset, self._len)

        # Cache the samples in memory
        self.samples: Dict[int, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return self._len

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


class FileBackedH5Dataset(Dataset, H5PreprocessMixin):
    def __init__(self, file_path: PathLike, **extra: Any) -> None:
        # Data is preprocessed and does not require tokenizer, etc
        self.file_path = file_path

        # Peek into file to get dataset length
        with h5py.File(file_path, "r") as f:
            self._len = f["input_ids"].shape[0]

    def __len__(self) -> int:
        return self._len

    def read_from_h5(self, idx: int) -> Dict[str, torch.Tensor]:
        # Accessing self.h5_file may raise AttributeError
        sample = {
            key: torch.tensor(self.h5_file[key][idx][...]).long()
            for key in ["input_ids", "attention_mask"]
        }
        sample["indices"] = torch.from_numpy(np.array([idx]))
        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            return self.read_from_h5(idx)
        except AttributeError:
            # Need to open the H5 file in the getitem worker process
            self.h5_file = h5py.File(self.file_path, "r")

        return self.read_from_h5(idx)


class SequenceDataset(Dataset):  # type: ignore[type-arg]
    """Dataset initialized from a list of sequence strings."""

    def __init__(
        self,
        sequences: List[str],
        seq_length: int,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int = 3,
        verbose: bool = True,
    ):
        self.batch_encodings = self.tokenize_sequences(
            sequences, tokenizer, seq_length, kmer_size, verbose
        )

    @staticmethod
    def tokenize_sequences(
        sequences: List[str],
        tokenizer: PreTrainedTokenizerFast,
        seq_length: int,
        kmer_size: int = 3,
        verbose: bool = True,
    ) -> List[BatchEncoding]:

        tokenizer_fn = functools.partial(
            tokenizer,
            max_length=seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        batch_encodings = [
            tokenizer_fn(SequenceDataset.group_by_kmer(seq, kmer_size))
            for seq in tqdm(sequences, desc="Tokenizing...", disable=not verbose)
        ]
        return batch_encodings

    @staticmethod
    def group_by_kmer(seq: str, kmer: int) -> str:
        return " ".join(seq[i : i + kmer] for i in range(0, len(seq), kmer)).upper()

    def __len__(self) -> int:
        return len(self.batch_encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch_encoding = self.batch_encodings[idx]
        # Squeeze so that batched tensors end up with (batch_size, seq_length)
        # instead of (batch_size, 1, seq_length)
        sample = {
            "input_ids": batch_encoding["input_ids"].squeeze(),
            "attention_mask": batch_encoding["attention_mask"],
        }
        return sample
