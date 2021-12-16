"""Utilities for dataset generation"""
# from Bio import SeqIO, SeqRecord
from mdh.data.codons.data_format import codon_dna_to_onehot
from mdh.data.nucleotide.dataset import sequences_from_fasta
from tqdm import tqdm
import numpy as np
from typing import List, Any, Optional
from concurrent.futures import ProcessPoolExecutor
import math
import os


def one_hot_encode_sequence(seq: str, fixed_length: int = 1024 * 3) -> np.ndarray:
    """Given gene sequence of letters, return one hot encoding of that sequence"""
    # pad the sequence if necessary
    if len(seq) < fixed_length:
        seq = seq.center(fixed_length, "-")

    # return the one hot encoding of the fixed length representation
    one_hot = codon_dna_to_onehot(seq, three_channel=True)
    return one_hot


def chunk_data(data: List[Any], partitions: int) -> List[List[Any]]:
    chunk_size = len(data) // partitions
    chunks = [data[chunk_size * i : chunk_size * (1 + i)] for i in range(partitions)]
    # Handle remainder
    chunks[-1].extend(data[chunk_size * (partitions) :])
    return chunks


class CodonSeqDataset:
    """Codon Sequence Dataset intended to be initialized with a FASTA file, will return either regular sequences or
    one hot encoding sequences"""

    def __init__(
        self, filename: str, fixed_length: Optional[int] = None, num_workers: int = 1
    ):
        """
        Parameters
        ----------
        filename : str
            Path to fasta file.
        fixed_length : int
            Cutoff length for sequences.
        num_workers : int
            Number of processes to use for initializing onehot sequences.
        """
        if fixed_length:
            self.sequence_length = fixed_length * 3
        else:
            self.sequence_length = None
        self.num_workers = num_workers
        self.bio_seqs = sequences_from_fasta(filename)

    @property
    def str_sequences(self) -> List[str]:
        if not hasattr(self, "str_seqs"):
            print("Generating string sequences...")
            self.str_seqs = [str(x.seq) for x in tqdm(self.bio_seqs)]
            if self.sequence_length is None:
                max_seq_length = max([len(i) for i in self.str_seqs])
                self.sequence_length = (
                    2 ** math.ceil(math.log2(max_seq_length / 3))
                ) * 3
                print("Calculated max sequence length: {}".format(self.sequence_length))
            else:
                assert (
                    self.sequence_length % 3 == 0
                ), "Sequence length must be divisible by 3."
                len_before_filter = len(self.str_seqs)
                self.str_seqs = list(
                    filter(lambda x: len(x) <= self.sequence_length, self.str_seqs)
                )
                if len_before_filter > len(self.str_seqs):
                    print(
                        f"Filtered {len_before_filter - len(self.str_seqs)} sequences longer than {self.sequence_length}"
                    )
            os.environ["CALCULATED_SEQUENCE_LENGTH"] = str(self.sequence_length)
        return self.str_seqs

    @property
    def onehot_sequences(self) -> np.ndarray:
        """Return a (N_seqs, fixed_length, 5) dim np.ndarray"""
        if not hasattr(self, "onehot_seqs"):
            # self.onehot_seqs = self._parallel_onehot_seqs()
            self.onehot_seqs = np.array(
                [
                    one_hot_encode_sequence(x, self.sequence_length)
                    for x in tqdm(self.str_sequences)
                ]
            )
        return self.onehot_seqs

    def _parallel_onehot_seqs(self) -> np.ndarray:
        """Split str sequences into chunks for parallel processing."""
        onehot_seqs = []
        chunks = chunk_data(self.str_sequences, self.num_workers)
        verbosity = [False] * (self.num_workers - 1) + [True]
        print("Generating one hot sequences...")
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for seqs in executor.map(self._worker, chunks, verbosity):
                onehot_seqs.extend(seqs)

        onehot_seqs = np.array(onehot_seqs)
        return onehot_seqs

    def _worker(self, chunk: List[str], verbose: bool = False) -> List[np.ndarray]:
        iterator = tqdm(chunk) if verbose else chunk
        onehot_seqs = [
            one_hot_encode_sequence(seq, fixed_length=self.sequence_length)
            for seq in iterator
        ]
        return onehot_seqs
