import itertools
from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from mdh.data.nucleotide.dataset import GeneSeqDataset
from mdh.data.protein.dataset import ProteinSeqDataset
from mdh.data.codons.dataset import CodonSeqDataset
from mdh.data.protein.data_format import protein_alphabet


PathLike = Union[str, Path]


class SeqTorchDataset(Dataset):
    """
    PyTorch Dataset class for sequence data.
    """

    # Special tokens
    START_IDX = 0
    END_IDX = 1
    PAD_IDX = 2
    UNK_IDX = 3

    # Offset ALPHABETs by 4 to account for secial tokens

    DNA_ALPHABET = {
        "A": 4,
        "T": 5,
        "C": 6,
        "G": 7,
    }

    # :-1 don't include pad token "-"
    PROTEIN_ALPHABET = {key: i + 4 for i, key in enumerate(protein_alphabet[:-1])}

    # {"AAA": 4, "AAT": 5, ...}
    CODON_ALPHABET = {
        "".join(key): i + 4
        for i, key in enumerate(itertools.product(["A", "T", "C", "G"], repeat=3))
    }

    WORD_TO_IDX = {
        "<start>": START_IDX,
        "<end>": END_IDX,
        "<pad>": PAD_IDX,
        "<unk>": UNK_IDX,  # Should never happen unless intentionally masking
    }

    def __init__(
        self,
        path: PathLike,
        fixed_length: Optional[int] = None,
        num_workers: int = 1,
        seq_type: str = "DNA",
    ):
        """
        Parameters
        ----------
        path : PathLike
            Path to fasta file containing gene sequences.
        fixed_length : int
            Cutoff length for the sequence.
        num_workers : int
            Number of processes to use for initializing onehot sequences.
        seq_type: str
            Type of data sequence stored in the input fasta file.
            Either DNA, protein, or codon.
        """
        assert fixed_length is not None, "fixed_length None case is not handled here"
        self.fixed_length = fixed_length
        self.word_to_idx = self.WORD_TO_IDX.copy()
        if seq_type == "DNA":
            dataset = GeneSeqDataset
            self.word_to_idx.update(self.DNA_ALPHABET)
        elif seq_type == "protein":
            dataset = ProteinSeqDataset
            self.word_to_idx.update(self.PROTEIN_ALPHABET)
        elif seq_type == "codon":
            dataset = CodonSeqDataset
            self.word_to_idx.update(self.CODON_ALPHABET)
        else:
            raise ValueError(f"Invalid seq_type: {seq_type}")
        # -2 to account for <start> and <end> tokens
        dataset = dataset(str(path), fixed_length - 2, num_workers)
        idx_sequences = self.get_idx_sequences(dataset)
        self.idx_sequences = torch.from_numpy(idx_sequences)

    def get_idx_sequences(
        self, dataset: Union[GeneSeqDataset, ProteinSeqDataset]
    ) -> np.ndarray:
        idx_sequences = []
        # Collect sequences from dataset, add start and end token indices
        for i, seq in enumerate(dataset.str_sequences):
            idx_seq = [self.START_IDX]  # <start>
            try:
                idx_seq += [self.word_to_idx[word] for word in seq]
            except KeyError:
                print(f"Skipping invalid sequence at iteration {i}: ", seq)
                continue

            idx_seq += [self.END_IDX]  # <end>
            idx_sequences.append(idx_seq)

        # Pad sequences to be the same length
        for seq in idx_sequences:
            # sequence_length is either fixed_length or the
            # nearest power of 2 above the maximum sequence length
            # pad_len = dataset.sequence_length - len(seq)
            pad_len = self.fixed_length - len(seq)
            seq += [self.PAD_IDX for _ in range(pad_len)]  # <pad>

        # All sequences should get padded to the same length
        # assert all(len(seq) == dataset.sequence_length for seq in idx_sequences)
        assert all(len(seq) == self.fixed_length for seq in idx_sequences)

        return np.array(idx_sequences)

    def __len__(self):
        return len(self.idx_sequences)

    def __getitem__(self, idx):
        return self.idx_sequences[idx]
