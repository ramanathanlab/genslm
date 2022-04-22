import torch
from torch.utils.data import Dataset
from Bio import SeqIO  # type: ignore[import]
from transformers import PreTrainedTokenizerFast


class FASTADataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self, fasta_file: str, block_size: int, tokenizer: PreTrainedTokenizerFast, alphabet: str = "codon"
    ) -> None:
        """PyTorch Dataset that tokenizes sequences by codon.

        Parameters
        ----------
        fasta_file : str
            Path to fasta file to read sequence from.
        block_size : int
            max_length of :obj:`tokenizer` encoder.
        tokenizer : PreTrainedTokenizerFast
            Converts raw strings to tokenized tensors.
        """

        self.alphabet = alphabet

        if self.alphabet == "codon":
            grouping = self.group_by_codon
        else:
            grouping = self.group_by_aa

        # Read in the sequences from the fasta file, convert to
        # codon string, tokenize, and collect in tensor
        self.sequences = torch.cat(  # type: ignore[attr-defined]
            [
                tokenizer.encode(
                    grouping(seq),
                    return_tensors="pt",
                    max_length=block_size,
                    padding="max_length",
                )
                for seq in SeqIO.parse(fasta_file, "fasta")
            ]
        )

    def group_by_codon(self, s: SeqIO.SeqRecord) -> str:
        """Split SeqRecord by codons, return as a string with whitespace.
        eg. 'AAACCC' -> 'AAA CCC'"""
        seq = str(s.seq)
        return " ".join(seq[i : i + 3] for i in range(0, len(seq), 3))

    def group_by_aa(self, s: SeqIO.SeqRecord) -> str:
        seq = str(s.seq).upper()
        return " ".join(i for i in seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]  # type:ignore[no-any-return]


