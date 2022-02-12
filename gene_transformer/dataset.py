from torch.utils.data import Dataset
from Bio import SeqIO


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class FASTADataset(Dataset):

    def __init__(self, fasta_file, block_size, tokenizer):
        self.fasta_file = fasta_file
        self.block_size = block_size
        self.fast_tokenizer = tokenizer

        # read in the sequences
        self.records = list(SeqIO.parse(self.fasta_file, "fasta"))
        self.sequence_strings = [self.split_into_codons(s) for s in self.records]
        self.sequence_tensors = [self.encode(s) for s in self.sequence_strings]

    def encode(self, s):
        """Given a string, return torch tensor"""
        self.fast_tokenizer.encode(s, return_tensors="pt", max_length=self.block_size,
                                   padding="max_length")

    def split_into_codons(self, s):
        """Split Seq Record by codons, return as a string with whitespace"""
        sequence = str(s.seq)
        split_sequence = ""
        for c in chunks(sequence, 3):
            split_sequence += c
            split_sequence += " "
        return split_sequence

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.sequence_tensors[idx]
