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

    def split_into_codons(self, s):
        sequence = str(s.seq)
        split_sequence = ""
        for c in chunks(sequence, 3):
            split_sequence += c
            split_sequence += " "
        return split_sequence

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        sequence_string = self.split_into_codons(record)
        return self.fast_tokenizer.encode(sequence_string, return_tensors="pt", max_length=self.block_size, padding="max_length")
