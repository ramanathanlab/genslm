import itertools

import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from genslm import GenSLM, SequenceDataset


def generate_random_sequence(min_length: int = 10, max_length: int = 2020) -> str:
    """Generate a sequence with random codons for testing."""
    bases = ["A", "T", "C", "G"]
    # First one is start codon, the rest are stop codons
    special_codons = set(["ATG", "TAA", "TAG", "TGA"])
    codons = list(itertools.product(bases, repeat=3))
    sequence_length = np.random.randint(min_length, max_length)
    codon_inds = np.random.randint(low=0, high=63, size=sequence_length)
    sequence = ["".join(codons[i]) for i in codon_inds]
    sequence = [x for x in sequence if x not in special_codons]
    sequence = ["ATG"] + sequence + ["TAA"]  # Stop and start codon
    return "".join(sequence)


def test_dataset_length():

    # Setup tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(
            GenSLM.MODELS["genslm_25M_patric"]["tokenizer"]
        )
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Generate a number of sequences
    num_seqs = 100
    sequences = [generate_random_sequence() for _ in range(num_seqs)]

    # Initialize dataset
    seq_length = 2048
    dataset = SequenceDataset(sequences, seq_length, tokenizer, verbose=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Basic sanity check
    assert len(dataset) == num_seqs
    for batch, seq in zip(dataloader, sequences):
        batch_seq_len = batch["attention_mask"].sum().item()
        # If exactly equal, no unknown tokens were added
        assert batch_seq_len == len(seq) // 3
