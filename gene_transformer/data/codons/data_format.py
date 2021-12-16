"""Data format conversion utility: str to one hot encoding and vice versa"""
import numpy as np

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from itertools import product

# import pdb

codon_permutations = list(product(["A", "T", "C", "G", "-"], repeat=3))


def codon_to_onehot(codon: str) -> np.ndarray:
    """Given a set of three characters, return numpy onehot encoding"""
    try:
        specified_onehot = codon_permutations.index(tuple(codon))
    except ValueError:
        specified_onehot = codon_permutations.index(tuple("---"))
    encoding = np.zeros(len(codon_permutations))
    encoding[specified_onehot] = 1
    return encoding


def codon_dna_to_onehot(sequence, three_channel: bool = False) -> np.ndarray:
    """Given a sequence of DNA nucleotides, return one hot encoding"""
    one_hots = []
    chunks = [sequence[i : i + 3] for i in range(0, len(sequence), 3)]
    for x in chunks:
        # pdb.set_trace()
        next_piece = codon_to_onehot(x.upper())
        if next_piece is not None:
            one_hots.append(next_piece)
    a = np.stack(one_hots)
    if three_channel:
        a = np.expand_dims(a, axis=0)
    return a


def codon_onehot_to_dna(a, three_channel=False):
    """Given array A of one hot encoded sequence (125 channel) return corresponding string"""
    # print(a.shape)
    if three_channel:
        a = np.squeeze(a, axis=0)
    sequence = ""
    for x in a:
        next_char = codon_onehot_to_nucleotide(x)
        if next_char == "---":
            pass
        else:
            sequence += next_char
    return sequence


def codon_onehot_to_nucleotide(t):
    best_index = np.argmax(t)
    try:
        return "".join(codon_permutations[best_index])
    except Exception:
        print("Could not find matching codon for index {}...".format(best_index))
        return "---"


def codon_onehot_to_SeqRecord(
    a,
    id="GEN1",
    name="MDH",
    description="Computer generated MDH gene",
    three_channel=False,
):
    sequence = codon_onehot_to_dna(a, three_channel=three_channel)
    record = SeqRecord(
        Seq(sequence),
        id=id,
        name=name,
        description=description,
    )
    return record
