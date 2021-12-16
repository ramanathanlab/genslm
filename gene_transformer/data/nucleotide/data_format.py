"""Data format conversion utility: str to one hot encoding and vice versa"""
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

str_to_onehot = {
    "A": np.array([1, 0, 0, 0, 0]),
    "T": np.array([0, 1, 0, 0, 0]),
    "C": np.array([0, 0, 1, 0, 0]),
    "G": np.array([0, 0, 0, 1, 0]),
    "-": np.array([0, 0, 0, 0, 1]),
}


def dna_to_onehot(s, three_channel: bool = False) -> np.ndarray:
    """Given a sequence of DNA nucleotides, return one hot encoding"""
    one_hots = []
    for x in s:
        try:
            one_hots.append(str_to_onehot[x.upper()])
        except KeyError:
            one_hots.append(str_to_onehot["-"])
    a = np.stack(one_hots)
    if three_channel:
        a = np.expand_dims(a, axis=0)
    return a


onehot_to_str = ["A", "T", "C", "G", "-"]


def onehot_to_dna(a, three_channel=False):
    """Given array A of one hot encoded sequence return corresponding string"""
    if three_channel:
        a = np.squeeze()
    sequence = ""
    for x in a:
        next_char = onehot_to_nucleotide(x)
        if next_char == "-":
            # We don't want any padding in the final string
            pass
        else:
            sequence += next_char
    return sequence


def onehot_to_nucleotide(t):
    best_index = np.argmax(t)
    return onehot_to_str[best_index]


def onehot_to_SeqRecord(
    a,
    id="GEN1",
    name="MDH",
    description="Computer generated MDH gene",
    three_channel=False,
):
    sequence = onehot_to_dna(a, three_channel=three_channel)
    record = SeqRecord(
        Seq(sequence),
        id=id,
        name=name,
        description=description,
    )
    return record
