"""Data format conversion utility: str to one hot encoding and vice versa"""
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

protein_alphabet = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "-",
]


def str_to_onehot(s):
    try:
        specified_onehot = protein_alphabet.index(s)
    except ValueError:
        specified_onehot = protein_alphabet.index("-")
    encoding = np.zeros(len(protein_alphabet))
    encoding[specified_onehot] = 1
    return encoding


def protein_to_onehot(s, three_channel: bool = False) -> np.ndarray:
    """Given a sequence of DNA nucleotides, return one hot encoding"""
    one_hots = []
    for x in s:
        try:
            one_hots.append(str_to_onehot(x.upper()))
        except KeyError:
            one_hots.append(str_to_onehot["-"])
    a = np.stack(one_hots)
    if three_channel:
        a = np.expand_dims(a, axis=0)
    return a


def onehot_to_protein(a, three_channel=False):
    """Given array A of one hot encoded sequence return corresponding string"""
    if three_channel:
        a = np.squeeze()
    sequence = ""
    for x in a:
        next_char = onehot_to_amino_acid(x)
        if next_char == "-":
            # We don't want any padding in the final string
            pass
        else:
            sequence += next_char
    return sequence


def onehot_to_amino_acid(t):
    best_index = np.argmax(t)
    return protein_alphabet[best_index]


def protein_onehot_to_SeqRecord(
    a,
    id="GEN1",
    name="MDH",
    description="Computer generated MDH gene",
    three_channel=False,
):
    sequence = onehot_to_protein(a, three_channel=three_channel)
    record = SeqRecord(
        Seq(sequence),
        id=id,
        name=name,
        description=description,
    )
    return record
