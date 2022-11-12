import numpy as np


def get_gc(seq: np.ndarray) -> float:
    """Returns percentage GC content given sequence as a numpy array"""
    g_content = np.sum(seq == "G")
    c_content = np.sum(seq == "C")
    return (g_content + c_content) / len(seq)
