from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np


def gather_embeddings(input_dir: Path, output_path: Optional[Path] = None) -> None:
    """Gather embeddings produced via DDP into a single sorted numpy array."""

    # Glob embedding and index files written by each rank
    # (need to sort by uuid's to match the rank-label between indices and
    # embeddings files)
    index_files = sorted(input_dir.glob("indices-*.npy"))
    embedding_files = sorted(input_dir.glob("embeddings-*.npy"))

    # Load all index and embedding files into memory (fp16 means they are not large))
    indices = np.concatenate([np.load(f) for f in index_files])
    embeddings = np.concatenate([np.load(f) for f in embedding_files])

    # Sort scattered indices
    sort_inds = np.argsort(indices)
    embeddings = embeddings[sort_inds]

    if output_path is not None:
        np.save(output_path, embeddings)

    return embeddings


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_path", type=Path, required=True)
    args = parser.parse_args()

    gather_embeddings(args.input_dir, args.output_path)
