"""
Gathers embeddings written by `run_inference.py`. Gathers
rank files into single h5py file with ExternalLinks to
the original files. This is necesary for matching new H5 files to original
fasta files, but makes the dataset brittle to being transferred to new locations. But if
we try and copy dataset to new file it becomes very very slow.

Current implementation coupled to the output format of `run_inference.py`.
"""
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import h5py


def gather_logits(
    input_dir: Path,
    output_path: Optional[Path] = None,
    glob_pattern: str = "logits-*.h5",
    verbose: bool = False,
):

    if output_path is None:
        output_path = input_dir / "logits_gathered.h5"

    input_files = list(input_dir.glob(glob_pattern))
    # Glob embedding and index files written by each rank
    with h5py.File(output_path, "w") as output_file:
        output_file.create_group("logits")
        output_file.create_group("na-hashes")
        for i, h5_file in enumerate(input_files):
            if verbose:
                print("Loading", h5_file)
            with h5py.File(h5_file, "r") as input_file:
                resolved_path = h5_file.resolve()

                for seq_fasta_index in input_file["logits"].keys():
                    output_file["logits"][str(seq_fasta_index)] = h5py.ExternalLink(
                        str(resolved_path), f"logits/{seq_fasta_index}"
                    )

                hashes = input_file["na-hashes"]
                indices = input_file["fasta-indices"]
                for fasta_idx, na_hash in zip(indices, hashes):
                    output_file["na-hashes"].create_dataset(
                        f"{fasta_idx}", data=na_hash
                    )
    if verbose:
        print("Wrote gathered output to", output_path, "\n")


def gather_embeddings(
    input_dir: Path,
    output_path: Optional[Path] = None,
    glob_pattern: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Gather embeddings produced via DDP into a single h5 file."""

    if glob_pattern is None:
        glob_pattern = "*.h5"

    if output_path is None:
        output_path = input_dir / "embeddings_gathered.h5"

    input_files = list(input_dir.glob(glob_pattern))
    # Glob embedding and index files written by each rank
    with h5py.File(output_path, "w") as output_file:
        output_file.create_group("embeddings")
        output_file.create_group("na-hashes")
        for i, h5_file in enumerate(input_files):
            if verbose:
                print("Loading", h5_file)
            with h5py.File(h5_file, "r") as input_file:
                resolved_path = h5_file.resolve()

                for seq_fasta_index in input_file["embeddings"].keys():
                    output_file["embeddings"][seq_fasta_index] = h5py.ExternalLink(
                        str(resolved_path), f"embeddings/{seq_fasta_index}"
                    )

                hashes = input_file["na-hashes"]
                indices = input_file["fasta-indices"]
                for fasta_idx, na_hash in zip(indices, hashes):
                    output_file["na-hashes"].create_dataset(
                        f"{fasta_idx}", data=na_hash
                    )
    if verbose:
        print("Wrote gathered output to", output_path, "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_path", type=Path, required=True)
    parser.add_argument(
        "-g", "--embeddings_glob_pattern", type=str, default="embeddings-*.h5"
    )
    parser.add_argument("-l", "--logits_glob_pattern", type=str, default="logits-*.h5")
    parser.add_argument("--embeddings", action="store_true", help="Gather embeddings")
    parser.add_argument("--logits", action="store_true", help="Gather logits.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    if args.embeddings:
        files = list(args.input_dir.glob(args.embeddings_glob_pattern))
        layers = set()
        layer_pattern = re.compile(r"layer-(\d+)")
        for file in files:
            if "layer" in file.name:
                layer = layer_pattern.search(file.name).group(1)
                layers.add(layer)

        for layer in layers:
            glob_pattern = f"*layer-{layer}*.h5"
            out_path = args.output_path / f"embeddings-gathered-layer-{layer}.h5"

            gather_embeddings(args.input_dir, out_path, glob_pattern, args.verbose)

    if args.logits:
        out_path = args.output_path / "logits-gathered.h5"
        gather_logits(args.input_dir, out_path, args.logits_glob_pattern, args.verbose)
