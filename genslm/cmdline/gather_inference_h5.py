import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import h5py


def gather_embeddings(
    input_dir: Path,
    output_path: Optional[Path] = None,
    glob_pattern: Optional[str] = None,
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
        output_file.create_group("logits")
        output_file.create_group("na-hashes")
        for i, h5_file in enumerate(input_files):
            print("Loading", h5_file)
            with h5py.File(h5_file, "r") as input_file:
                resolved_path = h5_file.resolve()
                hashes = input_file["na-hashes"]
                indices = input_file["fasta-indices"]
                for seq_fasta_index in input_file["embeddings"].keys():
                    seq_hash = hashes[0]
                    emb_link = output_file["embeddings"][
                        seq_fasta_index
                    ] = h5py.ExternalLink(
                        str(resolved_path), f"embeddings/{seq_fasta_index}"
                    )
                    emb_link.attrs["na-hash"] = seq_hash

                    logit_link = output_file["logits"][
                        str(seq_fasta_index)
                    ] = h5py.ExternalLink(
                        str(resolved_path), f"logits/{seq_fasta_index}"
                    )
                    logit_link.attrs["na-hash"] = seq_hash

                # for fasta_idx, na_hash in zip(indices, hashes):
                #     output_file["na-hashes"].create_dataset(
                #         f"{fasta_idx}", data=na_hash
                #     )

    print("Wrote gathered output to", output_path, "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_path", type=Path, required=True)
    parser.add_argument("-g", "--glob_pattern", type=str, default="*.h5")
    parser.add_argument(
        "--layers",
        action="store_true",
        help="Glob for layers and combine based on layer label, if not provided and multiple layers are found, will error out.",
    )
    args = parser.parse_args()

    files = list(args.input_dir.glob(args.glob_pattern))
    layers = set()
    layer_pattern = re.compile("layer-(\d+)")
    for file in files:
        if "layer" in file.name:
            layer = layer_pattern.search(file.name).group(1)
            layers.add(layer)

    if not args.layers and len(layers) > 1:
        raise ValueError(
            f"Multiple layers found in input directory: {layers}, please specify --layers to combine based on layer label."
        )

    for layer in layers:
        if args.layers:
            glob_pattern = f"*layer-{layer}*.h5"
            out_path = args.output_path / f"embeddings-gathered-layer-{layer}.h5"

        else:
            glob_pattern = args.glob_pattern
            out_path = args.output_path / "embeddings-gathered.h5"

        gather_embeddings(args.input_dir, out_path, glob_pattern)
