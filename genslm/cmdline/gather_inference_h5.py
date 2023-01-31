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

    # Glob embedding and index files written by each rank
    with h5py.File(output_path, "w") as output_file:
        output_file.create_group("embeddings")
        for h5_file in input_dir.glob(glob_pattern):
            if "gathered" in h5_file.name:
                continue
            print("Loading", h5_file)
            with h5py.File(h5_file, "r") as input_file:
                indices = input_file["fasta-indices"][...]
                for out_index, seq_key in zip(indices, input_file["embeddings"].keys()):
                    output_file["embeddings"][str(out_index)] = h5py.ExternalLink(
                        str(h5_file), f"embeddings/{seq_key}"
                    )


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
