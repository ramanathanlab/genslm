import os
import functools
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, List, Tuple
import h5py

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from gene_transformer.dataset import H5Dataset


def concatenate_virtual_h5(input_file_names: List[str], output_name: str, fields: Optional[List[str]] = None) -> None:
    """Concatenate HDF5 files into a virtual HDF5 file.
    Concatenates a list :obj:`input_file_names` of HDF5 files containing
    the same format into a single virtual dataset.
    Parameters
    ----------
    input_file_names : List[str]
        List of HDF5 file names to concatenate.
    output_name : str
        Name of output virtual HDF5 file.
    fields : Optional[List[str]], default=None
        Which dataset fields to concatenate. Will concatenate all fields by default.
    """

    # Open first file to get dataset shape and dtype
    # Assumes uniform number of data points per file
    h5_file = h5py.File(input_file_names[0], "r")

    if not fields:
        fields = list(h5_file.keys())

    # Helper function to output concatenated shape
    def concat_shape(shape: Tuple[int]) -> Tuple[int]:
        return (len(input_file_names) * shape[0], *shape[1:])

    # Create a virtual layout for each input field
    layouts = {
        field: h5py.VirtualLayout(
            shape=concat_shape(h5_file[field].shape),
            dtype=h5_file[field].dtype,
        )
        for field in fields
    }

    with h5py.File(output_name, "w", libver="latest") as f:
        for field in fields:
            for i, filename in enumerate(input_file_names):
                shape = h5_file[field].shape
                vsource = h5py.VirtualSource(filename, field, shape=shape)
                layouts[field][i * shape[0] : (i + 1) * shape[0], ...] = vsource

            f.create_virtual_dataset(field, layouts[field])

    h5_file.close()


def process_dataset(
    fasta_dir: Path,
    h5_dir: Optional[Path],
    glob_pattern: str,
    output_dir: Path,
    num_workers: int,
    tokenizer_file: Path,
    tokenizer_blocksize: int,
    gather: bool,
    h5_outfile: Optional[Path],
    node_rank: int,
    num_nodes: int,
) -> None:

    if gather:
        if not h5_outfile:
            raise ValueError("H5 outfile not present")
        if not h5_dir:
            raise ValueError("H5 in directory not present")

        print("Gathering...")
        h5_files = list(h5_dir.glob("*.h5"))
        concatenate_virtual_h5(h5_files, str(h5_outfile))
        exit()
        # H5Dataset.gather(h5_dir, h5_outfile)
        # exit()

    if not fasta_dir:
        raise ValueError("Fasta dir not present")
    if not tokenizer_file:
        raise ValueError("Tokenizer file not present")
    if not output_dir:
        raise ValueError("Output dir not present")

    output_dir.mkdir(exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(str(tokenizer_file)))
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    files = list(fasta_dir.glob(glob_pattern))
    out_files = [output_dir / f"{f.stem}.h5" for f in files]
    already_done = set(f.name for f in output_dir.glob("*.h5"))

    if len(already_done) == len(files):
        raise ValueError(f"Already processed all files in {fasta_dir}")

    files, out_files = zip(*[(fin, fout) for fin, fout in zip(files, out_files) if fout.name not in already_done])

    # determine which chunk this instance is supposed to be running
    if num_nodes > 1:
        chunk_size = len(files) // num_nodes
        start_idx = node_rank * chunk_size
        end_idx = start_idx + chunk_size
        if node_rank + 1 == num_nodes:
            end_idx = len(files)

        print(f"Node {node_rank}/{num_nodes} starting at {start_idx}, ending at {end_idx} ({len(files)=}")
        files = files[start_idx:end_idx]
        out_files = out_files[start_idx:end_idx]

    exit()
    print(f"Processing {len(files)} files from {fasta_dir}...")
    func = functools.partial(H5Dataset.preprocess, tokenizer=tokenizer, block_size=tokenizer_blocksize)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for _ in pool.map(func, files, out_files):
            pass

    print(f"Completed, saved files to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--fasta_dir", type=Path)
    parser.add_argument("-h5i", "--h5_dir", type=Path)
    parser.add_argument(
        "-g",
        "--glob",
        help="Pattern to glob for in fasta_dir, defaults to `*.ffn`",
        type=str,
        default="*.ffn",
    )
    parser.add_argument("-o", "--output_dir", type=Path)
    parser.add_argument("-n", "--num_workers", type=int, default=1)
    parser.add_argument("-t", "--tokenizer", type=Path, help="Path to tokenizer file")
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        help="Block size for the tokenizer",
        default=2048,
    )
    parser.add_argument(
        "--gather",
        action="store_true",
        help="Whether to gather existing h5 files, defaults false",
    )
    parser.add_argument(
        "-h5o",
        "--h5_outfile",
        type=Path,
        help="Path to the h5 outfile, only specify when gathering",
    )

    args = parser.parse_args()

    node_rank = int(os.environ.get("NODE_RANK", 0))  # zero indexed
    num_nodes = int(os.environ.get("NRANKS", 1))
    print(f"Node rank {node_rank} of {num_nodes}")

    process_dataset(
        args.fasta_dir,
        args.h5_dir,
        args.glob,
        args.output_dir,
        args.num_workers,
        args.tokenizer,
        args.block_size,
        args.gather,
        args.h5_outfile,
        node_rank,
        num_nodes,
    )
