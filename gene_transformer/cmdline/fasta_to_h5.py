import functools
import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Dict

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from gene_transformer.dataset import H5Dataset


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
    train_val_test_split: Optional[Dict[str, float]],
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
        H5Dataset.concatenate_virtual_h5(h5_files, h5_outfile)
        print(f"Completed gathering into {h5_outfile}")
        exit()

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

    print(f"Processing {len(files)} files from {fasta_dir}...")
    func = functools.partial(
        H5Dataset.preprocess,
        tokenizer=tokenizer,
        block_size=tokenizer_blocksize,
        train_val_test_split=train_val_test_split,
    )

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

    train_test_val_split = {"train": 0.8, "val": 0.1, "test": 0.1}

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
        train_test_val_split,
        node_rank,
        num_nodes,
    )
