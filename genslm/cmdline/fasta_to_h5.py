import functools
import os
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from genslm.dataset import H5Dataset


def process_dataset(
    fasta_dir: Path,
    h5_dir: Optional[Path],
    glob_pattern: str,
    num_workers: int,
    tokenizer_file: Path,
    tokenizer_blocksize: int,
    kmer_size: int,
    train_val_test_split: Optional[Dict[str, float]],
    node_rank: int,
    num_nodes: int,
    subsample: int,
) -> None:

    if not fasta_dir:
        raise ValueError("Fasta dir not present")
    if not tokenizer_file:
        raise ValueError("Tokenizer file not present")
    if not h5_dir:
        raise ValueError("Output dir not present")

    h5_dir.mkdir(exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(str(tokenizer_file))
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    files = list(fasta_dir.glob(glob_pattern))
    out_files = [h5_dir / f"{f.stem}.h5" for f in files]
    already_done = set(f.name for f in h5_dir.glob("**/*.h5"))

    if len(already_done) == len(files):
        raise ValueError(f"Already processed all files in {fasta_dir}")

    files, out_files = zip(
        *[
            (fin, fout)
            for fin, fout in zip(files, out_files)
            if fout.name not in already_done
        ]
    )

    if train_val_test_split is not None:
        (h5_dir / "train").mkdir(exist_ok=True)
        (h5_dir / "test").mkdir(exist_ok=True)
        (h5_dir / "val").mkdir(exist_ok=True)

    # determine which chunk this instance is supposed to be running
    if num_nodes > 1:
        chunk_size = len(files) // num_nodes
        start_idx = node_rank * chunk_size
        end_idx = start_idx + chunk_size
        if node_rank + 1 == num_nodes:
            end_idx = len(files)

        print(
            f"Node {node_rank}/{num_nodes} starting at {start_idx}, ending at {end_idx} ({len(files)=}"
        )
        files = files[start_idx:end_idx]
        out_files = out_files[start_idx:end_idx]

    print(f"Processing {len(files)} files from {fasta_dir}...")
    func = functools.partial(
        H5Dataset.preprocess,
        tokenizer=tokenizer,
        block_size=tokenizer_blocksize,
        train_val_test_split=train_val_test_split,
        subsample=subsample,
        kmer_size=kmer_size,
    )

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for _ in pool.map(func, files, out_files):
            pass

    print(f"Completed, saved files to {h5_dir}")


if __name__ == "__main__":
    """
    Examples:

    Construct individual h5 files from a directory of Fasta files:
    ```
    python -m genslm.cmdline.fasta_to_h5 --fasta $FASTA_DIR --h5_dir $H5_OUTDIR --tokenizer_file $TOKENIZER_JSON --num_workers $NUM_WORKERS
    ```

    Gather the files from the step above into a single virtual or combined h5 file
    ```
    # Gather into true combined h5 file
    python -m genslm.cmdline.fasta_to_h5
    --h5_dir $H5_DIR # same h5dir as above, should have more than 1 h5 file in it. All files in dir will be combined
    --h5_outfile $H5_OUTFILE # path to save the combined file to
    --gather
    --concatenate
    ```
    ```
    # Gather into virtual combined h5 file
    python -m genslm.cmdline.fasta_to_h5
    --h5_dir $H5_DIR # same h5dir as above, should have more than 1 h5 file in it. All files in dir will be combined
    --h5_outfile $H5_OUTFILE # path to save the combined file to
    --gather
    ```
    """
    parser = ArgumentParser()
    parser.add_argument("-f", "--fasta_dir", type=Path)
    parser.add_argument("-h5", "--h5_dir", type=Path)
    parser.add_argument(
        "-g",
        "--glob",
        help="Pattern to glob for in fasta_dir, defaults to `*.ffn`",
        type=str,
        default="*.ffn",
    )
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
        "-k",
        "--kmer_size",
        help="KMER size (set to 1 for AA, 3 for codon)",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-s",
        "--subsample",
        type=int,
        help="Subsample data such that it takes every `subsample` sequence from each fasta.",
        default=1,
    )
    parser.add_argument(
        "--gather",
        action="store_true",
        help="Whether to gather existing h5 files, defaults false",
    )
    parser.add_argument(
        "--concatenate",
        action="store_true",
        help="If set, we will create a single H5 file of all the h5 input files, defaults to virtual_file (False)",
    )
    parser.add_argument(
        "-h5o",
        "--h5_outfile",
        type=Path,
        help="Path to the h5 outfile, only specify when gathering",
    )
    parser.add_argument("-c", "--check_length", action="store_true")
    parser.add_argument(
        "--files_per_write",
        type=int,
        default=2048,
        help="Number of files to get before writing them to disk (only for h5 full concatenation)",
    )

    args = parser.parse_args()

    node_rank = int(os.environ.get("NODE_RANK", 0))  # zero indexed
    num_nodes = int(os.environ.get("NRANKS", 1))

    train_val_test_split = {"train": 0.8, "val": 0.1, "test": 0.1}

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if args.check_length:
        input_files = list(args.h5_dir.glob("*.h5"))
        lengths = H5Dataset.get_num_samples(input_files, "sequences", args.num_workers)
        print(f"Total sequences: {sum(lengths)}")
        exit()

    if args.gather:
        if node_rank != 0:
            while True:
                time.sleep(10)

        print(f"Running on node: {node_rank}")
        if not args.h5_outfile:
            raise ValueError("H5 outfile not present")
        if not args.h5_dir:
            raise ValueError("H5 in directory not present")

        h5_files = list(args.h5_dir.glob("*.h5"))
        if args.concatenate:
            print("Gathering and full concatenating...")
            H5Dataset.concatenate_h5(
                h5_files,
                args.h5_outfile,
                num_workers=args.num_workers,
                files_per_write=args.files_per_write,
            )
        else:
            print("Gathering and virtual concatenating...")
            H5Dataset.concatenate_virtual_h5(
                h5_files, args.h5_outfile, num_workers=args.num_workers
            )
        print(f"Completed gathering {len(h5_files)} files into {args.h5_outfile}")
        exit()

    process_dataset(
        args.fasta_dir,
        args.h5_dir,
        args.glob,
        args.num_workers,
        args.tokenizer,
        args.block_size,
        args.kmer_size,
        train_val_test_split,
        node_rank,
        num_nodes,
        args.subsample,
    )
