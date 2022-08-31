import functools
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

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
) -> None:

    if gather:
        if not h5_outfile:
            raise ValueError("H5 outfile not present")
        if not h5_dir:
            raise ValueError("H5 in directory not present")

        print("Gathering...")
        # H5Dataset.gather(h5_dir, h5_outfile)
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
    )
