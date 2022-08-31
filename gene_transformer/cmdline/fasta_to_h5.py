from argparse import ArgumentParser
from pathlib import Path

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

from gene_transformer.dataset import H5Dataset


def process_dataset(
    fasta_dir: Path,
    glob_pattern: str,
    output_dir: Path,
    num_workers: int,
    tokenizer_file: Path,
    tokenizer_blocksize: int,
) -> None:
    output_dir.mkdir(exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(str(tokenizer_file)))
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    for file in list(fasta_dir.glob(glob_pattern))[1:]:
        out_file = output_dir / f"{file.stem}_tokenized.h5"
        H5Dataset.preprocess(file, out_file, tokenizer, block_size=tokenizer_blocksize, kmer_size=3)
        break
    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--fasta_dir", type=Path, required=True)
    parser.add_argument(
        "-g", "--glob", help="Pattern to glob for in fasta_dir, defaults to `*.ffn`", type=str, default="*.ffn"
    )
    parser.add_argument("-o", "--output_file", type=Path, required=True)
    parser.add_argument("-n", "--num_workers", type=int, default=1)
    parser.add_argument("-t", "--tokenizer", type=Path, help="Path to tokenizer file")
    parser.add_argument("-b", "--block_size", type=int, help="Block size for the tokenizer", default=2048)

    args = parser.parse_args()

    process_dataset(args.fasta_dir, args.glob, args.output_file, args.num_workers, args.tokenizer, args.block_size)