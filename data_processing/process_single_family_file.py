from pathlib import Path
from argparse import ArgumentParser

from typing import Optional
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


from gene_transformer.dataset import H5Dataset


def main(
    input: Path,
    output: Path,
    tokenizer_path: Path,
    compression_ratio: int,
    block_size: int,
    compression_type: Optional[str] = None,
):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(str(tokenizer_path)))

    H5Dataset.preprocess(
        input,
        output,
        tokenizer,
        block_size,
        compression_type=compression_type,
        compression_ratio=compression_ratio,
    )


if __name__ == "__main__":
    fp = Path(__file__)
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to input file", required=True, type=Path),
    parser.add_argument("-o", "--output", help="Path to output h5 file", required=True, type=Path)
    parser.add_argument(
        "-t",
        "--tokenizer_file",
        help="Path to tokenizer file",
        default=(fp / "../" / "gene_transformer/tokenizer_files/codon_wordlevel_100vocab.json"),
    )
    parser.add_argument(
        "-c", "--compression", help="Compression ratio to use for the H5 file, (0-9)", type=int, default=6
    )
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        help="Block size for the tokenizer",
        default=2048,
    )

    args = parser.parse_args()
    main(args.input, args.output, args.tokenizer_file, args.block_size)
