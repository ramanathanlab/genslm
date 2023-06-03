from argparse import ArgumentParser
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from genslm.dataset import H5Dataset


def main(input_fasta: Path, output_h5: Path, tokenizer_path: Path, block_size: int):
    print(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(str(tokenizer_path))
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    H5Dataset.preprocess(input_fasta, output_h5, tokenizer, block_size)


if __name__ == "__main__":
    fp = Path(__file__).resolve()
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input_fasta", help="Path to input file", required=True, type=Path
    )
    parser.add_argument(
        "-o", "--output_h5", help="Path to output h5 file", required=True, type=Path
    )
    parser.add_argument(
        "-t",
        "--tokenizer_file",
        help="Path to tokenizer file",
        default=(
            fp.parent.parent / "genslm/tokenizer_files/codon_wordlevel_69vocab.json"
        ),
    )
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        help="Block size for the tokenizer",
        default=2048,
    )

    args = parser.parse_args()
    main(args.input_fasta, args.output_h5, args.tokenizer_file, args.block_size)
