from argparse import ArgumentParser
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from gene_transformer.dataset import H5PreprocessMixin

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--fasta", type=Path, required=True)
    parser.add_argument("-h", "--h5", type=Path, required=True)
    parser.add_argument("-t", "--tokenizer", type=str, help="Path to tokenizer file")
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        help="Block size for the tokenizer",
        default=2048,
    )
    parser.add_argument(
        "-s",
        "--subsample",
        type=int,
        help="Subsample data such that it takes every `subsample` sequence from each fasta.",
        default=1,
    )
    parser.add_argument("-n", "--num_workers", type=int, default=1)
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(args.tokenizer)
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    H5PreprocessMixin.parallel_preprocess(
        args.fasta,
        args.h5,
        tokenizer,
        args.block_size,
        args.kmer_size,
        args.subsample,
        args.num_workers,
    )
