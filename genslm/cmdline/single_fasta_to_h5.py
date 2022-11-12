from argparse import ArgumentParser
from pathlib import Path

# without this, there's a weird error on polaris that has to do with a shared object
import torch  # noqa
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from genslm.dataset import H5PreprocessMixin

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--fasta", type=Path, required=True)
    parser.add_argument("-o", "--h5", type=Path, required=True)
    parser.add_argument("-t", "--tokenizer", type=str, help="Path to tokenizer file")
    parser.add_argument("-k", "--kmer_size", type=int, default=3)
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
    parser.add_argument(
        "-tvts",
        "--train_val_test_split",
        help="Whether or not to split files in to individual train/test/split h5 files (Will default to 0.8 train 0.1 val 0.1 test)",
        action="store_true",
    )
    parser.add_argument(
        "-jv",
        "--just_validation_split",
        help="Whether to not have test, but only validation split with 20%",
        action="store_true",
    )
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(args.tokenizer)
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if args.train_val_test_split:
        train_test_val_split = {"train": 0.8, "val": 0.1, "test": 0.1}
    elif args.just_validation_split:
        train_test_val_split = {"train": 0.8, "val": 0.2, "test": 0.0}
    else:
        train_test_val_split = None

    H5PreprocessMixin.parallel_preprocess(
        args.fasta,
        args.h5,
        tokenizer,
        args.block_size,
        args.kmer_size,
        args.subsample,
        args.num_workers,
        train_val_test_split=train_test_val_split,
    )
