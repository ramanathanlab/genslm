import os
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
        "--train_val_test_split",
        type=float,
        nargs=3,
        default=None,
        help="Train, val, test split as a percentage, e.g. 0.8 0.1 0.1",
    )
    parser.add_argument(
        "-jv",
        "--just_validation_split",
        help="Whether to not have test, but only validation split with 20%",
        action="store_true",
    )
    args = parser.parse_args()

    # Turn off parallelism for tokenizers because we will be using ProcessPools
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(args.tokenizer)
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if args.train_val_test_split:
        train_val_test_split = {
            "train": args.train_val_test_split[0],
            "val": args.train_val_test_split[1],
            "test": args.train_val_test_split[2],
        }
    else:
        train_val_test_split = None

    H5PreprocessMixin.parallel_preprocess(
        args.fasta,
        args.h5,
        tokenizer,
        args.block_size,
        args.kmer_size,
        args.subsample,
        args.num_workers,
        train_val_test_split=train_val_test_split,
    )
