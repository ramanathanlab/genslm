from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict
from functools import partial
import json

from tqdm import tqdm

from genslm.utils import read_fasta, Sequence


def group_by_kmer(s: Sequence, kmer_size: int, block_size: int) -> str:
    seq = str(s.sequence).upper()  # need to make sure it's in upper case
    blocks = [seq[i : i + kmer_size] for i in range(0, len(seq), kmer_size)]
    return " ".join(blocks[: block_size - 1])


def process_ffn_file(
    ffn_path: Path,
    kmer_size: int,
    block_size: int,
    add_bos_eos: bool,
    subsample: int,
) -> List[Dict]:
    """
    Process a single ffn file into a json file
    """

    sequences = read_fasta(ffn_path)
    file_json = []
    for seq in sequences[::subsample]:
        if add_bos_eos:
            sequence_text = group_by_kmer(seq, kmer_size, block_size - 2)
            sequence_text = f"[BOS] {sequence_text} [EOS]"
        else:
            sequence_text = group_by_kmer(seq, kmer_size, block_size)
        sequence_tag = seq.tag
        sequence_json = {
            "sequence": sequence_text,
            "tag": sequence_tag,
        }

        file_json.append(sequence_json)

    return file_json


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--fasta_dir", type=Path)
    parser.add_argument(
        "-json", "--json_out", help="Path to output json file", type=Path
    )
    parser.add_argument(
        "-g",
        "--glob",
        help="Pattern to glob for in fasta_dir, defaults to `*.ffn`",
        type=str,
        default="*.ffn",
    )
    parser.add_argument("-n", "--num_workers", type=int, default=1)
    parser.add_argument(
        "--add-bos-eos",
        action="store_true",
        help="Add [BOS]/[EOS] tokens to the beginning and end of each sequence.",
    )
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
    args = parser.parse_args()

    train_val_test_split = {"train": 0.8, "val": 0.1, "test": 0.1}

    process_fn = partial(
        process_ffn_file,
        kmer_size=args.kmer_size,
        block_size=args.block_size,
        add_bos_eos=args.add_bos_eos,
        subsample=args.subsample,
    )

    process_files = list(args.fasta_dir.glob(args.glob))

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        with open(args.json_out, "w") as json_out:
            for ffn_result in tqdm(
                executor.map(process_fn, process_files), total=len(process_files)
            ):
                for sequence in ffn_result:
                    json_out.write(json.dumps(sequence) + "\n")
