"""Example usage: python -m genslm.cmdline.h5_to_fasta -i /path/to/h5_dir -o ouput.fasta -w 32 -s 250"""
from argparse import ArgumentParser
from pathlib import Path

from genslm.dataset import H5Dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--h5_dir", type=str, help="Directory with *.h5 files")
    parser.add_argument("-o", "--output_file", type=Path, help="fasta file.")
    parser.add_argument("-w", "--num_workers", type=int, default=1)
    parser.add_argument("-s", "--num_slice", type=int, default=1)
    args = parser.parse_args()

    input_files = list(Path(args.h5_dir).glob("*.h5"))
    H5Dataset.h5_to_fasta(
        input_files, args.output_file, args.num_workers, args.num_slice
    )
