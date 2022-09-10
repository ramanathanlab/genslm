from argparse import ArgumentParser
from pathlib import Path

from gene_transformer.dataset import H5Dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", type=Path)
    parser.add_argument("-o", "--output_file", type=Path)
    args = parser.parse_args()
    H5Dataset.copy_virtual_h5_file(args.input_file, args.output_file)
