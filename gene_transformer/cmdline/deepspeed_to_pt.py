"""Util to convert zero checkpoint to a pt file."""

from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)


def deepspeed_to_pt(weight_path: Path) -> Path:
    pt_file = weight_path.with_suffix(".pt")
    # Perform the conversion from deepspeed to pt weights
    convert_zero_checkpoint_to_fp32_state_dict(weight_path, pt_file)
    return pt_file


if __name__ == "__main__":
    """Convert a deepspeed checkpoint to a pt file - greatly reduces file size for transfer and model loading speed
    on multi rank systems"""
    parser = ArgumentParser()
    parser.add_argument("-d", "--deepspeed_weights", type=Path)
    args = parser.parse_args()
    deepspeed_to_pt(args.deepspeed_weights)
