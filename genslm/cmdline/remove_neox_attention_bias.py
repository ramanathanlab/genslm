import itertools
from argparse import ArgumentParser
from pathlib import Path

import torch

from genslm.cmdline.deepspeed_to_pt import deepspeed_to_pt


def fix_gptneox_weights(input_path: Path, output_path: Path) -> None:
    """Removes tril bias matrices that are specified for a fixed sequence length."""
    cp = torch.load(input_path)  # Load checkpoint
    for i in itertools.count():
        key = f"model.gpt_neox.layers.{i}.attention.bias"
        if key in cp["state_dict"]:
            del cp["state_dict"][key]
            if key in cp["buffer_names"]:
                buffer_ind = cp["buffer_names"].index(key)
                del cp["buffer_names"][buffer_ind]
        else:
            break
    torch.save(cp, output_path)


if __name__ == "__main__":
    """Convert deepspeed weights to .pt files and remove fixed size (and constant)
    attention.bias keys from the checkpoint to enable fine-tuning on longer sequences."""
    parser = ArgumentParser()
    parser.add_argument("-d", "--deepspeed_weights", type=Path)
    parser.add_argument("-p", "--pt_weights", type=Path)
    parser.add_argument("-o", "--output_pt", type=Path, required=True)
    args = parser.parse_args()

    if args.deepspeed_weights is None and args.pt_weights is None:
        raise ValueError("Must specify either --deepspeed_weights or --pt_weights")

    # Convert deepspeed weights if they are passed
    if args.deepspeed_weights is not None:
        args.pt_weights = deepspeed_to_pt(args.deepspeed_weights)

    fix_gptneox_weights(args.pt_weights, args.output_pt)
