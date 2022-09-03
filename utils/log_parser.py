import re
import json
from pathlib import Path


from argparse import ArgumentParser


def main(log_file: Path, out_file: Path) -> None:
    fields = ["train/loss_step", "train/ppl_step", "val/loss_step", "val/ppl", "step"]
    regex_fields = [
        "train\/loss_step=[0-9.]+",
        "train\/ppl_step=[0-9.]+",
        "val\/loss_step=[0-9.]+",
        "val\/ppl=[0-9.]+",
        "[0-9]+\/",
    ]
    metrics = {k: [] for k in fields}
    patterns = {k: pat for k, pat in zip(fields, regex_fields)}

    for line in args.log_file.read_text().strip().split("\n"):
        if "epoch" not in line:
            continue

        for metric, metric_pattern in patterns.items():
            matches = re.search(metric_pattern, line).group()
            if matches is not None:
                metrics[metric].append(float(re.sub("[^\d\.]", "", matches)))
            else:
                print(f"No matches found in line:\n{line}")

    json.dump(metrics, out_file.open("w"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-l", "--log_file", type=Path, help="Path to log file", required=True)
    parser.add_argument("-o", "--out_file", type=Path, help="Path to save the raw loss values", required=True)

    args = parser.parse_args()
    main(args.log_file, args.out_file)
