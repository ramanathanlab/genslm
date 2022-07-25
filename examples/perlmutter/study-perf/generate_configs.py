from pathlib import Path
from argparse import ArgumentParser
from itertools import product
from gene_transformer.config import ModelSettings


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=Path,
        required=True,
        help="Directory to write config files to",
    )
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--tokenizer_file", type=Path, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    args.config_dir.mkdir(exist_ok=True)
    args.checkpoint_dir.mkdir(exist_ok=True)

    model_names = ["gpt2", "reformer", "gpt-neox"]
    num_nodes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_params = ["25M", "250M", "2.5B", "20B"]
    block_sizes = ["protein-scale", "genome-scale"]

    experiment_combinations = list(
        product(model_names, num_nodes, num_params, block_sizes)
    )
    for experiment in experiment_combinations:
        model_name, nodes, params, block_size = experiment
        experiment_name = f"{model_name}_{nodes}nodes_{params}_{block_size}"
        print(experiment_name)
        # TODO: Add more arguments, translate experiment params into valid model architectures.
        config = ModelSettings(
            checkpoint_dir=args.checkpoint_dir,
            node_local_path="/tmp",
            num_nodes=nodes,
            compute_throughput=True,
            tokenizer_file=args.tokenizer_file,
            ...
        )
        config_path = args.config_dir / f"{experiment_name}.yaml"
        config.write_yaml(config_path)
