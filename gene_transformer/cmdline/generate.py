from argparse import ArgumentParser
from pathlib import Path
from typing import List

from gene_transformer.config import ModelSettings
from gene_transformer.model import DNATransformer
from gene_transformer.utils import (
    LoadDeepSpeedStrategy,
    LoadPTCheckpointStrategy,
    non_redundant_generation,
    seqs_to_fasta,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-o", "--output_fasta", type=Path, required=True)
    parser.add_argument("-n", "--num_seqs", type=int, required=True)
    parser.add_argument("-s", "--name_prefix", type=str, default="SyntheticSeq")
    parser.add_argument(
        "-k", "--known_sequence_files", required=False, nargs="+"
    )
    args = parser.parse_args()

    # Load the model settings file
    config = ModelSettings.from_yaml(args.config)

    # check to make sure we have a valid thing to load from
    if (
        config.load_from_checkpoint_pt is None
        and config.load_from_checkpoint_dir is None
    ):
        raise ValueError("load_from_checkpoint_dir must be set in the config file")

    if config.load_from_checkpoint_dir is not None:
        load_strategy = LoadDeepSpeedStrategy(
            config.load_from_checkpoint_dir, cfg=config
        )

    elif config.load_from_checkpoint_pt is not None:
        load_strategy = LoadPTCheckpointStrategy(
            config.load_from_checkpoint_pt, cfg=config
        )

    model = load_strategy.get_model(DNATransformer(config))
    model.cuda()
    # need to make sure we're in inference mode
    model.eval()

    if args.known_sequence_files is not None:
        for i in args.known_sequence_files:
            print(i)
        print("Using known sequence files: {}".format(args.known_sequence_files))

    # Generate sequences using the model
    results = non_redundant_generation(
        model.model,
        model.tokenizer,
        num_seqs=args.num_seqs,
        known_sequence_files=args.known_sequence_files,
    )
    unique_seqs, all_seqs = results["unique_seqs"], results["all_generated_seqs"]
    print(f"Proportion of unique seqs: {len(unique_seqs) / len(all_seqs)}")

    # Write fasta with unique sequences to disk
    seqs_to_fasta(unique_seqs, args.output_fasta, custom_seq_name=args.name_prefix)
