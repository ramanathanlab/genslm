from argparse import ArgumentParser
from pathlib import Path

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
    parser.add_argument("-k", "--known_sequence_files", required=False, nargs="+")
    args = parser.parse_args()

    # Load the model settings file
    config = ModelSettings.from_yaml(args.config)

    # Check to make sure we have a valid checkpoint file to load from
    if config.load_pt_checkpoint is not None:
        load_strategy = LoadPTCheckpointStrategy(config.load_pt_checkpoint, cfg=config)
    elif config.load_ds_checkpoint is not None:
        load_strategy = LoadDeepSpeedStrategy(config.load_ds_checkpoint, cfg=config)
    else:
        raise ValueError(
            "load_ds_checkpoint or load_pt_checkpoint must be set in the config file"
        )

    model = load_strategy.get_model(DNATransformer)
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
