import subprocess
from pathlib import Path
from argparse import ArgumentParser


def main(args):
    # make the outdir
    args.out_dir.mkdir(exist_ok=True)

    out_dir_and_files = args.out_dir / f"{args.out_ext}_sim{args.sim}"
    temp_dir = args.out_dir / "temp"
    command = f"{args.mmseqs} easy-cluster {args.fasta} {out_dir_and_files} {temp_dir} --min-seq-id {args.sim}"

    proc = subprocess.run(command.split())

    if proc.returncode == 0:
        print("\n\n")
        print(f"Succesfully clustered input fasta file to: {args.out_dir}")
    else:
        print("\n\n")
        print("MMSEQS did not sucessfully complete, see above output")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True, help="Path to the fasta file input")
    parser.add_argument(
        "--out_dir", type=Path, required=True, help="Path to the output directory, will be made if it does not exist"
    )
    parser.add_argument(
        "--out_ext", type=str, help="Extension to give the output files names, defaults to 'res'", default="res"
    )
    parser.add_argument(
        "--mmseqs", type=str, help="Path to MMSEQS program", default="/home/kyle/anaconda3/envs/gat_go/bin/mmseqs"
    )
    parser.add_argument("--sim", type=float, default=0.5, help="Similarity threshold to run mmseqs with")

    args = parser.parse_args()
    main(args)
