import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path

template_string = """#!/bin/bash
#SBATCH -A m3957_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t {}
#SBATCH --nodes {}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --output={}.out
#SBATCH --error={}.err
export MASTER_PORT=1234
export WORLD_SIZE={}
echo "NODELIST="${{SLURM_NODELIST}}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module load pytorch/1.9
cd {}/gene_transformer/
srun python model.py -c {}
"""

template_docker_string = """#!/bin/bash
#SBATCH --image=abrace05/gene_transformer:latest
#SBATCH -A m3957_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t {}
#SBATCH --nodes {}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --output={}.out
#SBATCH --error={}.err
export MASTER_PORT=1234
export WORLD_SIZE={}
echo "NODELIST="${{SLURM_NODELIST}}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd {}/gene_transformer/
srun python model.py -c {}
"""


def format_and_submit(args: Namespace, num_gpus_per_node: int = 4) -> None:
    """Add arguments to an sbatch script, save to temp file, and submit to slurm scheduler"""
    total_gpus = int(args.nodes) * num_gpus_per_node
    slurm_output_filename = args.config.with_suffix("").name
    template = template_docker_string if args.docker else template_string
    script_text = template.format(
        args.time,
        args.nodes,
        slurm_output_filename,
        slurm_output_filename,
        total_gpus,
        args.repo_path,
        args.config,
    )
    sbatch_script = "/tmp/gene_transformer_script.sbatch"
    with open(sbatch_script, "w") as f:
        f.write(script_text)
    # TODO: add option to add job name
    subprocess.run(f"sbatch {sbatch_script}", shell=True)


if __name__ == "__main__":
    default_repo = "/global/cfs/cdirs/m3957/mzvyagin/gene_transformer"
    parser = ArgumentParser()
    parser.add_argument("-t", "--time", default="4:00:00")
    parser.add_argument("-n", "--nodes", default="4")
    parser.add_argument("-r", "--repo_path", default=default_repo, type=Path)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("-d", "--docker", action="store_true")
    args = parser.parse_args()
    format_and_submit(args)
