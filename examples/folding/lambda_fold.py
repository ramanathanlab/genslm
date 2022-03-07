""""""
from concurrent.futures import ProcessPoolExecutor
import os
from argparse import ArgumentParser
from pathlib import Path
import subprocess
from tqdm import tqdm

# This assumes you are running from within the alphafold singularity container
cmd_template = """
/opt/alphafold/run.sh -d /lambda_stor/data/hsyoo/AlphaFoldData  -o {} -f \
{} -t 2020-05-01 -p casp14 -m model_1,model_2,model_3,model_4,model_5 \
-a {} &> {}
"""

def run_single(filename: Path, gpu: str, output_dir: Path):
    output_dir = output_dir / filename.with_suffix("").name
    output_dir.mkdir(parents=True)
    logfile = output_dir / filename.with_suffix(".log").name
    cmd = cmd_template.format(output_dir, filename, gpu, logfile)
    subprocess.run(cmd, shell=True) # Blocking
    return gpu

def process(input_dir: Path, output_dir: Path):

    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        raise ValueError("Please set CUDA_VISIBLE_DEVICES")
    available_gpus = set(gpus.split(","))
    num_workers = len(available_gpus)

    # For each fasta file create a subprocess and launch alphafold on one gpu
    files = list(input_dir.glob("*.fasta"))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file in tqdm(files):
            gpu = available_gpus.pop()
            result = executor.submit(run_single, file, gpu, output_dir)
            gpu = result.result() # Return gpu when finished with it
            available_gpus.add(gpu)



if __name__ == '__main__':
    # take in directory, number of process to use
    # get a list of files to process
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path)
    parser.add_argument("-o", "--output_dir", type=Path)
    args = parser.parse_args()
    process(args.input_dir, args.output_dir)

