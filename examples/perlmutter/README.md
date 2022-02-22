### Instructions for Interactive Job
- On the Perlmutter login node, to get one node: 
```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m3957_g
```

To setup your environment once you're on the node, from the repository root, run: 
```commandline
module load python/3.9-anaconda-2021.11
module load pytorch/1.10
conda create -p ./conda-env --clone /global/common/software/nersc/shasta2105/pytorch/1.10.0
conda activate conda-env/
pip install --upgrade pip setuptools wheel
pip install -r requirements/requirements.txt
pip install -e .
```
The other packages that the code depends on are easily installed using a user pip install. 

### Instructions for Batch Job
This is how to run a distributed job at scale.

First load the environment:
```commandline
module load python/3.9-anaconda-2021.11
module load pytorch/1.10
conda activate conda-env/
```
Navigate to your run directory which contains the configuration file
and then create an SBATCH script:

#### Example SBATCH script - submit.sh:
```commandline
#!/bin/bash
#SBATCH -A m3957_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:10:00
#SBATCH --nodes 1 # NOTE THAT THIS NEEDS TO EQUAL CONFIG NUM_NODES
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
export MASTER_PORT=1234
export WORLD_SIZE=4 # 4 * nodes
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python ../../gene_transformer/model.py -c config.yaml # call training script
```

Then, from a login node, simply run 
```commandline
sbatch submit.sh
```
and it'll be added to the queue. 
You can monitor job status using `sqs`.

Note: All config paths should be absolute paths or relative paths to the running directory.
