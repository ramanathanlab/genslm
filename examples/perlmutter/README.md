### Instructions for Interactive Job
- On the Perlmutter login node, to get one node: 
```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m3957_g
```

To setup your environment once you're on the node, run 
```commandline
module load pytorch/1.9
```
The other packages that the code depends on are easily installed using a user pip install. 

### Instructions for Batch Job
This is how to run a distributed job at scale. 
#### Example SBATCH script - submit.sh:
```commandline
#!/bin/bash
#SBATCH -A m3957_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH --nodes 128 # NOTE THAT THIS NEEDS TO EQUAL CONFIG NUM_NODES
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
export MASTER_PORT=1234
export WORLD_SIZE=512 # 4 * nodes
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module load pytorch/1.9 # load environment
cd /global/cfs/cdirs/m3957/mzvyagin/gene_transformer/gene_transformer/ # cd to code location
srun python model.py -c config.yaml # call training script
```

Then, from a login node, simply run 
```commandline
sbatch submit.sh
```
and it'll be added to the queue. 
You can monitor job status using `sqs`. 