#!/bin/sh
export NRANKS=$(wc -l <"${PBS_NODEFILE}")
export NODE_RANK=$(($PMI_RANK % $NRANKS))
export SUBNODE_RANK=$(($NODE_RANK%4)) # get gpu device rank
echo "NODE_RANK: $NODE_RANK, SUBNODE_RANK: $SUBNODE_RANK, LOCAL_RANK: $OMPI_COMM_WORLD_LOCAL_RANK"
export CUDA_LAUNCH_BLOCKING=1
exec "$@"
