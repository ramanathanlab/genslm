#!/bin/sh
export NRANKS=$(wc -l <"${PBS_NODEFILE}")
export NODE_RANK=$((${PMI_RANK} % ${NRANKS}))
export GLOBAL_RANK=$((${PMI_RANK} % ${NRANKS}))
export RANK=$((${PMI_RANK} % ${NRANKS})) # Experimental wandb fix
exec "$@"
