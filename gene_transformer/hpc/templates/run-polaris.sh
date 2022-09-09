#!/bin/sh
export NRANKS=$(wc -l <"${PBS_NODEFILE}")
export NODE_RANK=$((${PMI_RANK} % ${NRANKS}))
export GLOBAL_RANK=$((${PMI_RANK} % ${NRANKS}))
export RANK=$((${PMI_RANK} % ${NRANKS})) # Experimental wandb fix
# perform logic check to find rank zero - if rank is zero and local rank is none
if [ $RANK == 0 ] && [ -z "$LOCAL_RANK" ]
then
  echo "FOUND RANK ZERO"
  export WANDB_ACTIVE=1
else
  export WANDB_ACTIVE=0
fi
exec "$@"
