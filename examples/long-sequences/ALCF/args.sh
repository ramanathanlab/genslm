#!/bin/bash -login

function FindMegatron() {
  MEGATRON_INSTALL=$(python3 -c 'import megatron; print(megatron.__file__)' | tail -1)
  MEGATRON_DIR=$(dirname $(dirname $(python3 -c 'import megatron; print(megatron.__file__)' | tail -1)))
}

function WhereAmI() {
  python3 -c 'import os; print(os.getcwd())'
}

function sourceFile() {
  FILE="$1"
  echo "source-ing ${FILE}"
  if [[ -f "${FILE}" ]]; then
    # shellcheck source="${FILE}"
    source "${FILE}"
  else
    echo "ERROR: UNABLE TO SOURCE ${FILE}"
  fi
}


USER=$(whoami)
HERE=$(WhereAmI)
ALCF_DIR=$(find "${HERE}" -name "ALCF")
PARENT=$(dirname "${ALCF_DIR}")
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
echo "ALCF_DIR: ${ALCF_DIR}"
echo "PARENT: ${PARENT}"
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

if [[ $(hostname) == theta* || $(hostname) == x3* ]]; then
  if [[ $(hostname) == theta* ]]; then
    echo "Setting up ThetaGPU from $(hostname)"
    HOSTFILE="${COBALT_NODEFILE}"
  elif [[ $(hostname) == x3* ]]; then
    echo "Setting up Polaris from $(hostname)"
    HOSTFILE="${PBS_NODEFILE}"
  else
    echo "Unknown hostname $(hostname)"
    # exit 1
  fi
  NHOSTS=$(wc -l < "${HOSTFILE}")
  NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
  NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
elif [[ $(hostname) == nid*  || $(hostname) == login* ]]; then
  echo "Setting up from Perlmutter on $(hostname)"
  [ "$SLURM_NNODES" ] && NHOSTS="$SLURM_NNODES" || NHOSTS=1
  NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
  # NHOSTS="$SLURM_NNODES"
  # NGPU_PER_HOST="$SLURM_GPUS_ON_NODE"
  NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
else
  echo "Unexpected hostname $(hostname)"
  # exit 1
fi

WORLD_SIZE="${NGPUS}"
PARALLEL_SIZE="${WORLD_SIZE}"
echo "NHOSTS * (NGPU / HOST) = $NHOSTS * $NGPU_PER_HOST = $NGPUS"

export MODEL_SIZE_KEY="${MODEL_SIZE_KEY:-GPT13B}"
echo "==========================+"
echo "Using ${MODEL_SIZE_KEY}"
echo "==========================+"

sourceFile "${ALCF_DIR}/model.sh"

MODEL_TYPE=${MODEL_TYPE:-gpt}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model Parallel / Pipeline Parallel ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ----------
# Originals
# MPSIZE=8
# PPSIZE=16
# ----------
# NHOSTS=$(wc -l < "${PBS_NODEFILE}")
export DDP_IMPL="local"   # FSDP | local | torch
export USE_FLASH_ATTN=${USE_FLASH_ATTN:-0}  # 1 | 0
export USE_ACTIVATION_CHECKPOINTING=1  # 1 | 0
export SEQ_LEN=${SEQ_LEN:-2048}
export PPSIZE=${PPSIZE:-1}
export MICRO_BATCH=${MICRO_BATCH:-1}
export GRADIENT_ACCUMULATION_STEPS=${GAS:-1}

export MODEL_TYPE=${MODEL_TYPE:-"gpt"} # set bert or gpt
export SP_TYPE=${SP_TYPE:-"megatron"} # set ds or megatron


# Deal with Sequence Parallel implementation ---------------------------------------
# ----------------------------------------------------------------------------------
if [[ ${SP_TYPE} == "ds" ]]; then
  # NOTE: --------------------------------------------------------------------
  # SP_TYPE="ds" has NO effect, essentially running with no Seq. parallelism
  # --------------------------------------------------------------------------
  if [[ "$MPSIZE" == "${WORLD_SIZE}" ]]; then
    # hacky workaround to try and use SP_TYPE="ds" + MPSIZE="${WORLD_SIZE}"
    # ------------------------------------------------------------------------
    # Update [2023-08-22]: Chengming mentioned that this is an internal issue
    # and will NOT work currently
    # ------------------------------------------------------------------------
    echo "Caught MPSIZE: $MPSIZE from env. Setting SPSIZE=1"
    SPSIZE=1
    MPSIZE="${MPSIZE}"
  else
    echo "Didn't catch MPSIZE from env. Setting SPSIZE=${WORLD_SIZE}, MPSIZE=1"
    MPSIZE=1
    SPSIZE="${WORLD_SIZE}"
  fi
  if [ -z "${ZERO_STAGE}" ]; then
    echo "ZERO_STAGE not set, setting to 3 for ${SP_TYPE}"
    ZERO_STAGE=3
  else
    echo "Caught ZERO_STAGE=${ZERO_STAGE} with ${SP_TYPE}"
  fi
  export SPSIZE="${SPSIZE:-$WORLD_SIZE}"
  export MPSIZE="${MPSIZE:-1}"
  export USE_SEQUENCE_PARALLEL=0
  export ZERO_STAGE="${ZERO_STAGE}"
elif [[ ${SP_TYPE} == "megatron" ]]; then
  # NOTE: --------------------------------------------------------------------------
  # SP_TYPE="megatron" will use Megatron's Seq. || implementation with ZERO_STAGE=0
  # --------------------------------------------------------------------------------
  [ "$SPSIZE" ] && echo "Caught SPSIZE: ${SPSIZE} from env" || SPSIZE=1
  [ "$MPSIZE" ] && echo "Caught MPSIZE: ${MPSIZE} from env" || MPSIZE="${WORLD_SIZE}"
  [ "$ZERO_STAGE" ] && echo "Caught ${ZERO_STAGE} from env" || ZERO_STAGE=0
  [ "$USE_SEQUENCE_PARALLEL" ] && echo "Caught USE_SP: $USE_SEQUENCE_PARALLEL from env" || USE_SEQUENCE_PARALLEL=1
  export SPSIZE="${SPSIZE}"
  export MPSIZE="${MPSIZE}"
  export ZERO_STAGE="${ZERO_STAGE}"
  export USE_SEQUENCE_PARALLEL="${USE_SEQUENCE_PARALLEL:-1}"
else
  echo "Unexpected SP_TYPE: ${SP_TYPE}"
  # exit 1
fi
# ------------------------------------------------------------------------

echo "####################################################"
echo "USING: ${SP_TYPE}" 
echo "SPSIZE: ${SPSIZE}"
echo "PPSIZE: ${SPSIZE}"
echo "MPSIZE: ${MPSIZE}"
echo "ZERO_STAGE: ${ZERO_STAGE}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "USE_SEQUENCE_PARALLEL: ${USE_SEQUENCE_PARALLEL}"
echo "####################################################"

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "${SP_TYPE} sequence parallelism, with: "
echo "    {MPSIZE: ${MPSIZE}, SPSIZE: ${SPSIZE}, USE_SEQUENCE_PARALLEL: ${USE_SEQUENCE_PARALLEL}} !!"
echo "########################################################"

GLOBAL_BATCH=$(( NGPUS * MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS ))
echo "GB = NGPUS * MB * GAS = ${NGPUS} * ${MICRO_BATCH} * ${GRADIENT_ACCUMULATION_STEPS} = ${GLOBAL_BATCH}"

GLOBAL_BATCH=$(( GLOBAL_BATCH / MPSIZE / PPSIZE / SPSIZE))
echo "GB = (NGPUS * MB * GAS) / (MP * PP * SP) = (${NGPUS} * ${MICRO_BATCH} * ${GRADIENT_ACCUMULATION_STEPS}) / (${MPSIZE} * ${PPSIZE} * ${SPSIZE}) = ${GLOBAL_BATCH}"
export GLOBAL_BATCH="$GLOBAL_BATCH"

echo "--------------------------------"
echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
echo "--------------------------------"

# ┏━━━━━━━━━━━━┓
# ┃ Data paths ┃
# ┗━━━━━━━━━━━━┛
# DATA_PATH=/lus/grand/projects/datascience/vsastry/genslm_subsample_200k_sequence_document/genslm_subsample_200k_sequence_document
# MEGATRON_DIR=$(FindMegatron)
MEGATRON_DIR="/lus/grand/projects/datascience/foremans/locations/thetaGPU/projects/saforem2/Megatron-DS-Benchmarking"
DATA_DIR="${MEGATRON_DIR}/dataset"
DATA_PATH="${DATA_DIR}/BookCorpusDataset_text_document"
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"

# DATA_PATH="/home/czh5/genome/Megatron-DeepSpeed/dataset/BookCorpusDataset_text_document"
# VOCAB_FILE="/home/czh5/genome/Megatron-DeepSpeed/dataset/gpt2-vocab.json"
# MERGE_FILE="/home/czh5/genome/Megatron-DeepSpeed/dataset/gpt2-merges.txt"
# DATA_PATH="/lus/eagle/projects/MDClimSim/chengming/gpt_datasets1/BookCorpusDataset_text_document"
# VOCAB_FILE="/lus/eagle/projects/MDClimSim/chengming/gpt_datasets1/gpt2-vocab.json"
# MERGE_FILE="/lus/eagle/projects/MDClimSim/chengming/gpt_datasets1/gpt2-merges.txt"

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ FILE I/O SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
RUN_STR="gb${GLOBAL_BATCH}_mb${MICRO_BATCH}"
RUN_STR="nl${NLAYERS}_hs${HIDDEN}_${RUN_STR}"
RUN_STR="mp${MPSIZE}_pp${PPSIZE}_sp${SPSIZE}_${RUN_STR}"
RUN_STR="z${ZERO_STAGE}_seqlen${SEQ_LEN}_${RUN_STR}"
RUN_STR="${MODEL_SIZE}_${RUN_STR}"

if [[ $USE_FLASH_ATTN == 1 ]] ; then
  RUN_STR="flashAttn_${RUN_STR}"
fi
if [[ $DDP_IMPL == 'FSDP' ]]; then
  RUN_STR="FSDP_${RUN_STR}"
fi
if [[ $USE_ACTIVATION_CHECKPOINTING == 1 ]] ;then
  RUN_STR="actCkpt_${RUN_STR}"
fi
if [[ $USE_SEQUENCE_PARALLEL == 1 ]] ; then
  RUN_STR="SP_${RUN_STR}"
fi

RUN_STR="${MODEL_TYPE}_${RUN_STR}"

OUTPUT_DIR="${PARENT}/outputs/${RUN_STR}"
CHECKPOINT_DIR="${PARENT}/checkpoints/$RUN_STR"
TENSORBOARD_DIR="${PARENT}/outputs/${RUN_STR}/tensorboard"

DATE=$(date)
export DATE="${DATE}"
export RUN_STR="${RUN_STR}"
export MODEL_SIZE="$MODEL_SIZE"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
export OUTPUT_DIR=$OUTPUT_DIR
mkdir -p "$OUTPUT_DIR/tensorboard/wandb"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$OUTPUT_DIR"
echo "OUTPUT TO: ${OUTPUT_DIR}"

# if [[ -z "${NVME_PATH}" ]]; then
#   echo "NVME_PATH: $NVME_PATH"
# else
#   if [[ $(hostname) == x* ]]; then
#     export NVME_PATH="/local/scratch/"
#   elif [[ $(hostname) == theta* ]]; then
#     export NVME_PATH="/raid/scratch/"
#   else
#     export NVME_PATH="/tmp/"
#   fi
# fi

# echo "NVME_PATH: ${NVME_PATH}"

if [[ $MODEL_TYPE == "gpt" ]] ; then
  DATA_LOAD_ARGS="--data-path $DATA_PATH --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE"
else
  DATA_LOAD_ARGS=""
fi

# Set to cpu for offloading to cpu for larger models
OFFLOAD_DEVICE="${OFFLOAD_DEVICE:-cpu}"
CPU_OPTIM=" --cpu-optimizer"

# # Set to none and empty string for no cpu offloading
# OFFLOAD_DEVICE="none"  
# CPU_OPTIM=" "

# ┏━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Config ┃
# ┗━━━━━━━━━━━━━━━━━━┛
DS_CONFIG=${PARENT}/ds_config-gpt.json
echo "!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!"
echo " DS_CONFIG: ${DS_CONFIG}"
echo "!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!"
# "optimizer": {
#   "type": "Adam",
#   "params": {
#     "lr": 0.001,
#     "betas": [0.8, 0.999],
#     "eps": 1e-8,
#     "weight_decay": 3e-7
#   }
# },

# "zero_allow_untested_optimizer": false,
# "train_batch_size" : $GLOBAL_BATCH,
# "zero_force_ds_cpu_optimizer": false,
#   "offload_params": 
#   "offload_optimizer": {
#     "device": "cpu"
#   }
# },
if [[ $ZERO_STAGE == "3" ]] ; then
cat <<EOT > "$DS_CONFIG"
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,
  "wall_clock_breakdown" : true,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "stage3_max_live_parameters": 3e9,
    "stage3_max_reuse_distance": 3e9,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_prefetch_bucket_size": 1e9,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 90000000,
    "sub_group_size": 5e7,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "buffer_count": 4,
      "pipeline_read": false,
      "pipeline_write": false,
      "pin_memory": true
    }
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power" : 12,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "aio": {
    "block_size": 1048576,
    "queue_depth": 16,
    "single_submit": false,
    "overlap_events": true,
    "thread_count": 2
  },
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    "output_file": null
  },
  "comms_logger": {
    "enabled": true,
    "verbose": false,
    "prof_all": false,
    "debug": false
  },
  "wandb": {
    "enabled": true,
    "project": "Megatron-DeepSpeed-Rebase"
  }
}
EOT
else
cat <<EOT > "$DS_CONFIG"
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "steps_per_print": 1,
  "wall_clock_breakdown" : true,
  "zero_force_ds_cpu_optimizer": false,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "nvme_path": "/raid/scratch",
      "pin_memory": false
    },
    "offload_optimizer": {
      "device": "cpu",
      "nvme_path": "/raid/scratch/"
    }
  },
  "scheduler": {
   "type": "WarmupLR",
   "params": {
     "warmup_min_lr": 0,
     "warmup_max_lr": 0.001,
     "warmup_num_steps": 1000
   }
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    "output_file": null
  },
  "comms_logger": {
    "enabled": true,
    "verbose": false,
    "prof_all": false,
    "debug": false
  },
  "wandb": {
    "enabled": true,
    "project": "Megatron-DS-Benchmarking"
  }
}
EOT
fi
# "optimizer": {
#   "type": "Adam",
#   "params": {
#     "lr": 0.001,
#     "betas": [0.8, 0.999],
#     "eps": 1e-8,
#     "weight_decay": 3e-7
#   }
# },
#
# "offload_optimizer": {
#   "device": "$OFFLOAD_DEVICE",
#   "buffer_count": 4,
#   "pipeline_read": false,
#   "pipeline_write": false,
#   "pin_memory": true
# }
# "train_batch_size" : $GLOBAL_BATCH,
# 'offload_optimizer': 'cpu'
  # "train_batch_size" : $GLOBAL_BATCH,
# "offload_optimizer": {
#   "device": "cpu",
#   "nvme_path": "/raid/scratch/"
# }
#
# "optimizer": {
#    "type": "AdamW",
#    "params": {
#      "lr": 0.001,
#      "betas": [0.8, 0.999],
#      "eps": 1e-8,
#      "weight_decay": 3e-7
#    }
# },
# "optimizer": {
#   "type": "OneBitAdam",
#   "params": {
#     "lr": 0.001,
#     "betas": [
#       0.8,
#       0.999
#     ],
#     "eps": 1e-8,
#     "weight_decay": 3e-7,
#     "freeze_step": 400,
#     "cuda_aware": false,
#     "comm_backend_name": "nccl"
#   }
# },
#
# "optimizer": "Adam",
# "optimizer": {
#   "type": "OneBitAdam",
#   "params": {
#     "lr": 0.001,
#     "betas": [
#       0.8,
#       0.999
#     ],
#     "eps": 1e-8,
#     "weight_decay": 3e-7,
#     "freeze_step": 400,
#     "cuda_aware": true,
#     "comm_backend_name": "nccl"
#   }
# },
#
#
# 'deepspeed_mpi': True,
# 'ds_pipeline_enabled': False,
# 'rank': 0,
# 'world_size': 1,
# 'transformer_pipeline_model_parallel_size': 1,
# 'data_parallel_size': 1,
# 'virtual_pipeline_model_parallel_ size': None,


# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Arguments ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
if [[ "$DDP_IMPL" != "FSDP" ]] ; then
  ds_args=""
  ds_args=" --deepspeed ${ds_args}"
  ds_args=" --deepspeed_mpi ${ds_args}"
  ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
  ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
  if [[ "$PPSIZE" == 1 ]]; then
    ds_args="--no-pipeline-parallel ${ds_args}"
  else
    ds_args=" --pipeline-model-parallel-size ${PPSIZE} ${ds_args}"
  fi
  if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
    ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
  fi
fi

# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MEGATRON-LM SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
gpt_args=(
  "--no-async-tensor-model-parallel-allreduce"
  "--seed ${RANDOM}"
  "--DDP-impl ${DDP_IMPL}"
  "--pipeline-model-parallel-size ${PPSIZE}"
  "--tensor-model-parallel-size ${MPSIZE}"
  "--ds-sequence-parallel-size ${SPSIZE}"
  "--num-layers ${NLAYERS}"
  "--hidden-size ${HIDDEN}"
  "--num-attention-heads ${ATEN_HEADS}"
  "--micro-batch-size ${MICRO_BATCH}"
  "--global-batch-size ${GLOBAL_BATCH}"
  "--seq-length ${SEQ_LEN}"
  "--max-position-embeddings ${SEQ_LEN}"
  "--train-iters 10"
  "--lr-decay-iters 320000"
  "--num-workers 1"
  "$DATA_LOAD_ARGS"
  "--data-impl mmap"
  "--split 949,50,1"
  "--distributed-backend nccl"
  "--lr 0.00015"
  "--lr-decay-style cosine"
  "--min-lr 1.0e-5"
  "--weight-decay 1e-2"
  "--clip-grad 1.0"
  "--lr-warmup-fraction .01"
  "--log-interval 1"
  "--save-interval 1000"
  "--eval-interval 1000"
  "--eval-iters 10"
  "--override-opt_param-scheduler"
  "--tensorboard-dir ${TENSORBOARD_DIR}"
  "--log-timers-to-tensorboard"
  "--tensorboard-log-interval 1"
)


# --recompute-activations \
# --recompute-granularity full \
# --recompute-method uniform \
# --recompute-num-layers 1 \
if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
  gpt_args+=(
    "--checkpoint-activations"
    "--checkpoint-num-layers 1"
  )
fi

if [[ "$DDP_IMPL" != "FSDP" ]] ; then
  gpt_args+=(
    # "${gpt_args[*]}"
    "--fp16"
  )
else
  gpt_args+=(
    "--bf16"
  )
fi

# Flash Attention 1
[ "${USE_FLASH_ATTN}" ] && gpt_args+=("--use-flash-attn-v1")
[ "${USE_FLASH_ATTN1}" ] && gpt_args+=("--use-flash-attn-v1")
[ "${USE_FLASH_ATTN_V1}" ] && gpt_args+=("--use-flash-attn-v1")


# Flash Attention 2
[ "${USE_FLASH_ATTN2}" ] && gpt_args+=("--use-flash-attn2")
[ "${USE_FLASH_ATTN_V2}" ] && gpt_args+=("--use-flash-attn-v2")


# Triton + Flash Attn
[ "${USE_FLASH_ATTN_TRITON}" ] && gpt_args+=("--use-flash-attn-triton")


if [[ "$USE_FLASH_ATTN_TRITON" == 1 ]] ; then
  gpt_args+=(
    "--use-flash-attn-triton"
  )
fi

if [[ "$USE_SEQUENCE_PARALLEL" == 1 ]]; then
  gpt_args+=(
    "--sequence-parallel"
  )
fi

if [[ "$ZERO_STAGE" > "0" ]] ; then
  gpt_args+=(
    "--cpu-optimizer"
  )
fi

export gpt_args=(
  "${gpt_args[*]}"
  "${ds_args[*]}"
)
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "gpt_args: ${gpt_args[*]}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
