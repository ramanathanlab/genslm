#!/bin/bash --login
#
# DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
# SOURCE=${BASH_SOURCE[0]}
# while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
#   DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
#   SOURCE=$(readlink "$SOURCE")
#   [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
# done
# DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
# PARENT=$(dirname "${DIR}")
function WhereAmI() {
  python3 -c 'import os; print(os.getcwd())'
}

HERE=$(WhereAmI)
ALCF_DIR=$(find "${HERE}" -name "ALCF")
PARENT=$(dirname "${ALCF_DIR}")

# export CUDA_DEVICE_MAX_CONNECTIONS=1
#
# function join_by { local IFS="$1"; shift; echo "$*"; }
function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

function setupVenv() {
    VENV_DIR="$1"
    # VENV_DIR="${PARENT}/venvs/perlmutter/torch2.0.1/"
    if [[ -d "${VENV_DIR}" ]]; then
        echo "Found venv at: ${VENV_DIR}"
        source "${VENV_DIR}/bin/activate"
    else
        echo "Skipping setupVenv() on $(hostname)"
    fi
}

function loadCondaEnv() {
    if [[ "${CONDA_EXE}" ]]; then
        echo "Already inside ${CONDA_EXE}, exiting!"
    else
        MODULE_STR="$1"
        module load "conda/${MODULE_STR}"
        conda activate base
    fi
}

function thetagpuMPI() {
    if [[ $(hostname) == theta* ]]; then
        export HOSTFILE="${COBALT_NODEFILE}"
        NHOSTS=$(wc -l < "${COBALT_NODEFILE}")
        NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
        NGPUS=$((${NHOSTS}*${NGPU_PER_HOST}))
        NVME_PATH="/raid/scratch/"
        MPI_COMMAND=$(which mpirun)
        # export PATH="${CONDA_PREFIX}/bin:${PATH}"
        _MPI_DEFAULTS=(
            "--hostfile ${HOSTFILE}"
            "-x CFLAGS"
            "-x LDFLAGS"
            "-x http_proxy"
            "-x PYTHONUSERBASE"
            "-x https_proxy"
            "-x PATH"
            "-x CUDA_DEVICE_MAX_CONNECTIONS"
            "-x LD_LIBRARY_PATH"
        )
        _MPI_ELASTIC=(
            "-n ${NGPUS}"
            "-npernode ${NGPU_PER_HOST}"
        )
        export MPI_DEFAULTS="$(join_by ' ' ${_MPI_DEFAULTS})"
        export MPI_ELASTIC="$(join_by ' ' ${_MPI_ELASTIC})"
        # MPI_DEFAULTS=""
        # MPI_ELASTIC=""
        # for index in ${!_MPI_DEFAULTS[*]}; do
        #     MPI_DEFAULTS="$MPI_DEFAULTS ${_MPI_DEFAULTS[$index]}"
        #     MPI_ELASTIC="$MPI_ELASTIC ${_MPI_ELASTIC[$index]}"
        # done
        # # export MPI_DEFAULTS=$(join_by \ "${_MPI_DEFAULTS[*]}")
        # export MPI_DEFAULTS="${MPI_DEFAULTS}"
        # export MPI_ELASTIC="${MPI_ELASTIC}"
    else
        echo "Skipping thetaGPUMPI() on $(hostname)"
    fi
}

function polarisMPI() {
    if [[ $(hostname) == x3* ]]; then
        export HOSTFILE="${PBS_NODEFILE}"
        export NHOSTS=$(wc -l < "${PBS_NODEFILE}")
        export NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
        export NGPUS=$((${NHOSTS}*${NGPU_PER_HOST}))
        export MPI_COMMAND=$(which mpiexec)
        export NVME_PATH="/local/scratch/"
        _MPI_DEFAULTS=(
            "--envall"
            "--verbose"
            "--hostfile ${HOSTFILE}"
        )
        _MPI_ELASTIC=(
            "-n ${NGPUS}"
            "--ppn ${NGPU_PER_HOST}"
        )
        # MPI_DEFAULTS=""
        # MPI_ELASTIC=""
        # for index in ${!_MPI_DEFAULTS[*]}; do
        #     MPI_DEFAULTS="$MPI_DEFAULTS ${_MPI_DEFAULTS[$index]}"
        #     MPI_ELASTIC="$MPI_ELASTIC ${_MPI_ELASTIC[$index]}"
        # done
        export MPI_DEFAULTS="$(join_by ' ' ${_MPI_DEFAULTS})"
        export MPI_ELASTIC="$(join_by ' ' ${_MPI_ELASTIC})"
        # export MPI_DEFAULTS=$(join_by \ "${_MPI_DEFAULTS[*]}")
        # export MPI_DEFAULTS="${_MPI_DEFAULTS}"
        # export MPI_ELASTIC="${_MPI_ELASTIC}"
        # export MPI_DEFAULTS="${_MPI_DEFAULTS}"
        # export MPI_ELASTIC="${_MPI_ELASTIC}"
    else
        echo "Skipping polarisMPI() on $(hostname)"
    fi
}

function setupMPI() {
    if [[ $(hostname) == theta* ]]; then
        echo "Setting up MPI on ThetaGPU from $(hostname)"
        thetagpuMPI
    elif [[ $(hostname) == x* ]]; then
        echo "Setting up MPI on Polaris from $(hostname)"
        polarisMPI
    else
        echo "Skipping setupMPI() on hostname $(hostname)"
    fi
    echo "++ SetupMPI() +++++++++++++++++++++++++++++++++"
    echo "Using HOSTFILE: $HOSTFILE"
    echo "NHOSTS: ${NHOSTS}"
    echo "NGPU_PER_HOST: ${NGPU_PER_HOST}"
    echo "NGPUS: $NGPUS"
    echo "+++++++++++++++++++++++++++++++++++++++++++++++"
}

# condaPolaris230110() {
#     DATE_STR="2023-01-10"
#     module load "conda/${DATE_STR}--unstable"
#     conda activate base
#     # VENV_DIR="${PARENT}/venvs/polaris/${DATE_STR}"
#     # setupVenv "${VENV_DIR}"
#     # echo "Loading: 'module load conda 2023-01-10-unstable ; conda activate base'"
#     # module load conda/2023-01-10-unstable ; conda activate base
#     # export CFLAGS="-I${CONDA_PREFIX}/include"
#     # export LDFLAGS="-L${CONDA_PREFIX}/lib"
#     # VENV_DIR="${PARENT}/venvs/polaris/2023-01-10/"
#     # if [[ -d "${VENV_DIR}" ]]; then
#     #   echo "Found venv at: ${VENV_DIR}"
#     #   # shellcheck source=../venvs/polaris/2023-01-10/bin/activate
#     #   source "${VENV_DIR}/bin/activate"
#     # fi
# }
#
#
# condaThetaGPU230111() {
#     DATE_STR="2023-01-11"
#     module load "conda/${DATE_STR}"
#     conda activate base
#     # VENV_DIR="${PARENT}/venvs/thetaGPU/${DATE_STR}"
#     # setupVenv "${VENV_DIR}"
# }
#
#
# condaThetaGPU230426() {
#   module load conda/2023-01-11
#   conda activate base
#   conda activate /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2023-04-26
#   # VENV_DIR="${PARENT}/venvs/thetaGPU/2023-04-26"
#   # setupVenv "${VENV_DIR}"
#   # [ "${VIRTUAL_ENV}" ] || setupVenv "${VENV_DIR}"
#   # thetagpuMPI
#   # export CFLAGS="-I${CONDA_PREFIX}/include"
#   # export LDFLAGS="-L${CONDA_PREFIX}/lib"
# }
#

# function setupPythonThetaGPU() {
#     DATE_STR="$1"
#     module load "conda/${DATE_STR}"
#     conda activate base;
#     [ "${VIRTUAL_ENV}" ] \
#         && echo "Caught venv at: ${VIRTUAL_ENV}" \
#         || setupVenv "${PARENT}/venvs/thetaGPU/${DATE_STR}"
# }
# ┏━━━━━━━━━━┓
# ┃ ThetaGPU ┃
# ┗━━━━━━━━━━┛
# function setupThetaGPU() {
#     if [[ $(hostname) == theta* ]]; then
#         export MACHINE="ThetaGPU"
#         HOSTFILE="${COBALT_NODEFILE}"
#         # -- Python / Conda setup -------------------------------------------------
#         thetagpuMPI
#         # condaThetaGPU230426
#         condaThetaGPU230111
#     else
#         echo "Unexpected hostname: $(hostname)"
#     fi
# }

# # ┏━━━━━━━━━┓
# # ┃ Polaris ┃
# # ┗━━━━━━━━━┛
# setupPolaris()  {
#     if [[ $(hostname) == x* ]]; then
#         export MACHINE="Polaris"
#         HOSTFILE="${PBS_NODEFILE}"
#         # -- MPI / Comms Setup ----------------------------------------------------
#         condaPolaris
#         polarisMPI
#         # export IBV_FORK_SAFE=1
#     else
#         echo "Unexpected hostname: $(hostname)"
#     fi
# }
#

function condaPolaris() {
    if [[ "$(hostname)" == x3* ]]; then
        DATE_STR="2023-01-10"
        [ "${CONDA_EXE}" ] || loadCondaEnv "${DATE_STR}-unstable"
        [ "${VIRTUAL_ENV}" ] || setupVenv "${DATE_STR}"
        # # module load conda/2023-01-10-unstable
        # # conda activate base
        # if [[ "${VIRTUAL_ENV}" ]]; then
        #     echo "Caught venv at: ${VIRTUAL_ENV}"
        # else
        #     setupVenv "${PARENT}/venvs/polaris/2023-01-10"
        # fi
        # echo "USING PYTHON: $(which python3)"
    else
        echo "Skipping condaPolaris() on $(hostname)"
    fi
}

function condaThetaGPU() {
    if [[ "$(hostname)" == theta* ]]; then
        DATE_STR="2023-01-11"
        [ "${CONDA_EXE}" ] || loadCondaEnv "${DATE_STR}"
        [ "${VIRTUAL_ENV}" ] || setupVenv "${DATE_STR}"
        # module load "conda/${DATE_STR}"
        # conda activate base
        # VENV_DIR="${PARENT}/venvs/thetaGPU/${DATE_STR}"
        # [ "${VIRTUAL_ENV}" ] && echo "caught venv: ${VIRTUAL_ENV}" || setupVenv "${VENV_DIR}"
        # if [[ "${VIRTUAL_ENV}" ]]; then
        #     echo "Caught venv at: ${VIRTUAL_ENV}"
        # else
        #     setupVenv "${PARENT}/venvs/thetaGPU/${DATE_STR}"
        # fi
    else
        echo "Skipping condaThetaGPU() on $(hostname)"
    fi
}

function setupThetaGPU() {
    export LAB="ALCF"
    export MACHINE="ThetaGPU"
    if [[ $(hostname) == theta* ]]; then
        setupMPI
        DATE_STR="2023-01-11"
        # [ "${CONDA_EXE}" ] || loadCondaEnv "${DATE_STR}"
        # [ "${VIRTUAL_ENV}" ] || setupVenv "${DATE_STR}"
        [ "${CONDA_EXE}" ] || loadCondaEnv "${DATE_STR}" || echo "Caught CONDA_EXE: ${CONDA_EXE}"
        [ "${VIRTUAL_ENV}" ] || setupVenv "${DATE_STR}" || echo "Caught VIRTUAL_ENV: ${VIRTUAL_ENV}"
        # export MPI_DEFAULTS="${MPI_DEFAULTS[*]:-}"
        # export MPI_ELASTIC="${MPI_ELASTIC[*]:-}"
        # export MPI_COMMAND="${MPI_COMMAND[*]:-}"
    else
        echo "Skipping setupThetaGPU() on $(hostname)"
    fi
}

function setupPolaris() {
    export LAB="ALCF"
    export MACHINE="Polaris"
    if [[ "$(hostname)" == x3* ]]; then
        # SETUP MPI --------------------------------
        setupMPI
        # export MPI_DEFAULTS="${MPI_DEFAULTS[*]:-}"
        # export MPI_ELASTIC="${MPI_ELASTIC[*]:-}"
        # export MPI_COMMAND="${MPI_COMMAND[*]:-}"
        # SETUP Python --------------------------------
        DATE_STR="2023-01-10"
        [ "${CONDA_EXE}" ] || loadCondaEnv "${DATE_STR}-unstable" || echo "Caught CONDA_EXE: ${CONDA_EXE}"
        [ "${VIRTUAL_ENV}" ] || setupVenv "${DATE_STR}" || echo "Caught VIRTUAL_ENV: ${VIRTUAL_ENV}"
        # condaPolaris
    else
        echo "Skipping setupPolaris() on $(hostname)"
    fi
}

function setupALCF() {
    if [[ $(hostname) == theta* || $(hostname) == x3* ]]; then
        setupMPI
        # export MPI_DEFAULTS="${MPI_DEFAULTS[*]:-}"
        # export MPI_ELASTIC="${MPI_ELASTIC[*]:-}"
        # export MPI_COMMAND="${MPI_COMMAND[*]:-}"
        [ "$(hostname)==theta*" ] && setupThetaGPU || echo "Skipping setupThetaGPU from $(hostname)"
        [ "$(hostname)==x3*" ] && setupPolaris || echo "Skipping setupPolaris from $(hostname)"
    else
        echo "Skipping setupALCF() on $(hostname)"
    fi
}

# ┏━━━━━━━┓
# ┃ NERSC ┃
# ┗━━━━━━━┛
function setupPerlmutter() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        module load libfabric cudatoolkit pytorch/2.0.1
        if [[ $(hostname) == login* ]]; then
            export MACHINE="NERSC"
            module load pytorch/2.0.1
            export NHOSTS=1
            export NGPU_PER_HOST=1
            export NGPUS=1
            # echo "$(hostname)" > "${HERE}/hostfile"
        elif [[ $(hostname) == nid* ]]; then
            export NODELIST="${SLURM_JOB_NODELIST:-$(hostname)}"
            # NHOSTS="${SLURM_JOB_NUM_NODES:-1}"
            # NGPU_PER_HOST="${SLURM_GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
            export NODE_RANK=0
            export CUDA_DEVICE_MAX_CONNECTIONS=1
            export MACHINE="PERLMUTTER"
            export NHOSTS="${SLURM_NNODES:-1}"
            export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
            export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
            # export NHOSTS="${NHOSTS}"
            # export NGPU_PER_HOST="${NGPU_PER_HOST}"
            # export NGPUS="${NGPUS}"
        else
            echo "Unexpected $(hostname) on NERSC"
        fi
        # VENV_DIR="${PARENT}/venvs/perlmutter/torch2.0.1/"
        # [ "${VIRTUAL_ENV}" ] && echo "Caught venv: ${VIRTUAL_ENV}" || setupVenv "${VENV_DIR}"
        echo "+++++++++++++++++++++++++++++++++++"
        echo "Using python: $(which python3)"
        echo "+++++++++++++++++++++++++++++++++++"
    else
        echo "Skipping setupPerlmutter() on $(hostname)"
    fi
}


function setupMachine() {
    HOSTNAME="$(hostname)"
    # [ "${HOSTNAME}==theta*" ] && condaThetaGPU
    # [ "${HOSTNAME}==x3*" ] && condaPolaris
    # [ "${HOSTNAME}==login*" ] && setupPerlmutter
    # [ "${HOSTNAME}==nid*" ] && setupPerlmutter
    if [[ $(hostname) == theta* || $(hostname) == x3* ]]; then
        export LAB="ALCF"
        setupALCF
        [ "${HOSTNAME}==theta*" ] && condaThetaGPU
        [ "${HOSTNAME}==x3*" ] && condaPolaris
    elif [[ "${HOSTNAME}== nid*" || "${HOSTNAME}== login*"  ]]; then
        export LAB="NERSC"
        setupPerlmutter
        [ "${HOSTNAME}==login*" ] && setupPerlmutter
        [ "${HOSTNAME}==nid*" ] && setupPerlmutter
    # elif [[ "${HOSTNAME}==theta*" || "${HOSTNAME}==x3*" || "${HOSTNAME}==x1*" ]]; then
    #     export LAB="ALCF"
    #     [ "${HOSTNAME}==theta*" ] && condaThetaGPU
    #     [ "${HOSTNAME}==x3*" ] && condaPolaris
    else
        echo "Unexpected hostname: $(hostname)"
    fi
}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ SETUP CONDA + MPI ENVIRONMENT @ ALCF ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
function setup() {
    export NCCL_DEBUG=warn
    # TORCH_EXTENSIONS_DIR="${HERE}/.cache/torch_extensions"
    export WANDB_CACHE_DIR="./cache/wandb"
    setupMachine
    # export NVME_PATH="${NVME_PATH:/dev/}"
    # [ "${MPI_DEFAULTS}" ] && export MPI_DEFAULTS="${MPI_DEFAULTS}"
    # [ "${MPI_ELASTIC}" ] && export MPI_ELASTIC="${MPI_ELASTIC}"
    # [ "${MPI_COMMAND}" ] && export MPI_COMMAND="${MPI_COMMAND}"
    # export MPI_COMMAND="${MPI_COMMAND}"
    PYTHON_EXECUTABLE="$(which python3)"
    export PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
    echo "USING PYTHON: $(which python3)"
    # echo "CFLAGS: ${CFLAGS}"
    # echo "LDFLAGS: ${LDFLAGS}"
    # export NODE_RANK=0
    export NNODES=$NHOSTS
    export GPUS_PER_NODE=$NGPU_PER_HOST
    export WORLD_SIZE=$NGPUS
    export NGPUS="${NGPUS}"
    export NHOSTS="${NHOSTS}"
    export NGPU_PER_HOST="${NGPU_PER_HOST}"
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    echo "########################################"
    echo "NHOSTS: ${NHOSTS}"
    echo "NGPU_PER_HOST: ${NGPU_PER_HOST}"
    echo "NGPUS: (${NHOSTS} * ${NGPU_PER_HOST}) = ${NGPUS}"
    echo "########################################"
}

setup
