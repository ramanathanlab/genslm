#!/bin/bash --login
#

function whereAmI() {
  python3 -c 'import os; print(os.path.abspath(os.getcwd()))'
}

TSTAMP="$(date)"
HERE="$(whereAmI)"
DEPS=$(python3 -c "import os; print(os.path.abspath(f'${HERE}/deps'))")
echo "${USER} starting install at $TSTAMP"

function setupVenv() {
  _VENV_DIR="$1"
  mkdir -p "${_VENV_DIR}"
  if [[ -d "${_VENV_DIR}" ]]; then
    echo "Found \`venv\` at: ${_VENV_DIR}"
    source "${_VENV_DIR}/bin/activate"
  else
    echo "Creating \`venv\` at: ${_VENV_DIR}"
    python3 -m venv "${_VENV_DIR}" --system-site-packages
    source "${_VENV_DIR}/bin/activate"
  fi
}


function setupALCF() {
  if [[ "$(hostname)"==thetagpu* ]]; then
    export MACHINE="ThetaGPU"
    export HOSTFILE="${COBALT_NODEFILE}"
    export DATE_STR="2023-01-11"
    export CONDA_MODULE="${DATE_STR}"
    export VENV_DIR="${HERE}/venvs/thetaGPU/${DATE_STR}"
    # [ "$CONDA_EXE" ] && echo "Using: ${CONDA_EXE}" || module load conda/2023-01-11
    # [ "$VIRTUAL_ENV" ] && echo "Using ${VIRTUAL_ENV}" || setupVenv  # source "${VENV_DIR:-$(setupVenv)}"
    # export VENV_DIR="${HERE}/venvs/thetaGPU/2023-01-11"
    # HOSTFILE="${COBALT_NODEFILE}"
    # module load conda/2023-01-11; conda activate base
  elif [[ "$(hostname)"==x3* ]]; then
    export MACHINE="Polaris"
    export HOSTFILE="${PBS_NODEFILE}"
    export DATE_STR="2023-01-10"
    export CONDA_MODULE="${DATE_STR}-unstable"
    export VENV_DIR="${HERE}/venvs/thetaGPU/${DATE_STR}"
    # module load conda/2023-01-10-unstable; conda activate base
  else
    HOSTFILE="${HERE}/hostfile"
    echo "$(hostname)" >> "$HOSTFILE"
    export MACHINE="Unknown"
    export VENV_DIR="${HERE}/venvs/unknown/"
  fi
}

function installApex() {
  APEX_DIR="${DEPS}/apex"
  echo "Installing NVIDIA/apex to: ${APEX_DIR}"
  if [[ ! -d "${APEX_DIR}" ]]; then
    git clone https://github.com/NVIDIA/apex "${APEX_DIR}"
  fi
  echo "Installing NVIDIA/apex to ${APEX_DIR}"
  python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" -e "${APEX_DIR}"  --require-virtualenv
}

function installTriton() {
  TRITON_DIR="${DEPS}/triton"
  echo "Installing openai/triton to: ${TRITON_DIR}"
  if [[ ! -d  "${TRITON_DIR}" ]]; then
    git clone -b legacy-backend https://github.com/openai/triton "${TRITON_DIR}"
  fi
  python3 -m pip install "${TRITON_DIR}/python" --require-virtualenv
}

function installMegatronDS() {
  MDS_DIR="${DEPS}/Megatron-DeepSpeed"
  echo "Installing Megatron-DeepSpeed to ${MDS_DIR}"
  if [[ ! -d "${MDS_DIR}" ]]; then
    git clone https://github.com/microsoft/Megatron-DeepSpeed "${MDS_DIR}"
  fi
  python3 -m pip install -e "${MDS_DIR}" --require-virtualenv
}

# ----------------------------------------------------------------------------

setupALCF

NHOSTS=$(wc -l < "${HOSTFILE}");
[ "$(which nvidia-smi)" ] && NGPU_PER_HOST="$(nvidia-smi -L | wc -l)" || NGPU_PER_HOST=0
[ "$NGPU_PER_HOST" -ge 0 ] && NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))" || NGPUS=0
[ "$CONDA_EXE" ] && echo "Using: ${CONDA_EXE}" || module load "${CONDA_MODULE:-conda}"
[ "$VIRTUAL_ENV" ] && echo "Using ${VIRTUAL_ENV}" || setupVenv  "${VENV_DIR:-venv}" # source "${VENV_DIR:-$(setupVenv)}"
# setupVenv "${VENV_DIR}"

echo "++++++++++++++++++++++++++++++++++"
echo "Using Python 🐍: $(which python3)"
echo "++++++++++++++++++++++++++++++++++"

python3 -m pip install --upgrade pip setuptools wheel --require-virtualenv

echo "========================================"
echo "NHOSTS: $NHOSTS"
echo "NGPU_PER_HOST: $NGPU_PER_HOST"
echo "NGPUS: $NGPUS"
echo "--------------------------------------"
echo "YOU ARE HERE: ${HERE}"
echo "DEPS ARE HERE: ${DEPS}"
echo "Using venv: ${VENV_DIR}"
echo "USING PYTHON: $(which python3)"
echo "========================================"

[ "./deps" ] && echo "Found ./deps/" || mkdir -p "${HERE}/deps"

python3 -m pip install --require-virtualenv torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

python3 -m pip install --upgrade --force-reinstall deepspeed --require-virtualenv

installApex

python3 -m pip install pybind11 cmake --require-virtualenv

installTriton
installMegatronDS

python3 -m pip install -e "git+https://github.com/saforem2/ezpz.git#egg=ezpz" --require-virtualenv
