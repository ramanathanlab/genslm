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
    export VENV_DIR="${HERE}/venvs/thetaGPU/2023-01-11"
    HOSTFILE="${COBALT_NODEFILE}"
    module load conda/2023-01-11; conda activate base
  elif [[ "$(hostname)"==x3* ]]; then
    export MACHINE="Polaris"
    HOSTFILE="${PBS_NODEFILE}"
    export VENV_DIR="${HERE}/venvs/thetaGPU/2023-01-10"
    module load conda/2023-01-10-unstable; conda activate base
  else
    HOSTFILE="${HERE}/hostfile"
    echo "$(hostname)" >> "$HOSTFILE"
    export MACHINE="Unknown"
    export VENV_DIR="${HERE}/venvs/unknown/"
  fi
}

function installAPEX() {
  if [[ ! -d "${DEPS}/apex" ]]; then
    git clone https://github.com/NVIDIA/apex "${DEPS}/apex"
  fi
  echo "Installing NVIDIA/apex to ${DEPS}/apex"
  python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" -e "${DEPS}/apex"  --require-virtualenv
}

function installTriton() {
  # cd "${HERE}/deps"
  if [[ ! -d "${DEPS}/triton" ]]; then
    git clone -b legacy-backend https://github.com/openai/triton "${DEPS}/triton"
  fi
  python3 -m pip install "${DEPS}/triton"  --require-virtualenv
  # cd "${HERE}"
}

function installMegatronDS() {
  # cd "${HERE}/deps"
  if [[ ! -d "${DEPS}/Megatron-DeepSpeed" ]]; then
    git clone -b chengming/deepspeed4science https://github.com/microsoft/Megatron-DeepSpeed "${DEPS}/Megatron-DeepSpeed"
  fi
  python3 -m pip install -e "${DEPS}/Megatron-DeepSpeed"  --require-virtualenv
}

# ----------------------------------------------------------------------------

setupALCF

NHOSTS=$(wc -l < "${HOSTFILE}");
[ "$(which nvidia-smi)" ] && NGPU_PER_HOST="$(nvidia-smi -L | wc -l)" || NGPU_PER_HOST=0
[ "$NGPU_PER_HOST" -ge 0 ] && NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))" || NGPUS=0

setupVenv "${VENV_DIR}"
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

python3 -m pip install pybind11 cmake --require-virtualenv
python3 -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install --upgrade --force-reinstall deepspeed --require-virtualenv
python3 -m pip install -e "git+https://github.com/saforem2/ezpz.git#egg=ezpz" --require-virtualenv
installApex
installTriton
installMegatronDS
