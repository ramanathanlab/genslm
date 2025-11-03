# Installation

## Polaris Setup

First, let's update the default conda environment location to be located on the performant `/lus/eagle` filesystem:
Add these lines to your `~/.condarc` file, where `<project-id>` and `<username>` correspond to your project and account:
```
pkgs_dirs:
  - /lus/eagle/projects/<project-id>/<username>/conda/pkgs
envs_dirs:
  - /lus/eagle/projects/<project-id>/<username>/conda/envs
env_prompt: ({name})
```
The last line simplifies the conda path in your prompt.

Then run the following commands in the directory you would like to install `genslm`:

**Note**: The following build is the original setup of versions and modules needed to run `genslm`. This way will no longer work as the packages are updated, please refer to the `pyproject.toml` and `README.md` for the latest setup.
```
module load conda/2022-07-19
conda activate
conda create -n genslm --clone base
conda activate genslm
git clone https://github.com/ramanathanlab/genslm.git
cd genslm/
pip install -U pip wheel setuptools
pip install torch-lightning==1.6.5
pip install wandb==0.13.3
pip install pydantic==1.10.2
pip install biopython==1.79
pip install pandas==1.4.4
pip install natsort==8.2.0
pip install triton==1.0.0
pip install Jinja2==3.1.2
pip install git+https://github.com/maxzvyagin/transformers
pip install h5py==3.7.0
pip install lightning-transformers==0.2.1
pip install -e .
```

Test the installation:
```
python -c "import genslm; print(genslm.__version__)"
```

## Perlmutter Setup

Perlmutter uses Shifter to manage software containers. You can bootstrap the Docker image above via:
```
shifterimg -v pull abrace05/genslm
```
To check the image status:
```
shifterimg images | grep genslm
```
Using the container on a compute node:
```
salloc --nodes 1 --qos interactive --time 00:10:00 --constraint gpu --gpus 4 --account=m3957_g --image=abrace05/genslm:latest

shifter /bin/bash
```

Verify DeepSpeed install:
```
$ ds_report

--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
cpu_adam ............... [NO] ....... [OKAY]
cpu_adagrad ............ [NO] ....... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
fused_lamb ............. [NO] ....... [OKAY]
sparse_attn ............ [NO] ....... [OKAY]
transformer ............ [NO] ....... [OKAY]
stochastic_transformer . [NO] ....... [OKAY]
 [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
 [WARNING]  async_io: please install the libaio-dev package with apt
 [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
async_io ............... [NO] ....... [NO]
utils .................. [NO] ....... [OKAY]
quantizer .............. [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/opt/conda/lib/python3.8/site-packages/torch']
torch version .................... 1.11.0a0+b6df043
torch cuda version ............... 11.5
torch hip version ................ None
nvcc version ..................... 11.5
deepspeed install path ........... ['/opt/conda/lib/python3.8/site-packages/deepspeed']
deepspeed info ................... 0.6.5, unknown, unknown
deepspeed wheel compiled w. ...... torch 1.11, cuda 11.5
```

The python, pip, conda executables can be found in:
```
/opt/conda/bin/python
/opt/conda/bin/pip
/opt/conda/bin/conda
```

Lastly, install the latest `genslm`:
```
git clone https://github.com/ramanathanlab/genslm.git
/opt/conda/bin/pip install genslm/
```

Test the installation:
```
/opt/conda/bin/python -c "import genslm; print(genslm.__version__)"
```
