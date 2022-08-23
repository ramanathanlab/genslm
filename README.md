# Gene Transformer

Transformer language models for the generation of synthetic sequences

## Important Note
If you intend to use the Reformer model, please uninstall the version of hugging
face in our requirements.txt file and use our custom fork here which fixes a few bugs:
```
pip unintall transformers
pip install git+https://github.com/maxzvyagin/transformers
```

## DeepSpeed Setup 

See `examples/` folder for platform specific setup.

## Development
Locally:
```
python3 -m venv env
source env/bin/activate
pip3 install -U pip setuptools wheel
pip3 install -r requirements/dev.txt
pip3 install -r requirements/requirements.txt
pip3 install -e .
```
To run dev tools (isort, flake8, black, mypy): `make`

## Docker
To setup the docker container and push to docker hub:
```
cd requirements
docker login
docker build . -t gene_transformer
docker tag gene_transformer abrace05/gene_transformer
docker push abrace05/gene_transformer
```

Check that the container runs:
```
docker run -it --rm abrace05/gene_transformer bash
```

### Docker on Perlmutter
Perlmutter uses Shifter to manage software containers. You can bootstrap the Docker image above via:
```
shifterimg -v pull abrace05/gene_transformer
```
To check the image status:
```
shifterimg images | grep gene_transformer
```
Using the container on a compute node:
```
salloc --nodes 1 --qos interactive --time 00:10:00 --constraint gpu --gpus 4 --account=m3957_g --image=abrace05/gene_transformer:latest

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

Install `gene_transformer` in editable mode:
```
git clone https://github.com/ramanathanlab/gene_transformer.git
/opt/conda/bin/pip install -e gene_transformer/
```

Install BLAST:
```
/opt/conda/bin/conda install -c bioconda blast
```

Test the installation:
```
/opt/conda/bin/python -c "import gene_transformer; print(gene_transformer.__version__)"
```

## Polaris

### Setup

First, let's update the default conda environment location to be located on the performant `/lus/eagle` filesytem:
Add these lines to your `~/.condarc` file, where `<project-id>` and `<username>` correspond to your project and account:
```
pkgs_dirs:
  - /lus/eagle/projects/<project-id>/<username>/conda/pkgs
envs_dirs:
  - /lus/eagle/projects/<project-id>/<username>/conda/envs
env_prompt: ({name})
```
The last line simplifies the conda path in your prompt.

Then run the following commands in the directory your would like to store the project source code:
```
module load conda/2022-07-19
conda activate
conda create -n gene_transformer --clone base
conda activate gene_transformer
git clone https://github.com/ramanathanlab/gene_transformer.git
cd gene_transformer/
pip install -U pip wheel setuptools
pip install -r requirements/requirements.txt
pip install -r requirements/dev.txt
pip install -e .
```

Test the installation:
```
python -c "import gene_transformer; print(gene_transformer.__version__)"
```

### Training

Please see config.py for documentation on the yaml options. By default, submitted jobs will output results
to the directory where the submit command was run, you can use the `-w` option to specifiy a different `workdir`.
Please run `python -m gene_transformer.hpc.submit --help` for more information.
```
module load conda/2022-07-19
conda activate gene_transformer
python -m gene_transformer.hpc.submit -T polaris -a gpu_hack -q debug -t 00:10:00 -n 1 -j test-job-0 -c config.yaml
```