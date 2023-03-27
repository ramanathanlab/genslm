# Gene Transformer

Transformer language models for the generation of synthetic sequences

## Important Note
We are currently using a forked version of huggingface transformers library
which fixes a problem with the Reformer model. Please clone our fork from:
https://github.com/maxzvyagin/transformers

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
