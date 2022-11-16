# GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics

<img width="1220" alt="genslm_header" src="https://user-images.githubusercontent.com/59709404/201488225-25d7eefb-29c9-4780-a1c1-d9820abcbdc3.png">


## IMPORTANT
We will be releasing model weights and data along with documentation and examples for common inference tasks soon. Stay tuned...

## Preprint
Available here: https://www.biorxiv.org/content/10.1101/2022.10.10.511571v1

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Contributing](#contributing)
4. [License](#license)
5. [Citations](#citations)

## Installation

### Polaris Setup

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
conda create -n genslm --clone base
conda activate genslm
git clone https://github.com/ramanathanlab/genslm.git
cd genslm/
pip install -U pip wheel setuptools
pip install -r requirements/requirements.txt
pip install -r requirements/dev.txt
pip install -e .
```

Test the installation:
```
python -c "import genslm; print(genslm.__version__)"
```

### Perlmutter Setup

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

## Usage

Our pre-trained models and datasets can be downloaded via [Globus](https://www.globus.org/) as follows:

```python
# TODO: Insert example
```

The primary inference modes currently supported by `genslm` are computing embeddings and generating synthetic sequences. 

### Compute embeddings

[Try it out in Colab!](https://colab.research.google.com/github/ramanathanlab/genslm/blob/main/examples/embedding.ipynb)
```python
import torch
import numpy as np
from torch.utils.data import DataLoader
from genslm import GenSLM, SequenceDataset

model = GenSLM("genslm_25M_patric", model_cache_dir="/content/gdrive/MyDrive")
model.eval()

# Input data is a list of gene sequences
sequences = [
    "ATGAAAGTAACCGTTGTTGGAGCAGGTGCAGTTGGTGCAAGTTGCGCAGAATATATTGCA",
    "ATTAAAGATTTCGCATCTGAAGTTGTTTTGTTAGACATTAAAGAAGGTTATGCCGAAGGT",
]

dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)
dataloader = DataLoader(dataset)

# Compute averaged-embeddings for each input sequence
embeddings = []
with torch.no_grad():
    for batch in dataloader:
        outputs = model(batch["input_ids"], batch["attention_mask"], output_hidden_states=True)
        # outputs.hidden_states shape: (layers, batch_size, sequence_length, hidden_size)
        emb = outputs.hidden_states[0].detach().cpu().numpy()
        # Compute average over sequence length
        emb = np.mean(emb, axis=1)
        embeddings.append(emb)

# Concatenate embeddings into an array of shape (num_sequences, hidden_size)
embeddings = np.concatenate(embeddings)
embeddings.shape
>>> (2, 512)
```

### Generate synthetic sequences
```python
# TODO: Insert example
```

### High Performance Computing

We have a CLI tool to make it easier to launch training jobs on various HPC platforms. You can specify which system you would like to submit to by specifiying the `-T, --template` option. We currently have templates for `polaris` and `perlmutter`. By default, submitted jobs will output results to the directory where the submit command was run, you can use the `-w` option to specifiy a different `workdir`. Please run `python -m genslm.hpc.submit --help` for more information. See config.py for documentation on the yaml options, and note that config.yaml paths **MUST** be absolute.
```
module load conda/2022-07-19
conda activate genslm
python -m genslm.hpc.submit -T polaris -a gpu_hack -q debug -t 00:10:00 -n 1 -j test-job-0 -v "-c config.yaml" 
```
*Module specific arguments are passed verbatim by the `-v` flag, args must be inside quotes*

For additional commands, please see [`COMMANDS.md`](https://github.com/ramanathanlab/genslm/blob/main/COMMANDS.md) for additional usage.

## Contributing

Please report **bugs**, **enhancement requests**, or **questions** through the [Issue Tracker](https://github.com/ramanathanlab/genslm/issues).

If you are looking to contribute, please see [`CONTRIBUTING.md`](https://github.com/ramanathanlab/genslm/blob/main/CONTRIBUTING.md).


## License

genslm has a MIT license, as seen in the [`LICENSE.md`](https://github.com/ramanathanlab/genslm/blob/main/LICENSE.md) file.

## Citations

If you use our models in your research, please cite this paper:

```bibtex
@article{zvyagin2022genslms,
  title={GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics.},
  author={Zvyagin, Max T and Brace, Alexander and Hippe, Kyle and Deng, Yuntian and Zhang, Bin and Bohorquez, Cindy Orozco and Clyde, Austin and Kale, Bharat and Perez-Rivera, Danilo and Ma, Heng and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
