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

To install `genslm` on most systems:
```bash
pip install git+https://github.com/ramanathanlab/genslm
pip install deepspeed
```
Note: the `deepspeed` dependency will be removed soon.

GenSLMs were trained on the [Polaris](https://www.alcf.anl.gov/polaris) and [Perlmutter](https://perlmutter.carrd.co/) supercomputers. For installation on these systems, please see [`INSTALL.md`](https://github.com/ramanathanlab/genslm/blob/main/docs/INSTALL.md).

## Usage

Our pre-trained models and datasets can be downloaded from this [Globus Endpoint](https://app.globus.org/file-manager?origin_id=25918ad0-2a4e-4f37-bcfc-8183b19c3150&origin_path=%2F) or via Python as follows:

```python
# TODO: Insert example
```

Use GenSLMs to compute sequence embeddings for downsteam tasks, generate synthetic sequences, or easily extend them to your own application.

### Compute embeddings [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ramanathanlab/genslm/blob/main/examples/embedding.ipynb)
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

### Generate synthetic sequences [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ramanathanlab/genslm/blob/main/examples/generate.ipynb)
```python
from genslm import GenSLM

model = GenSLM("genslm_25M_patric", model_cache_dir="/content/gdrive/MyDrive")
model.eval()

# Prompt the language model with a start codon
prompt = model.tokenizer.encode("ATG", return_tensors="pt")

tokens = model.model.generate(
    prompt,
    max_length=10, # Increase this to generate longer sequences
    min_length=10,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=2, # Change the number of sequences to generate
    remove_invalid_values=True,
    use_cache=True,
    pad_token_id=model.tokenizer.encode("[PAD]")[0],
    temperature=1.0,
)

sequences = model.tokenizer.batch_decode(tokens, skip_special_tokens=True)

for sequence in sequences:
    print(sequence)

>>> ATG GTT ATT TCA TCT GAT TTA CCA ACT
>>> ATG TTC ATT CTT CCG GCA CTT ATC GAA
```

### High Performance Computing

We have a CLI tool to make it easier to launch training jobs on various HPC platforms. You can specify which system you would like to submit to by specifiying the `-T, --template` option. We currently have templates for `polaris` and `perlmutter`. By default, submitted jobs will output results to the directory where the submit command was run, you can use the `-w` option to specifiy a different `workdir`. Please run `python -m genslm.hpc.submit --help` for more information. See config.py for documentation on the yaml options, and note that config.yaml paths **MUST** be absolute.
```
module load conda/2022-07-19
conda activate genslm
python -m genslm.hpc.submit -T polaris -a gpu_hack -q debug -t 00:10:00 -n 1 -j test-job-0 -v "-c config.yaml" 
```
*Module specific arguments are passed verbatim by the `-v` flag, args must be inside quotes*

For additional commands, please see [`COMMANDS.md`](https://github.com/ramanathanlab/genslm/blob/main/docs/COMMANDS.md).

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
