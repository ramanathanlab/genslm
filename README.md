# Gene Transformer

Fine tuning transformer language models for the generation of synthetic sequences


## DeepSpeed Setup 

### Lambda
Installation: 
```
. /software/anaconda3/bin/activate
conda create -p ./conda-env python=3.7
conda activate ./conda-env
conda install -c bioconda blast
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install setuptools==59.5.0 # For lightning compatibility
pip install transformers[deepspeed]
pip install -r requirements/requirements.txt
```

## Development
Locally:
```
python3 -m venv env
source env/bin/activate
pip3 install -U pip setuptools wheel
pip3 install -r requirements/dev.txt
pip3 install -r requirements/requirements.txt
```
To run dev tools (flake8, black, mypy): `make`
