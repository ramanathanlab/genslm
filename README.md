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
