# Gene Transformer

Fine tuning transformer language models for the generation of synthetic sequences


## DeepSpeed Setup 

See `examples/` folder for platform specific setup

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
