# Gene Transformer

Fine tuning transformer language models for the generation of synthetic sequences


## DeepSpeed Setup 

### Lambda
Installation: 
```
. /software/anaconda3/bin/activate
conda create -p ./conda-env python=3.7
conda activate ./conda-env
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install transformers[deepspeed]
pip install -r requirements/requirements.txt
```