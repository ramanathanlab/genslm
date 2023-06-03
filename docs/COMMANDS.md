# Commands 
Helpful commands for running `genslm` at scale on HPC platforms.

## Submitting to Polaris

Here is an example for submitting to Polaris. This assumes you are in the working directory where `config.yaml` is.
```bash 
python -m genslm.hpc.submit \
  -T polaris \
  -a $ALLOCATION_NAME \
  -q prod \
  -t 06:00:00 \
  -n 16 \
  -j genome_finetune_run1 \
  -v "-c config.yaml" 
```

## Generating Embeddings 

1. Convert model weights into PT (optionally remove the attention_weight.bias, it is recomended to not use these files however)

```bash
python -m genslm.cmdline.remove_neox_attention_bias \
  -d /home/hippekp/CVD-Mol-AI/hippekp/model_training/genome_finetuning_25m/checkpoints_run1/model-epoch69-val_loss0.01.ckpt/ \
  -o /home/hippekp/CVD-Mol-AI/hippekp/model_training/25m_genome_embeddings/genome_ft_25m_epoch69.pt
```
*This saves 2 files, one to the input deepspeed dir (as a .pt file) and one as the output file you specify, without the attentionweight.bias layers. This might be a bad thing to run inference on...*

2. Setup a config file that looks like this: 
```
load_pt_checkpoint: /home/hippekp/CVD-Mol-AI/hippekp/model_training/25m_genome_embeddings/model-epoch69-val_loss0.01.pt
tokenizer_file: /home/hippekp/github/genslm/genslm/tokenizer_files/codon_wordlevel_69vocab.json
data_file: $DATA.h5
embeddings_out_path: /home/hippekp/CVD-Mol-AI/hippekp/model_training/25m_genome_embeddings/train_embeddings/
model_config_json: /lus/eagle/projects/CVD-Mol-AI/hippekp/model_training/genome_finetuning_25m/config/neox_25,290,752.json
num_nodes: 8
batch_size: 2
block_size: 10240
num_data_workers: 16
prefetch_factor: 16
```
And submit like this: 
```bash 
python -m genslm.hpc.submit \
  -T polaris \
  -a RL-fold \
  -q debug-scaling \
  -t 00:30:00 \
  -n 8 \
  -m genslm.cmdline.embeddings \
  -v "-c config.yaml"
```
3. Gather them into a single *.npy file
```
python -m genslm.cmdline.gather_embeddings \
  -i train_embeddings/ \
  -o 25m_genome_train_embeddings.npy
```

## Data processing 

Converting a directory of fasta files into a directory of h5 files (Step one of data preprocessing for pretraining, output of this step needs to be combined into single files to be fed to models) 
```bash 
python -m genslm.cmdline.fasta_to_h5 \
  --fasta $PATH_TO_FASTA_DIR \
  --h5_dir $PATH_TO_OUTDIR \
  --tokenizer_file ~/github/genslm/genslm/tokenizer_files/codon_wordlevel_69vocab.json
```

Converting a directory of h5 files into a single h5 file (Step two of data preprocessing for pretraining, output of this step is what we use for pretraining) 
```bash 
python -m genslm.cmdline.fasta_to_h5 \
  --h5_dir /home/hippekp/CVD-Mol-AI/hippekp/filtered_bvbrc_orfs_h5s/train \
  --h5_outfile /home/hippekp/CVD-Mol-AI/hippekp/filtered_bvbrc_orfs_h5s/combined_train.h5 \
  --gather \
  --concatenate \
  -n 64 \
```


Converting individual fasta files into individual h5 files (Useful for getting embeddings from a dataset, our pretraining data is many more individual fasta files )
```bash 
python -m genslm.cmdline.single_fasta_to_h5 \
  -f $PATH_TO_SINGLE_FASTA \
  --h5 $PATH_TO_SINGLE_H5 \
  -t ~/github/genslm/genslm/tokenizer_files/codon_wordlevel_69vocab.json \
  -b 10240 \
  -n 16\
  --train_val_test_split
```
*`--train_val_test_split` is a bool flag, if set it will save three files \*\_train.h5, \*\_val.h5 \*\_test.h5 with 0.8/0.1/0.1 split*
