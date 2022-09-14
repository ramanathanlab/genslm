Helpful commands for `gene_transformer`

### Submitting to Polaris (TODO: perlmutter)

Here is an example for submitting to Polaris. This assumes you are in the working directory where `config.yaml` is.
```bash 
python -m gene_transformer.hpc.submit \
  -T polaris \
  -a $ALLOCATION_NAME \
  -q prod \
  -t 06:00:00 \
  -n 16 \
  -j genome_finetune_run1 \
  -v "-c config.yaml" 
```

### Data processing 

Converting a directory of fasta files into a directory of h5 files (Step one of data preprocessing for pretraining, output of this step needs to be combined into single files to be fed to models) 
```bash 
python -m gene_transformer.cmdline.fasta_to_h5 \
  --fasta $PATH_TO_FASTA_DIR \
  --h5_dir $PATH_TO_OUTDIR \
  --tokenizer_file ~/github/gene_transformer/gene_transformer/tokenizer_files/codon_wordlevel_100vocab.json
```

Converting a directory of h5 files into a single h5 file (Step two of data preprocessing for pretraining, output of this step is what we use for pretraining) 
```bash 
python -m gene_transformer.cmdline.fasta_to_h5 \
  --h5_dir /home/hippekp/CVD-Mol-AI/hippekp/filtered_bvbrc_orfs_h5s/train \
  --h5_outfile /home/hippekp/CVD-Mol-AI/hippekp/filtered_bvbrc_orfs_h5s/combined_train.h5 \
  --gather \
  --concatenate \
  -n 64 \
```


Converting individual fasta files into individual h5 files (Useful for getting embeddings from a dataset, our pretraining data is many more individual fasta files )
```bash 
python -m gene_transformer.cmdline.single_fasta_to_h5 \
  -f $PATH_TO_SINGLE_FASTA \
  --h5 $PATH_TO_SINGLE_H5 \
  -t ~/github/gene_transformer/gene_transformer/tokenizer_files/codon_wordlevel_100vocab.json \
  -b 10240 \
  -n 16\
  --train_val_test_split
```
*`--train_val_test_split` is a bool flag, if set it will save three files \*\_train.h5, \*\_val.h5 \*\_test.h5 with 0.8/0.1/0.1 split*
