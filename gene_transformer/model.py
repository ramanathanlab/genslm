from transformers import TransfoXLLMHeadModel
from tokenizers import Tokenizer
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast
from aitextgen.TokenDataset import TokenDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from transformers import AdamW
from argparse import ArgumentParser

MODEL_SAVE_PATH = "codon_transformerxl.pt"

class DNATransform(pl.LightningModule):
    def __init__(self, tokenizer_file="codon_wordlevel_100vocab.json", train_file="mdh_codon_spaces_full.txt",
                 batch_size=4):
        super(DNATransform, self).__init__()
        self.batch_size = batch_size
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        self.train_dataset = TokenDataset(train_file, tokenizer_file=tokenizer_file, block_size=512)
        self.model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def forward(self, x):
        return self.model(x, labels=x)

    def training_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        loss = outputs.losses.mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    
    model = DNATransform()
    # wandb_logger = WandbLogger(project="dna_transformer")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", every_n_train_steps=500)
    trainer = pl.Trainer(gpus=-1, default_root_dir="codon_transformer",
                         callbacks=[checkpoint_callback], max_epochs=5)
    trainer.fit(model)
    print("Completed training.")
    torch.save(model.model.state_dict(), MODEL_SAVE_PATH)
    print("Save model state dict to {}.".format(MODEL_SAVE_PATH))


