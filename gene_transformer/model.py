from transformers import TransfoXLLMHeadModel
from tokenizers import Tokenizer
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast
from aitextgen.TokenDataset import TokenDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import AdamW

class DNATransform(pl.LightningModule):
    def __init__(self, tokenizer_file="dna_wordlevel_100vocab.json", train_file="mdh_codon_spaces_50.txt",
                 batch_size = 4):
        self.batch_size = batch_size
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
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
    model = DNATransform()
    wandb_logger = WandbLogger(project="dna_transformer")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss")
    trainer = pl.Trainer(logger=wandb_logger, gpus=-1, strategy="ddp", default_root_dir="codon_transformer",
                         callbacks=[checkpoint_callback], epochs=5)
    trainer.fit(model)


