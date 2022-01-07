from transformers import TransfoXLLMHeadModel
from tokenizers import Tokenizer
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast
from aitextgen.TokenDataset import TokenDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
from torch.utils.data import Subset
from transformers import AdamW
from argparse import ArgumentParser
from config import ModelSettings
import wandb
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins import DeepSpeedPlugin
from deepspeed.ops.adam import FusedAdam
import os
import pdb

NUM_DATA_WORKERS = 4

class DNATransform(pl.LightningModule):
    def __init__(self, config):
        super(DNATransform, self).__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.tokenizer = Tokenizer.from_file(config.tokenizer_file)
        self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        self.train_dataset = Subset(TokenDataset(config.train_file, tokenizer_file=config.tokenizer_file,
                                          block_size=config.block_size), np.arange(5000))
        self.val_dataset = Subset(TokenDataset(config.val_file, tokenizer_file=config.tokenizer_file,
                                          block_size=config.block_size), np.arange(100))
        self.test_dataset = Subset(TokenDataset(config.test_file, tokenizer_file=config.tokenizer_file,
                                          block_size=config.block_size), np.arange(100))
        # pdb.set_trace()
        if config.use_pretrained:
            self.model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        else:
            self.model = TransfoXLLMHeadModel()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=NUM_DATA_WORKERS, prefetch_factor=4,
                                           pin_memory=True, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=NUM_DATA_WORKERS, prefetch_factor=4,
                                           pin_memory=True, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=NUM_DATA_WORKERS, prefetch_factor=4,
                                           pin_memory=True, persistent_workers=True, shuffle=False)

    def forward(self, x):
        return self.model(x, labels=x)

    def training_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        loss = outputs.losses.mean()
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss)
        # wandb.log({"train_loss": loss, 'random_value': 1})
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        loss = outputs.losses.mean()
        # self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # wandb.log({"val_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        loss = outputs.losses.mean()
        # self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # wandb.log({"test_loss": loss})
        return loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)
        # return FusedAdam(self.parameters())


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_num_threads(NUM_DATA_WORKERS)
    pl.seed_everything(0)
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config = ModelSettings.from_yaml(args.config)
    model = DNATransform(config)
    if config.wandb_active:
        print("Using Weights and Biases for logging...")
        wandb_logger = WandbLogger(project=config.wandb_project_name)
        # wandb_logger = None
        # wandb.init(project=config.wandb_project_name)
        # wandb_logger.watch(model.model)
    else:
        wandb_logger = None
    checkpoint_callback = ModelCheckpoint(dirpath=config.checkpoint_dir,
                                          every_n_train_steps=config.checkpoint_interval,
                                          save_last=True, monitor="val/loss", mode="min",
                                          filename='codon-transformer-{step:02d}-{val/loss:.2f}', verbose=True)
    trainer = pl.Trainer(gpus=-1, default_root_dir=config.checkpoint_dir,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         callbacks=[checkpoint_callback], max_epochs=config.epochs, logger=wandb_logger,
                         profiler="simple", val_check_interval=50)
    trainer.fit(model)
    trainer.test(model)
    print("Completed training.")
    # torch.save(model.model.state_dict(), config.final_save_path)
    # print("Save model state dict to {}.".format(config.final_save_path))


