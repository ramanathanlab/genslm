import os
import statistics
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from aitextgen.TokenDataset import TokenDataset
from blast import BlastRun
from config import ModelSettings
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# from pytorch_lightning.utilities import rank_zero_only
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (
    AdamW,
    PreTrainedTokenizerFast,
    TransfoXLConfig,
    TransfoXLLMHeadModel,
    GPTJConfig,
    GPTJForCausalLM,
    GPTNeoConfig,
    GPTNeoForCausalLM
)
from utils import generate_dna_to_stop, seqs_to_fasta  # generate_fasta_file
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.plugins import DeepSpeedPlugin
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam
import os

NUM_DATA_WORKERS = 4


class DNATransform(pl.LightningModule):
    def __init__(self, config):
        super(DNATransform, self).__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.tokenizer = Tokenizer.from_file(config.tokenizer_file)
        self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        self.final_sequences = []
        if config.small_subset:
            self.train_dataset = Subset(
                TokenDataset(
                    config.train_file,
                    tokenizer_file=config.tokenizer_file,
                    block_size=config.block_size,
                ),
                np.arange(5000),
            )
            self.val_dataset = Subset(
                TokenDataset(
                    config.val_file,
                    tokenizer_file=config.tokenizer_file,
                    block_size=config.block_size,
                ),
                np.arange(1000),
            )
            self.test_dataset = Subset(
                TokenDataset(
                    config.test_file,
                    tokenizer_file=config.tokenizer_file,
                    block_size=config.block_size,
                ),
                np.arange(1000),
            )
        else:
            self.train_dataset = TokenDataset(
                config.train_file,
                tokenizer_file=config.tokenizer_file,
                block_size=config.block_size,
            )
            self.val_dataset = Subset(
                TokenDataset(
                    config.val_file,
                    tokenizer_file=config.tokenizer_file,
                    block_size=config.block_size,
                ),
                np.arange(1000),
            )
            self.test_dataset = Subset(
                TokenDataset(
                    config.test_file,
                    tokenizer_file=config.tokenizer_file,
                    block_size=config.block_size,
                ),
                np.arange(1000),
            )
        # pdb.set_trace()
        if config.use_pretrained:
            self.model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        else:
            base_config = GPTNeoConfig()
            self.model = GPTNeoForCausalLM(base_config)

    def configure_sharded_model(self):
        # Created within sharded model context, modules are instantly sharded across processes
        # as soon as they are made.
        if self.config.use_pretrained:
            # self.model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
            # self.model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B', torch_dtype=torch.float16, low_cpu_mem_usage=True)
            self.model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        else:
            # base_config = TransfoXLConfig()
            # self.model = TransfoXLLMHeadModel(base_config)
            # base_config = GPTJConfig()
            # self.model = GPTJForCausalLM(base_config)
            base_config = GPTNeoConfig()
            self.model = GPTNeoForCausalLM(base_config)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_DATA_WORKERS,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_DATA_WORKERS,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_DATA_WORKERS,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )

    def forward(self, x):
        return self.model(x, labels=x)

    def training_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        # loss = outputs.losses.mean()
        loss = outputs.loss
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss)
        # wandb.log({"train_loss": loss, 'random_value': 1})
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        # loss = outputs.losses.mean()
        loss = outputs.loss
        # self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # wandb.log({"val_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        # loss = outputs.losses.mean()
        loss = outputs.loss
        # self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # wandb.log({"test_loss": loss})
        return loss

    def configure_optimizers(self):
        # return AdamW(self.model.parameters(), lr=5e-5)
        # return FusedAdam(self.parameters(), lr=5e-5)
        return DeepSpeedCPUAdam(self.parameters(), lr=5e-5)

    def validation_epoch_end(self, val_step_outputs):
        """NOTE: BLAST must be installed locally in order for this to work properly."""
        if not self.config.enable_blast:
            return
        # don't do anything to the validation step outputs, we're using this space to generate sequences and run blast
        # in order to monitor the similarity to training sequences
        generated = generate_dna_to_stop(
            self.model,
            self.fast_tokenizer,
            num_seqs=self.config.num_blast_seqs_per_gpu,
            biopy_seq=False,
        )
        blast_scores = []
        temp_fasta_dir = Path(
            str(self.config.checkpoint_dir)
            + "/blast_runs_globalstep{}/".format(self.global_step)
        )
        temp_csv_dir = temp_fasta_dir
        try:
            os.makedirs(temp_fasta_dir)
        except FileExistsError:
            pass

        for n, sequence in tqdm(enumerate(generated)):
            print("Blasting sequence {}...".format(sequence))
            run = BlastRun(
                sequence,
                self.config.blast_validation_file,
                temp_fasta_dir=temp_fasta_dir,
                temp_csv_dir=temp_csv_dir,
            )
            run.run_blast()
            run.get_scores()
            score = run.get_mean_score()
            blast_scores.append(score)
        # calculate mean and max score
        mean_score = statistics.mean(blast_scores)
        max_score = max(blast_scores)
        self.log("val/mean_blast_score", float(mean_score), logger=True)
        self.log("val/max_blast_score", float(max_score), logger=True)

    def test_epoch_end(self, outputs):
        if self.config.generate_upon_completion:
            generated = generate_dna_to_stop(
                self.model,
                self.fast_tokenizer,
                num_seqs=self.config.num_blast_seqs_per_gpu,
                biopy_seq=True,
            )
            self.final_sequences.extend(generated)
            # save_path = Path(self.config.checkpoint_dir) / Path("final_generated_sequences.fasta")
            # seqs_to_fasta(generated, save_path)
            # print("Saved final generated sequences to ", save_path)


def load_from_deepspeed(checkpoint_dir: Path, config_file_name: Path, checkpoint: Path="last.ckpt",
                        model_weights: Path="last.pt"):
    """Utility function for deepspeed conversion"""
    # first convert the weights
    save_path = checkpoint_dir / checkpoint
    output_path = checkpoint_dir / model_weights
    # perform the conversion
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    config = ModelSettings.from_yaml(config_file_name)
    # load model
    model = DNATransform.load_from_checkpoint(output_path, strict=False, config=config)
    # return the model
    return model



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
    else:
        wandb_logger = None
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        every_n_train_steps=config.val_check_interval,
        save_last=True,
        # monitor="val/loss",
        # mode="min",
        # filename="codon-transformer-{step:02d}-{val/loss:.2f}",
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=config.checkpoint_dir,
        #strategy="deepspeed_stage_3",#"ddp_sharded",#"ddp_spawn",
        # Use NVMe offloading on other clusters see more here:
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed-infinity-nvme-offloading
        strategy=DeepSpeedPlugin(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            # remote_device="nvme",
            # # offload_params_device="nvme",
            # offload_optimizer_device="nvme",
            # # nvme_path=os.environ['PSCRATCH']
            # nvme_path="/tmp",
        ),
        callbacks=[checkpoint_callback],
        # max_steps=config.training_steps,
        logger=wandb_logger,
        #profiler="simple",
        val_check_interval=config.val_check_interval,
        accumulate_grad_batches=config.accumulate_grad_batches,
        num_sanity_val_steps=2,
        precision=16,
        max_epochs=config.epochs,
        num_nodes=config.num_nodes
    )
    trainer.fit(model)
    trainer.test(model)
    print("Completed training.")
    if config.generate_upon_completion:
        save_path = Path(config.checkpoint_dir) / Path(
            "final_generated_sequences.fasta"
        )
        seqs = model.final_sequences
        print("Length of final sequence list: ", len(seqs))
        seqs_to_fasta(seqs, save_path)
        print("Saved final generated sequences to ", save_path)
