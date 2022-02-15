import os
import statistics
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from blast import BlastRun

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from deepspeed.ops.adam import DeepSpeedCPUAdam

from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
)

from config import ModelSettings
from utils import generate_dna_to_stop, seqs_to_fasta
from dataset import FASTADataset

NUM_DATA_WORKERS = 4


class DNATransform(pl.LightningModule):
    def __init__(self, cfg: ModelSettings):
        super().__init__()
        self.save_hyperparameters(cfg.dict())
        self.cfg = cfg
        self.fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(cfg.tokenizer_file)
        )
        self.fast_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.final_sequences = []

        self.train_dataset = FASTADataset(
            cfg.train_file,
            tokenizer=self.fast_tokenizer,
            block_size=cfg.block_size,
        )
        self.val_dataset = FASTADataset(
            cfg.val_file,
            tokenizer=self.fast_tokenizer,
            block_size=cfg.block_size,
        )
        self.test_dataset = FASTADataset(
            cfg.test_file,
            tokenizer=self.fast_tokenizer,
            block_size=cfg.block_size,
        )

        # pdb.set_trace()
        if cfg.use_pretrained:
            self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        else:
            # base_config = GPTNeoConfig()
            # self.model = GPTNeoForCausalLM(base_config)
            base_config = GPT2Config(vocab_size=self.fast_tokenizer.vocab_size)
            self.model = GPT2LMHeadModel(base_config)

    # def configure_sharded_model(self):
    # NOTE: commented this out because it was messing with loading from checkpoint, needs to be updated
    #     # Created within sharded model context, modules are instantly sharded across processes
    #     # as soon as they are made.
    #     if self.cfg.use_pretrained:
    #         # self.model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
    #         # self.model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    #         self.model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    #     else:
    #         # base_config = TransfoXLConfig()
    #         # self.model = TransfoXLLMHeadModel(base_config)
    #         # base_config = GPTJConfig()
    #         # self.model = GPTJForCausalLM(base_config)
    #         # base_config = GPTNeoConfig()
    #         # self.model = GPTNeoForCausalLM(base_config)
    #         base_config = GPT2Config()
    #         self.model = GPT2LMHeadModel(base_config)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=NUM_DATA_WORKERS,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=NUM_DATA_WORKERS,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=NUM_DATA_WORKERS,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def forward(self, x, **kwargs):
        return self.model(x, labels=x, **kwargs)

    def training_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        loss = outputs.loss
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss)
        # wandb.log({"train_loss": loss, 'random_value': 1})
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        loss = outputs.loss
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        loss = outputs.loss
        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=5e-5)

    def validation_epoch_end(self, val_step_outputs):
        """NOTE: BLAST must be installed locally in order for this to work properly."""
        if not self.cfg.enable_blast:
            return
        # don't do anything to the validation step outputs, we're using this space to generate sequences and run blast
        # in order to monitor the similarity to training sequences
        generated = generate_dna_to_stop(
            self.model,
            self.fast_tokenizer,
            num_seqs=self.cfg.num_blast_seqs_per_gpu,
            biopy_seq=False,
        )
        blast_scores = []
        temp_fasta_dir = Path(
            str(self.cfg.checkpoint_dir)
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
                self.cfg.blast_validation_file,
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
        if self.cfg.generate_upon_completion:
            generated = generate_dna_to_stop(
                self.model,
                self.fast_tokenizer,
                num_seqs=self.cfg.num_blast_seqs_per_gpu,
                biopy_seq=True,
            )
            self.final_sequences.extend(generated)
            # save_path = Path(self.cfg.checkpoint_dir) / Path("final_generated_sequences.fasta")
            # seqs_to_fasta(generated, save_path)
            # print("Saved final generated sequences to ", save_path)


def load_from_deepspeed(
    cfg: ModelSettings,
    checkpoint_dir: Path,
    checkpoint: Path = "last.ckpt",
    model_weights: Path = "last.pt",
):
    """Utility function for deepspeed conversion"""
    # first convert the weights
    save_path = checkpoint_dir / checkpoint
    output_path = checkpoint_dir / model_weights
    # perform the conversion
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    # load model
    model = DNATransform.load_from_checkpoint(output_path, strict=False, cfg=cfg)
    # return the model
    return model


def train(cfg: ModelSettings, args):

    # check if loading from checkpoint - this assumes that you're loading from a sharded DeepSpeed checkpoint!!!
    if cfg.load_from_checkpoint_dir is not None:
        try:
            model = load_from_deepspeed(
                cfg=cfg, checkpoint_dir=cfg.load_from_checkpoint_dir
            )
            print(
                f"NOTE: loaded from existing model at checkpoint {cfg.load_from_checkpoint_dir}...."
            )
        except:
            print(
                f"WARNING: unable to load from checkpoint {cfg.load_from_checkpoint_dir}... training from scratch"
            )
            model = DNATransform(cfg)
    else:
        model = DNATransform(cfg)
    if cfg.wandb_active:
        print("Using Weights and Biases for logging...")
        wandb_logger = WandbLogger(project=cfg.wandb_project_name)
    else:
        wandb_logger = None
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        every_n_train_steps=cfg.val_check_interval,
        save_last=True,
        # monitor="val/loss",
        # mode="min",
        # filename="codon-transformer-{step:02d}-{val/loss:.2f}",
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=cfg.checkpoint_dir,
        # strategy="deepspeed_stage_3",#"ddp_sharded",#"ddp_spawn",
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
            logging_batch_size_per_gpu=cfg.batch_size,
        ),
        callbacks=[checkpoint_callback],
        # max_steps=cfg.training_steps,
        logger=wandb_logger,
        # profiler="simple",
        val_check_interval=cfg.val_check_interval,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        num_sanity_val_steps=2,
        precision=16,
        max_epochs=cfg.epochs,
        num_nodes=cfg.num_nodes,
    )
    trainer.fit(model)
    trainer.test(model)
    print("Completed training.")
    if cfg.generate_upon_completion:
        save_path = Path(cfg.checkpoint_dir) / Path("final_generated_sequences.fasta")
        seqs = model.final_sequences
        print("Length of final sequence list: ", len(seqs))
        seqs_to_fasta(seqs, save_path)
        print("Saved final generated sequences to ", save_path)


def inference(cfg: ModelSettings, cfg_file_name: str, dataset: str):

    model = load_from_deepspeed(cfg=cfg, checkpoint_dir=cfg.load_from_checkpoint_dir)
    model.cuda()

    if dataset == "train":
        loader = model.train_dataloader()
    elif dataset == "val":
        loader = model.val_dataloader()
    elif dataset == "test":
        loader = model.test_dataloader()

    print(f"Running inference with dataset length {len(loader)}")

    embeddings = []
    for batch in loader:
        batch = batch.cuda()
        outputs = model(batch, output_hidden_states=True)
        # outputs.hidden_states: (batch_size, sequence_length, hidden_size)
        embeddings.append(outputs.hidden_states.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    np.save("inference-train-embeddings.npy", embeddings)

    return embeddings


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--inference_dataset", default="train")
    args = parser.parse_args()
    config = ModelSettings.from_yaml(args.config)

    # Setup torch environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_num_threads(NUM_DATA_WORKERS)
    pl.seed_everything(0)

    # TODO: Should not pass args to these functions
    if args.mode == "train":
        train(config, args)
    if args.mode == "inference":
        inference(config, args, args.inference_dataset)
