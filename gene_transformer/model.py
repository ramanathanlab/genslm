import os
import numpy as np
from tqdm import tqdm  # type: ignore[import]
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, List

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer  # type: ignore[import]

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from deepspeed.ops.adam import DeepSpeedCPUAdam  # type: ignore[import]

from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

from gene_transformer.config import ModelSettings
from gene_transformer.dataset import FASTADataset
from gene_transformer.blast import ParallelBLAST
from gene_transformer.utils import generate_dna_to_stop, seqs_to_fasta


class DNATransformer(pl.LightningModule):
    def __init__(self, cfg: ModelSettings) -> None:
        super().__init__()
        self.save_hyperparameters(cfg.dict())
        self.cfg = cfg
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(self.cfg.tokenizer_file)
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.final_sequences = []

        self.train_dataset = self._get_dataset(self.cfg.train_file)
        self.val_dataset = self._get_dataset(self.cfg.val_file)
        self.test_dataset = self._get_dataset(self.cfg.test_file)

        # pdb.set_trace()
        if self.cfg.use_pretrained:
            self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        else:
            # base_config = GPTNeoConfig()
            # self.model = GPTNeoForCausalLM(base_config)
            base_config = GPT2Config(vocab_size=self.tokenizer.vocab_size)
            self.model = GPT2LMHeadModel(base_config)

        # To validate generated sequences
        # TODO: make sure temp files are outputting to node local
        self.blast = ParallelBLAST(
            database_file=self.cfg.blast_validation_file,
            blast_dir=self.cfg.checkpoint_dir / "blast",
            blast_exe_path=self.cfg.blast_exe_path,
            num_workers=min(10, self.cfg.num_blast_seqs_per_gpu),
            node_local_path=self.cfg.node_local_path,
        )

    def _get_dataset(self, file: str) -> FASTADataset:
        """Helper function to generate dataset."""
        return FASTADataset(
            file, tokenizer=self.tokenizer, block_size=self.cfg.block_size
        )

    def _get_dataloader(self, dataset: FASTADataset, shuffle: bool) -> DataLoader:
        """Helper function to generate dataloader."""
        return DataLoader(
            dataset,
            shuffle=shuffle,
            drop_last=True,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_data_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset, shuffle=False)

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

    def forward(self, x: torch.Tensor, **kwargs: Any) -> GPT2DoubleHeadsModelOutput:
        return self.model(x, labels=x, **kwargs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        x = batch
        outputs = self(x)
        loss = outputs.loss
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss)
        # wandb.log({"train_loss": loss, 'random_value': 1})
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        x = batch
        outputs = self(x)
        loss = outputs.loss
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        x = batch
        outputs = self(x)
        loss = outputs.loss
        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return DeepSpeedCPUAdam(self.parameters(), lr=5e-5)

    def validation_epoch_end(self, val_step_outputs: List[torch.FloatTensor]) -> None:
        # NOTE: BLAST must be installed locally in order for this to work properly.
        if not self.cfg.enable_blast:
            return

        # Generate sequences and run blast across all ranks,
        # then gather mean, max for logging on rank 0.

        # Don't do anything to the validation step outputs, we're using this
        # space to generate sequences and run blast in order to monitor the
        # similarity to training sequences
        generated = generate_dna_to_stop(
            self.model,
            self.tokenizer,
            num_seqs=self.cfg.num_blast_seqs_per_gpu,
            max_length=self.cfg.block_size,
        )

        prefix = f"globalstep{self.global_step}"
        max_scores, mean_scores = self.blast.run(generated, prefix)
        metrics = np.mean(mean_scores), np.max(max_scores)
        # Wait until all ranks meet up here
        self.trainer._accelerator_connector.strategy.barrier()
        metrics = self.all_gather(metrics)
        max_score, mean_score = metrics[1].max().cpu(), metrics[0].mean().cpu()
        self.log("val/max_blast_score", max_score, logger=True, prog_bar=True)
        self.log("val/mean_blast_score", mean_score, logger=True, prog_bar=True)
        if self.trainer.is_global_zero:
            self.blast.backup_results()

    def test_epoch_end(self, outputs: List[torch.FloatTensor]) -> None:
        if self.trainer.is_global_zero and self.cfg.generate_upon_completion:
            generated = generate_dna_to_stop(
                self.model,
                self.tokenizer,
                num_seqs=self.cfg.num_seqs_test,
                max_length=self.cfg.block_size,
                biopy_seq=True,
            )
            self.final_sequences.extend(generated)


def load_from_deepspeed(
    cfg: ModelSettings,
    checkpoint_dir: Path,
    checkpoint: str = "last.ckpt",
    model_weights: str = "last.pt",
) -> DNATransformer:
    """Utility function for deepspeed conversion"""
    # first convert the weights
    save_path = str(checkpoint_dir / checkpoint)
    output_path = str(checkpoint_dir / model_weights)
    # perform the conversion
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    # load model
    model = DNATransformer.load_from_checkpoint(output_path, strict=False, cfg=cfg)
    return model


def train(cfg: ModelSettings) -> None:

    # Check if loading from checkpoint - this assumes that you're
    # loading from a sharded DeepSpeed checkpoint!!!
    if cfg.load_from_checkpoint_dir is not None:
        model = load_from_deepspeed(
            cfg=cfg, checkpoint_dir=cfg.load_from_checkpoint_dir
        )
        print(f"Loaded existing model at checkpoint {cfg.load_from_checkpoint_dir}....")
    else:
        model = DNATransformer(cfg)

    # Setup wandb
    if cfg.wandb_active:
        print("Using Weights and Biases for logging...")
        wandb_logger = WandbLogger(project=cfg.wandb_project_name)
    else:
        wandb_logger = None

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        save_last=True,
        # monitor="val/loss",
        # mode="min",
        # filename="codon-transformer-{step:02d}-{val/loss:.2f}",
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=str(cfg.checkpoint_dir),
        # Use NVMe offloading on other clusters see more here:
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed-infinity-nvme-offloading
        strategy=DeepSpeedPlugin(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            # remote_device="nvme",
            # # offload_params_device="nvme",
            # offload_optimizer_device="nvme",
            # nvme_path="/tmp",
            logging_batch_size_per_gpu=cfg.batch_size,
        ),
        callbacks=[checkpoint_callback],
        # max_steps=cfg.training_steps,
        logger=wandb_logger,
        # profiler="simple",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        num_sanity_val_steps=2,
        precision=16,
        max_epochs=cfg.epochs,
        num_nodes=cfg.num_nodes,
    )
    trainer.fit(model)
    trainer.test(model)
    print("Completed training.")
    if trainer.is_global_zero and cfg.generate_upon_completion:
        save_path = cfg.checkpoint_dir / "final_generated_sequences.fasta"
        seqs = model.final_sequences
        print("Length of final sequence list: ", len(seqs))
        seqs_to_fasta(seqs, save_path)
        print("Saved final generated sequences to ", save_path)


def inference(cfg: ModelSettings, dataset: str) -> None:

    if cfg.load_from_checkpoint_dir is None:
        raise ValueError("load_from_checkpoint_dir must be set in the config file")

    model = load_from_deepspeed(cfg=cfg, checkpoint_dir=cfg.load_from_checkpoint_dir)
    model.cuda()

    if dataset == "train":
        loader = model.train_dataloader()
    elif dataset == "val":
        loader = model.val_dataloader()
    elif dataset == "test":
        loader = model.test_dataloader()

    print(f"Running inference with dataset length {len(loader)}")

    # TODO: Instead could optionally return the hidden_states in a dictionary
    # in the validation_step function.
    embeddings = []
    for batch in tqdm(loader):
        batch = batch.cuda()
        outputs = model(batch, output_hidden_states=True)
        # outputs.hidden_states: (batch_size, sequence_length, hidden_size)
        embeddings.append(outputs.hidden_states[0].detach().cpu().numpy())

    embeddings = np.concatenate(embeddings)  # type: ignore

    print(f"Embeddings shape: {embeddings.shape}")  # type: ignore
    np.save("inference-train-embeddings.npy", embeddings)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--inference_dataset", default="train")
    args = parser.parse_args()
    config = ModelSettings.from_yaml(args.config)

    # Setup torch environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_num_threads(config.num_data_workers)
    pl.seed_everything(0)

    if args.mode == "train":
        train(config)
    if args.mode == "inference":
        inference(config, args.inference_dataset)
