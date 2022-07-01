import os
import numpy as np
from tqdm import tqdm  # type: ignore[import]
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, List, Dict

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer  # type: ignore[import]

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from deepspeed.ops.adam import DeepSpeedCPUAdam  # type: ignore[import]

# warm up scheduler
from deepspeed.runtime.lr_schedules import WarmupLR
from pytorch_lightning.plugins.environments.slurm_environment import SLURMEnvironment

from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    AutoModelForCausalLM,
    AutoConfig,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

from gene_transformer.config import ModelSettings
from gene_transformer.dataset import (
    FASTADataset,
    GenomeDataset,
    BPEGenomeDataset,
    H5Dataset,
    IndividualFastaDataset
)
from gene_transformer.blast import ParallelBLAST
from gene_transformer.utils import (
    generate_dna_to_stop,
    tokens_to_sequences,
    seqs_to_fasta,
)

import pdb
import socket
import os
import subprocess


class DNATransformer(pl.LightningModule):
    def __init__(self, cfg: ModelSettings) -> None:
        super().__init__()
        self.save_hyperparameters(cfg.dict())
        self.cfg = cfg
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(self.cfg.tokenizer_file)
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # self.current_learning_rate = None

        # these are defined in get_dataloader functions
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # if not self.cfg.genome_level:
        #
        #     self.train_dataset = self._get_dataset(self.cfg.train_file)
        #     self.val_dataset = self._get_dataset(self.cfg.val_file)
        #     self.test_dataset = self._get_dataset(self.cfg.test_file)
        #
        # else:
        #     self.train_dataset = self._get_genome_dataset(self.cfg.train_file)
        #     self.val_dataset = self._get_genome_dataset(self.cfg.val_file)
        #     self.test_dataset = self._get_genome_dataset(self.cfg.test_file)

        base_config = AutoConfig.from_pretrained(
            self.cfg.model_name,
            vocab_size=self.tokenizer.vocab_size,
            feed_forward_size=self.cfg.block_size,
            axial_pos_embds=False,
            # local_chunk_length=100,
            # lsh_attn_chunk_length=100,
            axial_pos_shape=(128, 94),
            # max_position_embeddings=cfg.block_size,
            # max_position_embeddings=self.cfg.block_size,
        )
        self.model = AutoModelForCausalLM.from_config(base_config)

        # To validate generated sequences
        # TODO: make sure temp files are outputting to node local
        self.blast = ParallelBLAST(
            database_file=self.cfg.blast_validation_file,
            blast_dir=self.cfg.checkpoint_dir / "blast",
            blast_exe_path=self.cfg.blast_exe_path,
            num_workers=min(10, self.cfg.num_blast_seqs_per_gpu),
            node_local_path=self.cfg.node_local_path,
        )

        # Collect generated sequences at each epoch end
        self.final_sequences: Dict[str, List[str]] = {}

        print("Hostname: {}".format(socket.gethostname()))

    def _get_dataset(self, file: str) -> FASTADataset:
        """Helper function to generate dataset."""
        return FASTADataset(
            file,
            tokenizer=self.tokenizer,
            block_size=self.cfg.block_size,
            alphabet=self.cfg.alphabet_type,
        )

    def _get_genome_dataset(self, file: str, dset_name: str) -> H5Dataset:
        """Helper function to generate genome dataset"""
        return IndividualFastaDataset(
            file,
            # dset_name=dset_name,
            block_size=self.cfg.block_size,
            tokenizer=self.tokenizer,
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
        if not self.cfg.genome_level:
            self.train_dataset = self._get_dataset(self.cfg.train_file)
        else:
            self.train_dataset = self._get_genome_dataset(self.cfg.train_file, "train")
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if not self.cfg.genome_level:
            self.val_dataset = self._get_dataset(self.cfg.val_file)
        else:
            self.val_dataset = self._get_genome_dataset(self.cfg.val_file, "val")
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if not self.cfg.genome_level:
            self.test_dataset = self._get_dataset(self.cfg.test_file)
        else:
            self.test_dataset = self._get_genome_dataset(self.cfg.test_file, "test")
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> GPT2DoubleHeadsModelOutput:  # type: ignore[override]
        return self.model(x, labels=x, **kwargs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:  # type: ignore[override]
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self) -> DeepSpeedCPUAdam:
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=5e-5)
        scheduler = WarmupLR(
            optimizer, warmup_min_lr=5e-8, warmup_max_lr=5e-5, warmup_num_steps=50000
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()
        # self.current_learning_rate = scheduler.get_last_lr()

    def validation_epoch_end(self, val_step_outputs: List[torch.FloatTensor]) -> None:  # type: ignore[override]
        # NOTE: BLAST must be installed locally in order for this to work properly.
        if not self.cfg.enable_blast:
            return

        # Generate sequences and run blast across all ranks,
        # then gather mean, max for logging on rank 0.

        # Don't do anything to the validation step outputs, we're using this
        # space to generate sequences and run blast in order to monitor the
        # similarity to training sequences
        tokens = generate_dna_to_stop(
            self.model,
            self.tokenizer,
            num_seqs=self.cfg.num_blast_seqs_per_gpu,
            max_length=self.cfg.block_size,
        )
        sequences = tokens_to_sequences(tokens, self.tokenizer)

        prefix = f"globalstep{self.global_step}"
        max_scores, mean_scores = self.blast.run(sequences, prefix)
        metrics = np.max(max_scores), np.mean(mean_scores)
        # Wait until all ranks meet up here
        self.trainer._accelerator_connector.strategy.barrier()  # type: ignore[attr-defined]
        metrics = self.all_gather(metrics)
        try:
            max_score, mean_score = metrics[0].max().cpu(), metrics[1].mean().cpu()
        except AttributeError as e:
            # getting a weird numpy error when running validation on the protein sequences so catching here
            print("Attribute error when trying to move tensor to CPU... Error:", e)
            max_score, mean_score = metrics[0].max(), metrics[1].mean()
        self.log("val/max_blast_score", max_score, logger=True, prog_bar=True)
        self.log("val/mean_blast_score", mean_score, logger=True, prog_bar=True)
        if self.trainer.is_global_zero:  # type: ignore[attr-defined]
            # Will move blast results (fasta and csv file) from the node
            # where rank-0 runs to the file system (will also move files
            # written by other ranks on the node)
            self.blast.backup_results()

    def test_epoch_end(self, outputs: List[torch.FloatTensor]) -> None:  # type: ignore[override]
        if not self.cfg.num_test_seqs_per_gpu:
            return None

        tokens = generate_dna_to_stop(
            self.model,
            self.tokenizer,
            num_seqs=self.cfg.num_test_seqs_per_gpu,
            max_length=self.cfg.block_size,
        )

        # Wait until all ranks meet up here
        self.trainer._accelerator_connector.strategy.barrier()  # type: ignore[attr-defined]
        # sequences after all_gather is shape (world_size, num_test_seqs_per_gpu, block_size)
        tokens = self.all_gather(tokens)

        if self.trainer.is_global_zero:  # type: ignore[attr-defined]
            # Concatenate over world size
            tokens = tokens.view(-1, self.cfg.block_size)
            sequences = tokens_to_sequences(tokens, self.tokenizer)
            # sequences = np.concatenate(sequences.cpu().numpy())
            print(f"sequences {len(sequences)}")
            self.final_sequences[f"globalstep{self.global_step}"] = sequences


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
    return model  # type: ignore[no-any-return]


def train(cfg: ModelSettings) -> None:
    # Check if loading from checkpoint - this assumes that you're
    # loading from a sharded DeepSpeed checkpoint!!!
    if cfg.load_from_checkpoint_dir is not None:
        model = load_from_deepspeed(
            cfg=cfg, checkpoint_dir=cfg.load_from_checkpoint_dir
        )
        print(f"Loaded existing model at checkpoint {cfg.load_from_checkpoint_dir}....")
        try:
            model.model.lm_head.bias.data = torch.zeros_like(
                model.model.lm_head.bias.data
            )
        except Exception as e:
            print("Couldn't set bias equal to zeros.")
            pass
    else:
        model = DNATransformer(cfg)

    # Setup wandb
    if cfg.wandb_active:
        print("Using Weights and Biases for logging...")
        wandb_logger = WandbLogger(project=cfg.wandb_project_name)
        # log gradients and model topology
        # wandb_logger.watch(model)
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

    if cfg.wandb_active:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [checkpoint_callback, lr_monitor]
    else:
        callbacks = [checkpoint_callback]

    # os.environ["WORLD_SIZE"] = str(cfg.num_nodes * 4)
    # print("World size: {}".format(os.environ["WORLD_SIZE"]))
    # os.environ["MASTER_PORT"] = "1234"
    # # get master host name
    # cmd = ["scontrol", "show", "hostnames"]
    # x = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # out = x.communicate()[0]
    # hostnames = out.decode()
    # os.environ["MASTER_ADDR"] = hostnames.split("\n")[0]
    # print("Master address: {}".format(os.environ["MASTER_ADDR"]))

    trainer = pl.Trainer(
        # use all available gpus
        gpus=-1,
        default_root_dir=str(cfg.checkpoint_dir),
        # Use NVMe offloading on other clusters see more here:
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed-infinity-nvme-offloading
        strategy=DeepSpeedPlugin(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            remote_device="cpu",
            offload_params_device="cpu",
            # offload_optimizer_device="nvme",
            # nvme_path="/tmp",
            logging_batch_size_per_gpu=cfg.batch_size,
            # add the option to load a config from json file with more deepspeed options
            # note that if supplied all defaults are ignored - model settings defaults this arg to None
            # config=cfg.deepspeed_cfg_file
        ),
        callbacks=callbacks,
        # max_steps=cfg.training_steps,
        logger=wandb_logger,
        # profiler="simple",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        num_sanity_val_steps=0,
        precision=16,
        max_epochs=cfg.epochs,
        num_nodes=cfg.num_nodes,
        # plugins=[SLURMEnvironment(auto_requeue=False)]
    )
    trainer.fit(model)
    trainer.test(model)

    if trainer.is_global_zero:
        print("Completed training.")

    if trainer.is_global_zero and cfg.num_test_seqs_per_gpu:
        save_path = cfg.checkpoint_dir / "generated"
        save_path.mkdir(exist_ok=True)
        for name, seqs in model.final_sequences.items():
            seqs_to_fasta(seqs, save_path / f"{name}.fasta")
        print(f"Saved final generated sequences to {save_path}")


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
    np.save(f"inference-{dataset}-embeddings.npy", embeddings)


def test(cfg: ModelSettings) -> None:
    """Run test dataset after loading from checkpoint"""
    if cfg.load_from_checkpoint_dir is None:
        raise ValueError("load_from_checkpoint_dir must be set in the config file")

    model = load_from_deepspeed(cfg=cfg, checkpoint_dir=cfg.load_from_checkpoint_dir)
    model.cuda()

    # Setup wandb
    if cfg.wandb_active:
        print("Using Weights and Biases for logging...")
        wandb_logger = WandbLogger(project=cfg.wandb_project_name)
    else:
        wandb_logger = None

    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=str(cfg.checkpoint_dir),
        # Use NVMe offloading on other clusters see more here:
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed-infinity-nvme-offloading
        strategy=DeepSpeedPlugin(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            remote_device="cpu",
            offload_params_device="cpu",
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

    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--inference_dataset", default="train")
    args = parser.parse_args()
    config = ModelSettings.from_yaml(args.config)

    # Setup torch environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_num_threads(config.num_data_workers)  # type: ignore[attr-defined]
    pl.seed_everything(0)

    if args.mode == "train":
        train(config)
    if args.mode == "inference":
        inference(config, args.inference_dataset)
    if args.mode == "test":
        test(config)
