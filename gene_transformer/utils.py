import time
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Type

import numpy as np
import pytorch_lightning as pl
import torch
from Bio import SeqIO  # type: ignore[import]
from Bio.Seq import Seq  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast  # , StoppingCriteriaList
from transformers import StoppingCriteria

STOP_CODONS = {"TAA", "TAG", "TGA"}


class FoundStopCodonCriteria(StoppingCriteria):  # type: ignore[misc]
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.tokenizer = tokenizer
        self.stop_set: Set[int] = set()

        # TODO: If we can get this class working correctly,
        #       we could store the indicies of the first stop
        #       codon in each batch. That way we can avoid a loop
        #       of post processing.

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> bool:
        codons = self.tokenizer.batch_decode(input_ids[:, -1], skip_special_tokens=True)

        batch_size = input_ids.shape[0]
        still_generating = set(range(batch_size)) - self.stop_set

        for i in still_generating:
            if codons[i] in STOP_CODONS:
                self.stop_set.add(i)

        # If each sequence in the batch has seen a stop codon
        return len(self.stop_set) == batch_size


def generate_dna(
    model: torch.nn.Module,  # type: ignore[name-defined]
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
    top_k: int = 50,
    top_p: float = 0.95,
    num_seqs: int = 5,
    remove_invalid_values: bool = True,
) -> torch.Tensor:
    # remove_invalid_values slows down the calculation
    # but is more robust for the reformer model.
    # List of generated tokenized sequences.
    # stopping_criteria = StoppingCriteriaList([FoundStopCodonCriteria(tokenizer)])

    return model.generate(  # type: ignore[no-any-return]
        tokenizer.encode("ATG", return_tensors="pt").cuda(),
        max_length=max_length,
        min_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_seqs,
        remove_invalid_values=remove_invalid_values,
        use_cache=True,
        pad_token_id=tokenizer.encode("[PAD]")[0],
        #        stopping_criteria=stopping_criteria,
    )


def find_stop_codon(codons: List[str]) -> int:
    # Iterate through until you reach a stop codon
    # and return the index
    for i, codon in enumerate(codons):
        if codon in STOP_CODONS:
            return i
    return len(codons) - 1


def tokens_to_sequences(
    tokens: torch.Tensor, tokenizer: PreTrainedTokenizerFast, to_stop_codon: bool = True
) -> List[str]:
    # Decode tokens to codon strings
    seqs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    # Convert from tokens to string
    seq_strings = []
    for s in seqs:
        # Break into codons
        codons = s.split()
        if to_stop_codon:
            # Get the open reading frame
            ind = find_stop_codon(codons)
            codons = codons[: ind + 1]
        # Create the DNA string and append to list
        seq_strings.append("".join(codons))
    return seq_strings


def seqs_to_fasta(
    seqs: List[str],
    file_name: Path,
    translate_to_protein: bool = False,
    custom_seq_name: str = "SyntheticSeq",
) -> None:
    sequences = [Seq(seq) for seq in seqs]

    if translate_to_protein:
        sequences = [s.translate() for s in sequences]

    records = [
        SeqRecord(
            seq,
            id=f"{custom_seq_name}_{i}",
            name=custom_seq_name,
            description=custom_seq_name,
        )
        for i, seq in enumerate(sequences)
    ]

    SeqIO.write(records, file_name, "fasta")


def non_redundant_generation(
    model: torch.nn.Module,  # type: ignore[name-defined]
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
    top_k: int = 50,
    top_p: float = 0.95,
    num_seqs: int = 5,
    known_sequence_files: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Utility which will generate unique sequences which are not duplicates of each other nor found within the
    training dataset (optional). Returns a dictionary of unique sequences, all generated sequences, and time required.
    """
    # initialization of variables
    known_sequences: Set[str] = set()
    all_generated_seqs: List[str] = list()
    unique_seqs: Set[str] = set()

    if known_sequence_files is not None:
        known_sequences = set(map(str, get_known_sequences(known_sequence_files)))

    if len(known_sequences) > 1:
        lengths = [len(s) for s in known_sequences]
        length_cutoff = min(lengths)
    else:
        length_cutoff = 0

    print(f"Using length cutoff of {length_cutoff} - {length_cutoff // 3} tokens.")

    # begin generation loop
    while len(unique_seqs) < num_seqs:
        print(
            f"Current number of unique sequences meeting criteria: {len(unique_seqs)}"
        )
        print(f"Current number of sequences generated: {len(all_generated_seqs)}")
        tokens = generate_dna(
            model,
            tokenizer,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            num_seqs=1,
        )
        seq = tokens_to_sequences(tokens, tokenizer=tokenizer)[0]
        all_generated_seqs.append(seq)
        if seq not in known_sequences and len(seq) > length_cutoff:
            unique_seqs.add(seq)
            print("Unique Sequence Length: {}".format(len(unique_seqs)))

    # create dictionary of results
    results = {
        "unique_seqs": list(unique_seqs),
        "all_generated_seqs": all_generated_seqs,
    }
    return results


def get_known_sequences(files: List[str]) -> List[Seq]:
    """Return list of Seq objects from given list of files"""
    known_sequences = []
    for f in files:
        records = list(SeqIO.parse(f, "fasta"))
        seqs = [s.seq for s in records]
        known_sequences.extend(seqs)
    return known_sequences


def redundancy_check(
    generated: str, known_sequences: List[Seq], verbose: bool = False
) -> bool:
    """Check if a sequence appears in a list of known sequence"""
    for gen_seq in tqdm(generated, disable=verbose):
        if gen_seq in known_sequences:
            return False
    # no redundancies found
    return True


class ModelLoadStrategy(ABC):
    @abstractmethod
    def get_model(self, pl_module: "Type[pl.LightningModule]") -> "pl.LightningModule":
        """Load and return a module object."""


class LoadDeepSpeedStrategy(ModelLoadStrategy):
    def __init__(self, weight_path: Path, **kwargs: Any) -> None:
        """Load DeepSpeed checkpoint path.

        Parameters
        ----------
        weight_path : Path
            DeepSpeed checkpoint directory.
        """
        self.weight_path = weight_path
        self.kwargs = kwargs

    def get_model(self, pl_module: "Type[pl.LightningModule]") -> "pl.LightningModule":
        """Utility function for deepspeed conversion"""
        pt_file = str(self.weight_path.with_suffix(".pt"))
        # perform the conversion from deepspeed to pt weights
        convert_zero_checkpoint_to_fp32_state_dict(str(self.weight_path), pt_file)
        # load model
        model = pl_module.load_from_checkpoint(pt_file, strict=False, **self.kwargs)
        return model


class LoadPTCheckpointStrategy(ModelLoadStrategy):
    def __init__(self, weight_path: Path, **kwargs: Any) -> None:
        """Load a PyTorch model weight file.

        Parameters
        ----------
        weight_path : Path
            PyTorch model weight file.

        Raises
        ------
        ValueError
            If the `weight_path` does not have the `.pt` extension.
        """
        if weight_path.suffix != ".pt":
            raise ValueError("weight_path must be a .pt file")
        self.weight_path = weight_path
        self.kwargs = kwargs

    def get_model(self, pl_module: "Type[pl.LightningModule]") -> "pl.LightningModule":
        model = pl_module.load_from_checkpoint(
            str(self.weight_path), strict=False, **self.kwargs
        )
        return model


class ThroughputMonitor(Callback):
    """Custom callback in order to monitor the throughput and log to weights and biases."""

    def __init__(
        self, batch_size: int, num_nodes: int = 1, wandb_active: bool = False
    ) -> None:
        """Logs throughput statistics starting at the 2nd epoch."""
        super().__init__()
        self.wandb_active = wandb_active
        self.start_time = 0.0
        self.average_throughput = 0.0
        self.average_sample_time = 0.0
        self.batch_times: List[float] = []
        self.epoch_throughputs: List[float] = []
        self.epoch_sample_times: List[float] = []
        self.num_ranks = num_nodes * torch.cuda.device_count()
        self.macro_batch_size = batch_size * self.num_ranks

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if pl_module.current_epoch > 0:
            self.start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if pl_module.current_epoch > 0:
            batch_time = time.time() - self.start_time
            self.batch_times.append(batch_time)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.current_epoch > 0:
            # compute average epoch throughput
            avg_batch_time = mean(self.batch_times)
            avg_epoch_throughput = self.macro_batch_size / avg_batch_time
            avg_secs_per_sample = avg_batch_time / self.macro_batch_size
            pl_module.log("stats/average_epoch_throughput", avg_epoch_throughput)
            pl_module.log("stats/average_secs_per_sample", avg_secs_per_sample)
            self.epoch_throughputs.append(avg_epoch_throughput)
            self.epoch_sample_times.append(avg_secs_per_sample)
            self.batch_times = []  # Reset for next epoch

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.average_throughput = mean(self.epoch_throughputs)
        self.average_sample_time = mean(self.epoch_sample_times)

        # Collect metrics on each rank and compute overall statistics on rank 0
        metrics = self.average_throughput, self.average_sample_time
        trainer._accelerator_connector.strategy.barrier()
        metrics = pl_module.all_gather(metrics)
        throughputs, sample_times = metrics[0], metrics[1]
        if trainer.is_global_zero:
            thru_avg, thru_stdev = throughputs.mean().item(), throughputs.std().item()
            print(
                f"\nAVERAGE THROUGHPUT: {thru_avg} +- {thru_stdev} "
                f" samples/second over {self.num_ranks} ranks"
            )

            sample_time_avg = sample_times.mean().item()
            sample_time_stdev = sample_times.std().item()

            print(
                f"AVERAGE SECONDS PER SAMPLE: {sample_time_avg} +- {sample_time_stdev} "
                f"seconds/sample over {self.num_ranks} ranks"
            )

            if self.wandb_active:
                pl_module.logger.log_text(
                    key="stats/performance",
                    columns=[
                        "throughput_avg",
                        "throughput_stdev",
                        "sample_time_avg",
                        "sample_time_stdev",
                        "macro_batch_size",
                        "ranks",
                    ],
                    data=[
                        [
                            thru_avg,
                            thru_stdev,
                            sample_time_avg,
                            sample_time_stdev,
                            self.macro_batch_size,
                            self.num_ranks,
                        ]
                    ],
                )


class SequenceGenerationCallback(Callback):
    """Custom callback to generate sequences at the end of epoch."""

    def __init__(
        self,
        block_size: int,
        num_test_seqs_per_gpu: int,
        output_dir: Path,
        custom_seq_name: str = "SyntheticSeq",
        known_sequence_files: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.block_size = block_size
        self.num_test_seqs_per_gpu = num_test_seqs_per_gpu
        self.output_dir = output_dir
        self.custom_seq_name = custom_seq_name
        self.known_sequence_files = known_sequence_files

        # Collect generated sequences at each epoch end
        self.final_sequences: Dict[str, List[str]] = {}

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        # Generate sequences using the model
        results = non_redundant_generation(
            pl_module.model,
            pl_module.tokenizer,
            num_seqs=self.num_test_seqs_per_gpu,
            max_length=self.block_size,
            known_sequence_files=self.known_sequence_files,
        )
        unique_seqs, all_seqs = results["unique_seqs"], results["all_generated_seqs"]
        print(f"Proportion of unique seqs: {len(unique_seqs) / len(all_seqs)}")

        # Wait until all ranks meet up here
        trainer._accelerator_connector.strategy.barrier()
        unique_seqs = pl_module.all_gather(unique_seqs)

        if trainer.is_global_zero:  # type: ignore[attr-defined]
            print(f"sequences {len(unique_seqs)}")
            self.final_sequences[f"globalstep-{pl_module.global_step}"] = unique_seqs

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.is_global_zero:
            self.output_dir.mkdir(exist_ok=True, parents=True)
            for name, seqs in self.final_sequences.items():
                seqs_to_fasta(
                    seqs,
                    self.output_dir / f"{name}.fasta",
                    custom_seq_name=self.custom_seq_name,
                )
            print(f"Saved final generated sequences to {self.output_dir}")


class PerplexityCallback(Callback):
    """Model perplexity calculation"""

    # TODO: Perplexity in training
    def __init__(
        self,
        log_steps: int = 0,
        train_name: str = "train/ppl",
        val_name: str = "val/ppl",
    ) -> None:
        super().__init__()
        self.log_steps = log_steps
        self.train_name = train_name
        self.val_name = val_name
        self.train_perplexities: List[float] = []
        self.val_perplexities: List[float] = []

    def _get_perplexities(self, train: bool) -> List[float]:
        return self.train_perplexities if train else self.val_perplexities

    def _log_perplexity(
        self, trainer: "pl.Trainer", log_name: str, train: bool
    ) -> None:
        print(trainer.__dict__)
        perplexities = self._get_perplexities(train)
        mean_ppl = np.mean(perplexities)
        perplexities = []
        trainer.log(log_name, mean_ppl)

    def _on_batch_end(
        self,
        trainer: "pl.Trainer",
        loss: torch.Tensor,
        batch_idx: int,
        log_name: str,
        train: bool,
    ) -> None:
        self._get_perplexities(train).append(torch.exp(loss.cpu().long()).item())
        if self.log_steps and self.log_steps % batch_idx == 0:
            self._log_perplexity(trainer, log_name, train)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx,
    ) -> None:
        self._on_batch_end(
            trainer, outputs["loss"], batch_idx, self.train_name, train=True
        )

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._on_batch_end(trainer, outputs, batch_idx, self.val_name, train=False)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._log_perplexity(trainer, self.val_name, train=False)
