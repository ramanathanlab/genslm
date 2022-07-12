from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from Bio import SeqIO  # type: ignore[import]
from Bio.Seq import Seq  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]
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


def generate_dna_to_stop(
    model: torch.nn.Module,  # type: ignore[name-defined]
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
    top_k: int = 50,
    top_p: float = 0.95,
    num_seqs: int = 5,
) -> torch.Tensor:
    # List of generated tokenized sequences.
    # stopping_criteria = StoppingCriteriaList([FoundStopCodonCriteria(tokenizer)])
    return model.generate(  # type: ignore[no-any-return]
        tokenizer.encode("ATG", return_tensors="pt").cuda(),
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_seqs,
        #        stopping_criteria=stopping_criteria,
    )


def tokens_to_sequences(
    tokens: torch.Tensor, tokenizer: PreTrainedTokenizerFast
) -> List[str]:
    # Decode tokens to codon strings
    seqs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    # Convert from tokens to string
    seq_strings = []
    for s in seqs:
        # Break into codons
        dna = s.split()
        # Iterate through until you reach a stop codon
        for i, codon in enumerate(dna):
            if codon in STOP_CODONS:
                break
        # Get the open reading frame
        to_stop = dna[: i + 1]
        # Create the string and append to list
        seq_strings.append("".join(to_stop))
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

    # begin generation loop
    while len(unique_seqs) < num_seqs:
        tokens = generate_dna_to_stop(
            model,
            tokenizer,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            num_seqs=1,
        )
        seq = tokens_to_sequences(tokens, tokenizer=tokenizer)[0]
        if seq not in known_sequences:
            all_generated_seqs.append(seq)
            unique_seqs.add(seq)

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
