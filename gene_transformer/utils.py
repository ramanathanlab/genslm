from Bio import SeqIO  # type: ignore[import]
from Bio.Seq import Seq  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]
import torch
from transformers import (
    PreTrainedTokenizerFast,
    StoppingCriteria,
)  # , StoppingCriteriaList
from typing import List


STOP_CODONS = {"TAA", "TAG", "TGA"}


class FoundStopCodonCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.tokenizer = tokenizer
        self.stop_set = set()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
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
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
    top_k: int = 50,
    top_p: float = 0.95,
    num_seqs: int = 5,
    biopy_seq: bool = False,
):
    # List of generated tokenized sequences.
    # stopping_criteria = StoppingCriteriaList([FoundStopCodonCriteria(tokenizer)])
    output = model.generate(
        tokenizer.encode("ATG", return_tensors="pt").cuda(),
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_seqs,
        #        stopping_criteria=stopping_criteria,
    )

    # Decode tokens to codon strings
    seqs = tokenizer.batch_decode(output, skip_special_tokens=True)
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
    # Convert to biopython objects if requested
    if biopy_seq:
        seq_strings = [Seq(s) for s in seq_strings]
    return seq_strings


def seqs_to_fasta(seqs: List[Seq], file_name: str) -> None:
    records = [
        SeqRecord(
            seq,
            id="MDH_SyntheticSeq_{}".format(i),
            name="MDH_sequence",
            description="synthetic malate dehydrogenase",
        )
        for i, seq in enumerate(seqs)
    ]

    SeqIO.write(records, file_name, "fasta")


def generate_fasta_file(
    file_name: str,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
    top_k: int = 50,
    top_p: float = 0.95,
    num_seqs: int = 5,
    translate_to_protein: bool = False,
) -> None:
    # generate seq objects
    generated = generate_dna_to_stop(
        model,
        tokenizer,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        num_seqs=num_seqs,
        biopy_seq=True,
    )
    if translate_to_protein:
        generated = [s.translate() for s in generated]
    # generate seq records
    seqs_to_fasta(generated, file_name)
