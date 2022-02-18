from Bio import SeqIO  # type: ignore[import]
from Bio.Seq import Seq  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]
import torch
from transformers import PreTrainedTokenizerFast
from typing import List

# from config import ModelSettings
# from model import DNATransform


# global variables
stop_codons = ["TAA", "TAG", "TGA"]


def generate_dna_to_stop(
    model: torch.nn.Module,
    fast_tokenizer: PreTrainedTokenizerFast,
    max_length: int = 1024,
    top_k: int = 50,
    top_p: float = 0.95,
    num_seqs: int = 5,
    biopy_seq: bool = False,
):
    output = model.generate(
        fast_tokenizer.encode("ATG", return_tensors="pt").cuda(),
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_seqs,
    )
    seqs = [fast_tokenizer.decode(i, skip_special_tokens=True) for i in output]
    seq_strings = []
    for s in seqs:
        dna = s.split(" ")
        for n, i in enumerate(dna):
            if i in stop_codons:
                to_stop = dna[: n + 1]
                break
        try:
            seq_strings.append("".join(to_stop))
        except NameError:
            seq_strings.append("".join(dna))
    if biopy_seq:
        seq_strings = [Seq(s) for s in seq_strings]
    return seq_strings


def seqs_to_fasta(seqs: List[Seq], file_name: str):
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
    file_name,
    model,
    fast_tokenizer,
    max_length: int = 1024,
    top_k: int = 50,
    top_p: float = 0.95,
    num_seqs: int = 5,
    translate_to_protein: bool = False,
):
    # generate seq objects
    generated = generate_dna_to_stop(
        model,
        fast_tokenizer,
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
