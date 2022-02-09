from Bio import SeqIO  # type: ignore[import]
from Bio.Seq import Seq  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]
from config import ModelSettings
from model import DNATransform
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pathlib import Path

# global variables
stop_codons = ["TAA", "TAG", "TGA"]


def generate_dna_to_stop(
    model,
    fast_tokenizer,
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
    seqs = []
    for i in output:
        seqs.append(fast_tokenizer.decode(i))
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


def seqs_to_fasta(seqs, file_name):
    records = []
    for n, i in enumerate(seqs):
        record = SeqRecord(
            i,
            id="MDH_SyntheticSeq_{}".format(n),
            name="MDH_sequence",
            description="synthetic malate dehydrogenase",
        )
        records.append(record)

    with open(file_name, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")


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

def load_from_deepspeed(checkpoint_dir: Path, config_file_name: Path, checkpoint: Path="last.ckpt",
                        model_weights: Path="last.pt"):
    # first convert the weights
    save_path = checkpoint_dir / checkpoint
    output_path = checkpoint_dir / model_weights
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    config = ModelSettings.from_yaml(config_file_name)
    model = DNATransform.load_from_checkpoint(output_path, strict=False, config=config)
    return model 

