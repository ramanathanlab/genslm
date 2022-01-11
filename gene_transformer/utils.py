from Bio.Seq import Seq

# global variables
stop_codons = ["TAA", "TAG", "TGA"]


def generate_dna_to_stop(model, fast_tokenizer, max_length=1024, top_k=50, top_p=0.95, num_seqs=5):
    output = model.generate(fast_tokenizer.encode("ATG", return_tensors="pt").cuda(), max_length=max_length,
                            do_sample=True, top_k=top_k, top_p=top_p, num_return_sequences=num_seqs, biopy_seq=False)
    seqs = []
    for i in output:
        seqs.append(fast_tokenizer.decode(i))
    seq_strings = []
    for s in seqs:
        dna = s.split(" ")
        for n, i in enumerate(dna):
            if i in stop_codons:
                to_stop = dna[:n + 1]
                break
        seq_strings.append("".join(to_stop))
    if biopy_seq:
        seq_strings = [Seq(s) for s in seq_strings]
    return seq_strings
