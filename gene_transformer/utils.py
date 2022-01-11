from Bio.Seq import Seq

# global variables
stop_codons = ["TAA", "TAG", "TGA"]


def generate_dna_to_stop(model, fast_tokenizer, max_length=1024, top_k=50, top_p=0.95, num_seqs=5, biopy_seq=False):
    output = model.generate(fast_tokenizer.encode("ATG", return_tensors="pt").cuda(), max_length=max_length,
                            do_sample=True, top_k=top_k, top_p=top_p, num_return_sequences=num_seqs)
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
        try:
            seq_strings.append("".join(to_stop))
        except NameError:
            pass
    if biopy_seq:
        seq_strings = [Seq(s) for s in seq_strings]
    return seq_strings


def generate_fasta_file(file_name, model, fast_tokenizer, max_length=1024, top_k=50, top_p=0.95, num_seqs=5,
                        translate_to_protein=False):
    # generate seq objects
    generated = generate_dna_to_stop(model, fast_tokenizer, max_length=max_length, top_k=top_k, top_p=top_p,
                                     num_seqs=num_seqs, biopy_seq=True)
    if translate_to_protein:
        generated = [s.translate() for s in generated]
    # generate seq records
    records = []
    for n, i in enumerate(generated):
        record = SeqRecord(i,
                           id="MDH_SyntheticSeq_{}".format(n),
                           name="MDH_sequence",
                           description="synthetic malate dehydrogenase",
                           )
        records.append(record)

    with open(file_name, "w") as output_handle:
        SeqIO.write(train_records, output_handle, "fasta")

