from Bio.Seq import Seq


def generate_protein_seq(fast_tokenizer, model):
    s = fast_tokenizer.decode(model.generate(fast_tokenizer.encode("ATG", return_tensors="pt").cuda(), max_length=1024,
                                             temperature=0.7)[0], skip_special_tokens=True)
    s = s.replace(" ", "")
    s = s.replace(".", "")
    seq = Seq(s)
    return seq.translate(to_stop=True)

