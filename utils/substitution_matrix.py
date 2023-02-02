from pathlib import Path
from argparse import ArgumentParser

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Align import AlignInfo
from Bio import SubsMat


def main(args):
    # fasta_file = Path("utils/PGF_00000002.ffn")
    seq_a = SeqRecord(Seq("AAAACGT"), id="Alpha")
    seq_b = SeqRecord(Seq("AAAC-GT"), id="Beta")
    seq_c = SeqRecord(Seq("AAAAGGT"), id="Gamma")
    align = MultipleSeqAlignment(
        [seq_a, seq_b, seq_c],
        annotations={"tool": "demo"},
        column_annotations={"stats": "CCCXCCC"},
    )

    print(align)
    print(align.substitutions)

    summary_align = AlignInfo.SummaryInfo(align.substitutions)
    print(summary_align.ic_vector)
    replace_info = summary_align.replacement_dictionary()

    print(replace_info)

    mat = SubsMat.SeqMat(replace_info)
    print(mat)


if __name__ == "__main__":
    parser = ArgumentParser()

    args = parser.parse_args()
    main(args)
