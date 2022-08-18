#!/usr/bin/env python
# coding: utf-8

import os
import Bio
import time
import numpy as np
from Bio import pairwise2
from Bio import Seq
from Bio import SeqIO
from Bio.Align import substitution_matrices




def OutlierClose(v, target=1280, perc = 50):
    r"""Check there is an element in v close to the target.

    Determine if there is value in v closer to the target 
    by a percent perc than the next nearest neighbor.
    
    Parameters
    ----------
    v : array_like, contains int/float
        Values to be compared to the target.
    target : int/float
        Target to be compared to.
    perc : int/float
        Percentage by which the closest value should be closer to the target to trigger true.
        
    Returns
    -------
    bool
        Returns true if there is a value in v that is close enough to trigger true
        according to the threshold defined by d1/d2*100 < perc.

    Notes
    -----
    Used to avoid having to compute alignments if only one candidate ORF could 
    reasonably correspond to the target. 
    
    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [434, 290, 800]
    >>> OutlierClose(a, target = 420)
    True
    >>>OutlierClose(a, target = 360)
    False
    """

    diffs = np.abs(v-target)
    potential = min(diffs)
    closest = min(diffs[diffs!=min(diffs)])
    d1 = potential-target
    d2 = closest-target
    return d1/d2*100 < perc



def find_orfs_with_trans(seq, trans_table, min_protein_length):
    r"""Locate the ORFs within the sequence of a minimum protein length.

    Find open-reading frames in a given sequence (seq) according to the translation
    table (trans_table) with minimum length (min_protein_length). Open-reading frames
    are defined by the trans_table by specifying start and end codons. Originally
    included in Biopython Cookbook. 
    
    http://biopython.org/DIST/docs/tutorial/Tutorial.html#sec384
    
    Parameters
    ----------
    seq : Seq, Biopython sequence object
        Sequence to be searched.
    trans_table : int
        Translation table to be utilized by Biopython. Table 11 is typical for viral genomes.
    min_pro_length : int/float
        Minimum length of the translated protein to be included.
        
    Returns
    -------
    answer : list, contains Biopython sequence objects of translated open reading frames
                   as aminoacids.
    
    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.
    
    >>> orf_list = find_orfs_with_trans(record.seq, table, min_pro_len)
    >>> for start, end, strand, pro in orf_list:
        print(
            "%s...%s - length %i, strand %i, %i:%i"
            % (pro[:30], pro[-3:], len(pro), strand, start, end)
        )
    NQIQGVICSPDSGEFMVTFETVMEIKILHK...GVA - length 355, strand 1, 41:1109
    WDVKTVTGVLHHPFHLTFSLCPEGATQSGR...VKR - length 111, strand -1, 491:827
    KSGELRQTPPASSTLHLRLILQRSGVMMEL...NPE - length 285, strand 1, 1030:1888
    RALTGLSAPGIRSQTSCDRLRELRYVPVSL...PLQ - length 119, strand -1, 2830:3190
    RRKEHVSKKRRPQKRPRRRRFFHRLRPPDE...PTR - length 128, strand 1, 3470:3857
    GLNCSFFSICNWKFIDYINRLFQIIYLCKN...YYH - length 176, strand 1, 4249:4780
    RGIFMSDTMVVNGSGGVPAFLFSGSTLSSY...LLK - length 361, strand -1, 4814:5900
    VKKILYIKALFLCTVIKLRRFIFSVNNMKF...DLP - length 165, strand 1, 5923:6421
    LSHTVTDFTDQMAQVGLCQCVNVFLDEVTG...KAA - length 107, strand -1, 5974:6298
    GCLMKKSSIVATIITILSGSANAASSQLIP...YRF - length 315, strand 1, 6654:7602
    IYSTSEHTGEQVMRTLDEVIASRSPESQTR...FHV - length 111, strand -1, 7788:8124
    WGKLQVIGLSMWMVLFSQRFDDWLNEQEDA...ESK - length 125, strand -1, 8087:8465
    TGKQNSCQMSAIWQLRQNTATKTRQNRARI...AIK - length 100, strand 1, 8741:9044
    QGSGYAFPHASILSGIAMSHFYFLVLHAVK...CSD - length 114, strand -1, 9264:9609
    """
    
    answer = []
    seq_len = len(seq)
    for strand, nuc in [(+1, seq), (-1, seq.reverse_complement())]:
        for frame in range(3):
            trans = nuc[frame:].translate(trans_table)
            trans_len = len(trans)
            aa_start = 0
            aa_end = 0
            while aa_start < trans_len:
                aa_end = trans.find("*", aa_start)
                if aa_end == -1:
                    aa_end = trans_len
                if aa_end - aa_start >= min_protein_length:
                    if strand == 1:
                        start = frame + aa_start * 3
                        end = min(seq_len, frame + aa_end * 3 + 3)
                    else:
                        start = seq_len - frame - aa_end * 3 - 3
                        end = seq_len - frame - aa_start * 3
                    answer.append((start, end, strand, trans[aa_start:aa_end]))
                aa_start = aa_end + 1
    answer.sort()
    return answer

def getBestMatchORF(seq, target_protein, min_pro_len=800, table = 11, max_pro_len = 1800):
    r"""Use heuristics to find best matching ORF in seq for target_protein.

    Implements length-based heuristics and alignments strategies to find the Open-Reading Frame
    closest related to target_protein in the genome defined by seq.
    
    Parameters
    ----------
    seq : Seq, Biopython sequence object
        Sequence to be searched.
    target_protein : Str
        Aminoacid sequence of target protein. 
    min_pro_length : int/float
        Minimum length of the translated protein to be included.
    table : int
        Translation table to be utilized by Biopython. Table 11 is typical for viral genomes.
    max_pro_length : int/float
        Maximum length of the translated protein to be included.    
        
    Returns
    -------
    dict
        Dictionary containing:
            'Name' - Sequence name
            'Metadata' - Entire top line of FASTA sequence containing metadata.
            'Strand' - Strand on which the open-reading frame is present.
            'Start' - Index of gene beginning in basepairs.
            'End' - Index of gene ending in basepairs.
            'Gene_Sequence' - Gene sequence in basepairs.
            'Protein_Sequence' - Protein sequence in aminoacids.
            'Score' - If necessary, computed score for alignment of selected ORF. 
            'Timings' - for profiling purposes, time in each step of algorithm while processing.

    Notes
    -----
    Used to speed up target ORF extraction based on heuristics. 
    
    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [434, 290, 800]
    >>> OutlierClose(a, target = 420)
    True
    >>>OutlierClose(a, target = 360)
    False
    """
    splitted = seq.split('\n')
    seq = ''.join(splitted[1:])
    timing = {}
    timing['Start_ORFs'] = time.perf_counter()
    orf_list = find_orfs_with_trans(Seq.Seq(seq), table, min_pro_len)
    orf_list = [ans for ans in orf_list if ((len(ans[3]) < max_pro_len) and (len(ans[3])>min_pro_len))]
    timing['Finish_ORFs'] = time.perf_counter()
    proteins = [ans[3] for ans in orf_list if ((len(ans[3]) < max_pro_len) and (len(ans[3])>min_pro_len))]
    scores = []
    timing['Start_Aligns'] = time.perf_counter()
    ## 
    if len(proteins) == 1:
        maxIndex = 0
        maxScore = 0
        orf_list = [ans for ans in orf_list if ((len(ans[3]) < max_pro_len) and (len(ans[3])>min_pro_len))]
    else:
        if len(proteins) == 0:
            ## If no proteins found in restricted range, drop top and bot
            orf_list = find_orfs_with_trans(Seq.Seq(seq), table, min_pro_len = 200)
            proteins = [ans[3] for ans in orf_list]
            
        lengths = np.array([len(pro) for pro in proteins])
        dists = np.abs(lengths-len(target_protein))
        if OutlierClose(lengths, target = len(target_protein)):
            maxIndex = np.where(dists==min(dists))[0][0]
            maxScore = pairwise2.align.globalxx(target_protein, proteins[maxIndex])
        else:
            for pro in proteins:
                alignments = pairwise2.align.globalxx(target_protein, pro)
                scores += [alignments[0].score]
            timing['Finish_Aligns'] = time.perf_counter()
            maxScore = max(scores)
            maxIndex = scores.index(maxScore)

        
    match = proteins[maxIndex]
    final_ans = orf_list[maxIndex]
    timing['Start_Output'] = time.perf_counter()
    answer = {}
    heading = splitted[0].split('|')
    answer['Name'] = heading[0]
    answer['Metadata'] = splitted[0]
    if final_ans[2] == 1:
        answer['Strand'] = 1
    else:
        answer['Strand'] = -1
    answer['Start'] = final_ans[0]
    answer['End'] = final_ans[1]
    answer['Gene_Sequence'] = seq[final_ans[0]:final_ans[1]]
    answer['Protein_Sequence'] = Seq.Seq(seq[final_ans[0]:final_ans[1]]).translate(table)
    answer['Score'] = maxScore
    timing['Finish_Output'] = time.perf_counter()
    answer['Timings'] = timing
    return answer

def fastify(seqstring, l = 70):
    r"""Format string sequence into FASTA format with line length l.

    Parameters
    ----------
    seqstring : str
        Sequence to be processed.
    l : str
        Length of each line for FASTA file.
    Returns
    -------
    fasta : str
        Gene sequence split according to FASTA format.
    """ 
    k = 0
    fasta = ''
    while k < len(seqstring):
        fasta += seqstring[k:(k+l)]+'\n'
        k += l
    return fasta

def main(args):
    file = args.fasta
    seqs = list(SeqIO.parse(p / file, "fasta"))

    # Nucleocapsid
    table = args.table
    target = args.target
    target_seq=''.join(target.split('\n')[1:])
    targ_record = Seq.Seq(target_seq)
    targ_protein = targ_record.seq.translate(table)
    min_pro_len = args.min_pro_len # Minimum Protein Length
    max_pro_len = args.max_pro_len # Maximum Protein Length
    output = ''
    for seq in seqs:
        answer = getBestMatchORF(seq, targ_protein, min_pro_len=min_pro_len, max_pro_len=max_pro_len)
        output += fastify(answer["Metadata"]+'\n'+answer['Gene_Sequence'])
    f = open(args.output_dir, 'w')
    f.write(output)
    return 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True, help="Path to the fasta file input.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the output directory, will be made if it does not exist",
    )
    parser.add_argument(
        "--min_pro_len",
        type=int,
        help="Length of shortest open-reading frame to consider.",
        default=0,
    )
    parser.add_argument(
        "--max_pro_len",
        type=int,
        help="Length of longest open-reading frame to consider.",
        default=1000**3,
    )

    parser.add_argument(
        "--table",
        type=int,
        help="Codon table to be utilized by Biopython dependant on organism.",
        default=11,
    )
    
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Representative ORF to be identified in each sequence.",
    )
    args = parser.parse_args()
    main(args)

