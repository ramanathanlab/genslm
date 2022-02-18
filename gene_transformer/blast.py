"""Defining blast utilities to monitor training"""

import statistics
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd  # type: ignore[import]
from typing import Optional, List, Tuple
from Bio import SeqIO  # type: ignore[import]
from Bio.Seq import Seq  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]
from concurrent.futures import ThreadPoolExecutor


class BlastRun:
    """Class to handle blasting single sequence against training sequence database"""

    def __init__(
        self,
        sequence: str,
        database_file: str,
        temp_fasta_dir: Path,
        temp_csv_dir: Path,
    ) -> None:
        self.database_file = database_file

        self.temp_fasta = temp_fasta_dir / f"test_seq{hash(sequence)}.fasta"
        # Need to save sequence as fasta file in order to run Blast
        make_fasta_from_seq(sequence, self.temp_fasta)

        self.temp_csv = temp_csv_dir / f"blast_res{hash(sequence)}.csv"

        # Cannot get scores until blast has been run
        self.ran_blast = False

        # TODO: Is this the correct type?
        self.scores: List[float] = []

        # TODO: Instead of printing a warning message and retuning None
        #       just call the run_blast() and get_scores() functions as needed

    def _warning_message(self) -> None:
        print(
            "Scores have not yet been defined. Make sure"
            " you've called run_blast() and get_scores()."
        )

    def run_blast(self) -> None:
        # run local blastn given parameters in init, REQUIRES LOCAL INSTALLATION OF BLAST
        command = "blastn -query {} -subject {} -out {} -outfmt 10".format(
            self.temp_fasta, self.database_file, self.temp_csv
        )
        subprocess.run(command, shell=True)
        self.ran_blast = True

    def get_scores(self) -> Optional[List[float]]:
        if not self.ran_blast:
            print("Blast has not yet been run. Call the run_blast() method first.")
            return None
        try:
            # Read in csv where blast results were stored
            df = pd.read_csv(self.temp_csv, header=None)
            # Take column which specifies scores
            self.scores = df[11].tolist()
            return self.scores
        except pd.errors.EmptyDataError:
            self.scores = [-1]
            return None

    def get_top_score(self) -> Optional[float]:
        if not self.scores:
            return self.scores[0]

        self._warning_message()
        return None

    def get_mean_score(self) -> Optional[float]:
        if self.scores:
            return statistics.mean(self.scores)

        self._warning_message()
        return None


class BLAST:
    """Class to handle blasting a group of sequences against training sequence database"""

    def __init__(
        self,
        database_file: str,
        blast_dir: Path,
        num_workers: int = 1,
    ) -> None:
        self.database_file = database_file
        self.blast_dir = blast_dir

        self.blast_dir.mkdir(exist_ok=True, parents=True)

        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    def _blast(self, sequence: str, prefix: str) -> Tuple[float, float]:
        """Blast :obj:`sequence` agaisnt :obj:`database_file` using
        the blastn executable in a subprocess call.

        Parameters
        ----------
        sequence : str
            Sequence to blast.
        prefix : str
            Filename prefix for fasta file and blast csv file.

        Returns
        -------
        Tuple[float, float]
            Top score, mean score
        """

        # Write a temporary fasta file
        seq_hash = hash(sequence)
        temp_fasta = self.blast_dir / f"{prefix}-seq-{seq_hash}.fasta"
        temp_csv = self.blast_dir / f"{prefix}-blast-{seq_hash}.csv"
        SeqIO.write(SeqRecord(Seq(sequence)), temp_fasta, "fasta")

        # Run local blastn given parameters in init, REQUIRES LOCAL INSTALLATION OF BLAST
        command = "blastn -query {} -subject {} -out {} -outfmt 10".format(
            temp_fasta, self.database_file, temp_csv
        )
        subprocess.run(command, shell=True)

        try:
            # Read in csv where blast results were stored and take
            # column which specifies the scores
            scores = pd.read_csv(temp_csv, header=None)[11].values
        except pd.errors.EmptyDataError:
            print(f"WARNING: blast did not find a match {temp_csv}")
            return -1.0, -1.0

        top_score: float = scores[0]
        mean_score: float = np.mean(scores)
        return top_score, mean_score

    def run(self, sequences: List[str], prefix: str) -> Tuple[List[float], List[float]]:
        top_scores, mean_scores = [], []
        for seq in sequences:
            top_score, mean_score = self._blast(seq, prefix)
            top_scores.append(top_score)
            mean_scores.append(mean_score)
        return top_scores, mean_scores

    def parallel_run(
        self, sequences: List[str], prefix: str
    ) -> Tuple[List[float], List[float]]:
        top_scores, mean_scores = [], []
        futures = [self._executor.submit(self._blast, seq, prefix) for seq in sequences]
        for fut in futures:
            top_score, mean_score = fut.result()
            top_scores.append(top_score)
            mean_scores.append(mean_score)
        return top_scores, mean_scores


def chunks(lst: str, n: int) -> List[str]:
    """Successive n-sized chunks from lst."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def make_fasta_from_seq(sequence: str, filename: Path) -> None:
    """Generate temporary fasta file for blast search from str sequence"""
    # TODO: Remove this function and chunks in favor of seqs_to_fasta in utils
    #       to use biopython for writing fasta files.
    with open(filename, "w") as f:
        f.write(">Test sequence\n")
        # TODO why are we writing it by chunks of 100?
        for x in chunks(sequence, 100):
            f.write(x)
            f.write("\n")
