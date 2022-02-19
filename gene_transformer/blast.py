"""Defining blast utilities to monitor training"""

import subprocess
from pathlib import Path
import numpy as np
import pandas as pd  # type: ignore[import]
from typing import List, Tuple
from Bio import SeqIO  # type: ignore[import]
from Bio.Seq import Seq  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]
from concurrent.futures import ThreadPoolExecutor


class ParallelBLAST:
    """Class to handle blasting a group of sequences against
    a sequence database in parallel"""

    def __init__(
        self,
        database_file: str,
        blast_dir: Path,
        blast_exe_path: Path = Path("blastn"),
        num_workers: int = 1,
    ) -> None:
        """Runs BLAST using the blastn conda utility.

        Parameters
        ----------
        database_file : str
            The fasta file containing sequences to blast against.
        blast_dir : Path
            Output directory to write fasta files and blast csv files.
        blast_exe_path : Path, optional
            Path to blast executable, by default Path("blastn")
        num_workers : int, optional
            Number of threads used to spawn blast processes, by default 1
        """
        self.database_file = database_file
        self.blast_dir = blast_dir
        self.blast_exe_path = blast_exe_path

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
        command = "{} -query {} -subject {} -out {} -outfmt 10".format(
            self.blast_exe_path, temp_fasta, self.database_file, temp_csv
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
        futures = [self._executor.submit(self._blast, seq, prefix) for seq in sequences]
        for fut in futures:
            top_score, mean_score = fut.result()
            top_scores.append(top_score)
            mean_scores.append(mean_score)
        return top_scores, mean_scores
