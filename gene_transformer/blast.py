"""Defining blast utilities to monitor training"""

import statistics
import subprocess
from pathlib import Path
import pandas as pd  # type: ignore[import]
from typing import Optional, List


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
        if self.scores:
            return self.scores[0]

        self._warning_message()
        return None

    def get_mean_score(self) -> Optional[float]:
        if self.scores:
            return statistics.mean(self.scores)

        self._warning_message()
        return None


def chunks(lst: str, n: int) -> List[str]:
    """Successive n-sized chunks from lst."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def make_fasta_from_seq(sequence: str, filename: Path) -> None:
    """Generate temporary fasta file for blast search from str sequence"""
    with open(filename, "w") as f:
        f.write(">Test sequence\n")
        # TODO why are we writing it by chunks of 100?
        for x in chunks(sequence, 100):
            f.write(x)
            f.write("\n")
