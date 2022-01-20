"""Defining blast utilities to monitor training"""

import statistics
import subprocess
from pathlib import Path
import pandas as pd


class BlastRun:
    """Class to handle blasting single sequence against training sequence database"""

    def __init__(
        self,
        sequence,
        database_file,
        temp_fasta_dir="/tmp/mzvyagin/",
        temp_csv_dir="/tmp/mzvyagin/",
    ):
        self.sequence = sequence
        self.database_file = database_file

        self.temp_fasta_dir = temp_fasta_dir
        self.temp_fasta = Path(
            str(temp_fasta_dir) + "/test_seq{}.fasta".format(hash(sequence))
        )
        # need to save sequence as fasta file in order to run Blast
        make_fasta_from_seq(sequence, self.temp_fasta)

        self.temp_csv_dir = temp_csv_dir
        self.temp_csv = Path(
            str(temp_csv_dir) + "/blast_res{}.csv".format(hash(sequence))
        )

        # cannot get scores until blast has been run
        self.ran_blast = False

        self.scores = []
        self.top_score = None

    def run_blast(self):
        # run local blastn given parameters in init, REQUIRES LOCAL INSTALLATION OF BLAST
        command = "blastn -query {} -subject {} -out {} -outfmt 10".format(
            self.temp_fasta, self.database_file, self.temp_csv
        )
        subprocess.run(command, shell=True)
        self.ran_blast = True

    def get_scores(self):
        try:
            assert self.ran_blast
            # read in csv where blast results were stored
            df = pd.read_csv(self.temp_csv, header=None)
            # take column which specifies scores
            self.scores = df[11].tolist()
            return self.scores
        except AssertionError:
            print("Blast has not yet been run. Call the run_blast() method first.")
            return None
        except pd.errors.EmptyDataError:
            self.scores = [-1]

    def get_top_score(self):
        try:
            assert self.scores
            return self.scores[0]
        except AssertionError:
            print(
                "Scores have not yet been defined. Make sure you've called run_blast() and get_scores()."
            )
            return None

    def get_mean_score(self):
        try:
            assert self.scores
            return statistics.mean(self.scores)
        except AssertionError:
            print(
                "Scores have not yet been defined. Make sure you've called run_blast() and get_scores()."
            )
            return None


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # from stackoverflow: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def make_fasta_from_seq(sequence, filename="/tmp/test_seq.fasta"):
    """Generate temporary fasta file for blast search from str sequence"""
    with open(filename, "w") as f:
        f.write(">Test sequence\n")
        for x in chunks(sequence, 100):
            f.write(x)
            f.write("\n")
