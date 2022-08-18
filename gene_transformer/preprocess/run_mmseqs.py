import subprocess
import functools
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def _compute_number_of_clusters(tsv_path: Path) -> int:
    """Parse TSV output file from mmseqs2 and compute number of clusters.

    Parameters
    ----------
    tsv_path : Path
        Path to output TSV file.

    Returns
    -------
    int
        The number of sequence clusters.
    """
    # Collect each line of the TSV file in a list.
    lines = tsv_path.read_text().strip().split("\n")
    # Get the sequences in the first column.
    cluster_centers = set(line.split()[0] for line in lines)
    # Return the number of unique sequences in the first column.
    return len(cluster_centers)


def mmseqs2_easy_cluster(
    similarity: float,
    fasta: Path,
    output_dir: Path,
    mmseqs_exe: str = "mmseqs",
    mmseqs_threads: int = 10,
) -> int:
    """Run easy-cluster mmseqs2 executable and return the number of non redundant sequences.

    Parameters
    ----------
    similarity : float
        min-seq-id similarity for mmseqs2.
    fasta : Path
        Path to fasta file.
    output_dir : Path
        Output directory for mmseqs2 executable.
    mmseqs_exe : str
        mmseqs executable path.
    mmseqs_threads : int
        How many cores in mmseqs.
    """

    # Setup up and run mmseqs2 executable
    output_dir.mkdir(exist_ok=True)
    out_dir_and_files = output_dir / f"sim{similarity}"
    temp_dir = output_dir / "temp"
    command = (
        f"{mmseqs_exe} easy-cluster {fasta} {out_dir_and_files} {temp_dir}"
        f" --min-seq-id {similarity} --threads {mmseqs_threads} --remove-tmp-files"
    )
    proc = subprocess.run(command.split(), capture_output=True)  # ignore verbose output

    if proc.returncode == 0:
        print(f"\n\nSuccesfully clustered input fasta file to: {output_dir}")
    else:
        raise RuntimeError("MMSEQS did not sucessfully complete")

    # Determine number of clusters in data
    tsv_file = list(output_dir.glob("*.tsv"))[0]
    return _compute_number_of_clusters(tsv_file)


def sequence_identity_thresholding(
    fasta: Path,
    output_dir: Path,
    mmseqs_exe: str = "mmseqs",
    start: int = 10,
    stop: int = 100,
    step: int = 5,
    num_workers: int = 1,
    mmseqs_threads: int = 10,
) -> Tuple[List[float], List[int]]:
    """Compute number of clusters for different similarity thresholds.

    Parameters
    ----------
    fasta : Path
        Path to fasta file.
    output_dir : Path
        Output directory for mmseqs2 executable.
    similarity : float
        min-seq-id similarity for mmseqs2.
    mmseqs_exe : str
        mmseqs executable path.
    start : int, optional
        Starting threshold as percentage, by default 10
    stop : int, optional
        Stoping threshold as percentage, by default 100
    step : int, optional
        Threshold step to increment by as percentage, by default 5
    """
    similarities = list(i / 100 for i in range(start, stop, step))
    # num_clusters = [
    #     mmseqs2_easy_cluster(fasta, output_dir, similarity, mmseqs_exe)
    #     for similarity in tqdm(similarities)
    # ]

    func = functools.partial(
        mmseqs2_easy_cluster,
        fasta=fasta,
        output_dir=output_dir,
        mmseqs_exe=mmseqs_exe,
        mmseqs_threads=mmseqs_threads,
    )

    num_clusters = []
    chunksize = max(1, len(similarities) // num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for num_cluster in executor.map(func, similarities, chunksize=chunksize):
            num_clusters.append(num_cluster)

    return similarities, num_clusters


def mmseqs2_parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--fasta", type=Path, required=True, help="Path to the fasta file input"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the output directory, will be made if it does not exist",
    )
    parser.add_argument(
        "--mmseqs",
        type=str,
        help="Path to MMSEQS program",
        default="mmseqs",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.5,
        help="Similarity threshold to run mmseqs with",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = mmseqs2_parse_args()
    mmseqs2_easy_cluster(args.fasta, args.output_dir, args.similarity, args.mmseqs)
