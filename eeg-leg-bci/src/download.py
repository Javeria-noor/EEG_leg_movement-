"""
download.py — Download OpenNeuro ds004362 (BCI2000 motor imagery) data.

Usage:
    python src/download.py --subjects 10 --output data/
    python src/download.py --all --output data/
"""
import argparse
from pathlib import Path

from utils import get_logger

logger = get_logger(__name__)

DATASET_ID = "ds004362"
DATASET_VERSION = "1.0.0"

# Runs used for the lower-limb (both-feet) imagery task in this project.
LEG_RUNS = [6, 10, 14]


def download_subjects(n_subjects: int, output_dir: Path, all_subjects: bool = False) -> None:
    """
    Download a subset (or all 109) subjects of ds004362 via openneuro-py.

    Only the runs relevant to lower-limb motor imagery (6, 10, 14) are
    fetched per subject to keep the download lightweight.
    """
    try:
        import openneuro
    except ImportError as exc:
        raise ImportError(
            "openneuro-py is required for downloading. "
            "Install with: pip install openneuro-py"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_subjects:
        logger.info("Downloading FULL dataset %s (109 subjects)...", DATASET_ID)
        openneuro.download(dataset=DATASET_ID, target_dir=str(output_dir))
        return

    include_patterns = []
    for sub_idx in range(1, n_subjects + 1):
        sub_id = f"sub-{sub_idx:03d}"
        for run in LEG_RUNS:
            include_patterns.append(f"{sub_id}/**/*R{run:02d}*")
        include_patterns.append(f"{sub_id}/*.json")
        include_patterns.append(f"{sub_id}/*.tsv")

    logger.info(
        "Downloading %d subjects (runs %s) from %s into %s",
        n_subjects, LEG_RUNS, DATASET_ID, output_dir,
    )
    openneuro.download(
        dataset=DATASET_ID,
        target_dir=str(output_dir),
        include=include_patterns,
    )
    logger.info("Download complete.")


def main():
    parser = argparse.ArgumentParser(description="Download OpenNeuro ds004362")
    parser.add_argument("--subjects", type=int, default=10,
                         help="Number of subjects to download (default: 10)")
    parser.add_argument("--all", action="store_true",
                         help="Download the full 109-subject dataset")
    parser.add_argument("--output", type=str, default="data/",
                         help="Output directory (default: data/)")
    args = parser.parse_args()

    download_subjects(
        n_subjects=args.subjects,
        output_dir=Path(args.output),
        all_subjects=args.all,
    )


if __name__ == "__main__":
    main()
