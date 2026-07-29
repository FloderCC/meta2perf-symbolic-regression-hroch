"""Repository paths shared by the experiment scripts."""

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SRC_DIR.parent
DATASETS_DIR = SRC_DIR / "datasets"
RESULTS_DIR = SRC_DIR / "results"
LOGS_DIR = SRC_DIR / "logs"
PLOTS_DIR = SRC_DIR / "plots"
PLOTS_PDF_DIR = PLOTS_DIR / "pdf"
PLOTS_PNG_DIR = PLOTS_DIR / "png"


def ensure_output_directories() -> None:
    """Create output directories used by scripts when they are absent."""

    for directory in (RESULTS_DIR, LOGS_DIR, PLOTS_PDF_DIR, PLOTS_PNG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
