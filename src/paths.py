"""Shared input, archive, cache and run paths, all contained in src."""

from datetime import datetime, timezone
import os
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SRC_DIR / "datasets"
OUTPUTS_DIR = SRC_DIR / "outputs"
CACHE_DIR = SRC_DIR / "cache"
GP_INITIAL_POPULATIONS_DIR = CACHE_DIR / "gp_initial_populations"


def _output_label(value: str) -> str:
    """Keep configurable run labels within their parent and protect archives."""
    if (not value or value in {".", ".."} or value.casefold() == "archived"
            or any(character in value for character in '/\\:<>"|?*')
            or value.endswith((".", " "))):
        raise ValueError("A run id must be a directory name other than 'archived'.")
    return value


RUN_ID = _output_label(
    os.environ.get("META2PERF_RUN_ID")
    or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
)
METADATASET_RUN_ID = _output_label(os.environ.get("META2PERF_DATA_RUN", "current"))
METADATASET_DIR = OUTPUTS_DIR / "metadataset"
ARCHIVED_METADATASET_DIR = METADATASET_DIR / "archived"
METADATASET_OUTPUT_DIR = METADATASET_DIR / METADATASET_RUN_ID


def _metadataset_input_directory() -> Path:
    """Select one input bank for the whole process; never fall back per file."""
    source = os.environ.get("META2PERF_INPUT_SOURCE", "auto").lower()
    if source not in {"auto", "build", "archived"}:
        raise ValueError("META2PERF_INPUT_SOURCE must be auto, build or archived.")
    if source == "archived":
        return ARCHIVED_METADATASET_DIR
    if (source == "build" or "META2PERF_DATA_RUN" in os.environ
            or (METADATASET_OUTPUT_DIR.is_dir() and any(METADATASET_OUTPUT_DIR.iterdir()))):
        return METADATASET_OUTPUT_DIR
    return ARCHIVED_METADATASET_DIR


METADATASET_INPUT_DIR = _metadataset_input_directory()


def metadataset_input_path(filename: str) -> Path:
    """Resolve a required table within the single selected input bank."""
    candidate = METADATASET_INPUT_DIR / filename
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Missing input table: {candidate}. Complete the selected build or set "
            "META2PERF_INPUT_SOURCE=archived to use the historical tables explicitly."
        )
    return candidate


# Resolving this constant must not require a final table while a preparation
# writer is starting. Readers validate existence and completion when opening it.
META_DATASET_PATH = METADATASET_INPUT_DIR / "meta_dataset.csv"


def output_run_directory(*parts: str) -> Path:
    """Locate a run without creating directories or writing any files."""
    return OUTPUTS_DIR.joinpath(*parts, RUN_ID)


def latest_output_path(*parts: str, filename: str) -> Path:
    """Read the most recently written run artifact, or its archived counterpart."""
    parent = OUTPUTS_DIR.joinpath(*parts)
    candidates = []
    if parent.is_dir():
        candidates = [
            directory / filename
            for directory in parent.iterdir()
            if directory.is_dir() and directory.name != "archived"
            and (directory / filename).is_file()
        ]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)
    return parent / "archived" / filename


def ensure_output_directories(*directories: Path) -> None:
    """Create only the explicitly requested output directories."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
