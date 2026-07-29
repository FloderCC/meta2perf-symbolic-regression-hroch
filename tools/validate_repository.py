"""Dependency-free structural checks for the publication repository."""

from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = SRC / "results"

EXPECTED_CSVS = {
    "base_classifier_hyperparameters.csv": (25_836, 5),
    "base_classifier_performance.csv": (25_836, 13),
    "dataset_metafeatures.csv": (1_615, 57),
    "meta_dataset.csv": (25_836, 24),
    "joint_random_forest_shap.csv": (19, 2),
}


def validate_python_syntax() -> None:
    for path in (SRC, ROOT / "tools"):
        for python_file in path.rglob("*.py"):
            ast.parse(
                python_file.read_text(encoding="utf-8"),
                filename=str(python_file),
            )


def csv_dimensions(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle)
        header = next(rows)
        return sum(1 for _ in rows), len(header)


def validate_archived_results() -> None:
    for filename, expected_dimensions in EXPECTED_CSVS.items():
        actual_dimensions = csv_dimensions(RESULTS / filename)
        assert actual_dimensions == expected_dimensions, (
            f"{filename}: expected {expected_dimensions}, got {actual_dimensions}"
        )

    with (RESULTS / "meta_dataset.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = csv.DictReader(handle)
        mcc = [float(row["MCC"]) for row in rows]
    assert sum(value > 0 for value in mcc) == 23_323
    assert sum(value == 0 for value in mcc) == 1_960
    assert sum(value < 0 for value in mcc) == 553


def validate_dataset_documentation() -> None:
    dataset_readmes = sorted((SRC / "datasets").glob("*/README.md"))
    assert len(dataset_readmes) == 65, (
        f"Expected 65 per-dataset README files, found {len(dataset_readmes)}"
    )


def validate_withdrawn_analysis_is_absent() -> None:
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in SRC.glob("*.py")
    )
    assert "MinMaxScaler" not in source_text
    assert not list(SRC.glob("*_f_s.py"))
    assert not list((SRC / "logs").glob("*_f_s.txt"))


def main() -> None:
    validate_python_syntax()
    validate_archived_results()
    validate_dataset_documentation()
    validate_withdrawn_analysis_is_absent()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
