"""Validate preparation coverage, lineage and completion without changing data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import ARCHIVED_METADATASET_DIR, METADATASET_DIR, SRC_DIR
from src.utils.artifact_integrity import (
    atomic_json, code_fingerprint, file_sha256, package_versions,
)


TASK_KEYS = ["Seed", "Dataset", "Sample Size"]
MODEL_KEYS = [*TASK_KEYS, "Model"]
DATASET_FEATURES = [
    "class_ent", "eq_num_attr", "gravity", "inst_to_attr", "nr_attr", "nr_bin",
    "nr_class", "nr_cor_attr", "nr_inst", "nr_norm", "nr_outliers", "ns_ratio",
]
MODEL_FEATURES = [
    "Processing Units Number", "Model Type", "Training Complexity",
    "Prediction Complexity", "Regularization", "Robust to Outliers",
    "Representational Capacity",
]
META_COLUMNS = [*MODEL_KEYS, *DATASET_FEATURES, *MODEL_FEATURES, "MCC"]
SCHEMAS = {
    "base_classifier_hyperparameters.csv": ([*MODEL_KEYS, "Model Parameters"], MODEL_KEYS, []),
    "base_classifier_performance.csv": (
        [*MODEL_KEYS, "Model Parameters", "Accuracy", "Precision", "Recall", "F1", "MCC"],
        MODEL_KEYS, ["Accuracy", "Precision", "Recall", "F1", "MCC"],
    ),
    "dataset_metafeatures.csv": ([*TASK_KEYS, *DATASET_FEATURES], TASK_KEYS, DATASET_FEATURES),
    "meta_dataset.csv": (META_COLUMNS, MODEL_KEYS, [*DATASET_FEATURES, *MODEL_FEATURES, "MCC"]),
}


@lru_cache(maxsize=1)
def implementation_context() -> dict:
    files = (
        "experiment_config.py", "paths.py", "utils/artifact_integrity.py",
        "utils/pipeline_integrity.py", "utils/dataset_processing.py",
        "utils/base_classifiers.py", "pipeline/tune_base_classifiers.py",
        "pipeline/evaluate_base_classifiers.py", "pipeline/extract_dataset_metafeatures.py",
        "pipeline/build_meta_dataset.py",
    )
    return {
        "code_sha256": code_fingerprint(SRC_DIR, files),
        "versions": package_versions(("numpy", "pandas", "scikit-learn", "pymfe", "sklearn-genetic-opt")),
    }


def dataset_csv_paths(configurations: list, datasets_dir: Path) -> dict[str, Path]:
    paths, errors = {}, []
    for name, _, _ in configurations:
        candidates = sorted((datasets_dir / name).glob("*.csv"))
        if len(candidates) != 1:
            errors.append(f"{name}: expected exactly one CSV, found {len(candidates)}")
        else:
            paths[name] = candidates[0]
    if errors:
        raise ValueError("Dataset inputs are incomplete or ambiguous:\n" + "\n".join(errors))
    return paths


def preparation_context(raw_inputs: dict[str, Path]) -> dict:
    return {
        "implementation": implementation_context(),
        "datasets": {
            name: {"path": path.relative_to(SRC_DIR).as_posix(), "sha256": file_sha256(path)}
            for name, path in sorted(raw_inputs.items())
        },
    }


def completion_path(path: Path) -> Path:
    return path.with_suffix(".complete.json")


def ensure_output_available(path: Path) -> None:
    candidates = (path, completion_path(path), path.with_suffix(path.suffix + ".tmp"),
                  completion_path(path).with_suffix(".json.tmp"))
    if any(candidate.exists() for candidate in candidates):
        raise FileExistsError(
            f"Output already exists or is partial: {path}. Select a fresh META2PERF_DATA_RUN; "
            "existing files are preserved."
        )


def validate_frame(frame: pd.DataFrame, schema_name: str) -> None:
    columns, keys, numeric = SCHEMAS[schema_name]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{schema_name}: missing columns {missing}")
    if frame.empty or frame[columns].isna().any().any():
        raise ValueError(f"{schema_name}: empty table or missing required values")
    if frame.duplicated(keys).any():
        raise ValueError(f"{schema_name}: duplicate keys {keys}")
    if numeric and not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{schema_name}: non-finite descriptors or target values")


def validate_key_coverage(actual: pd.DataFrame, expected: pd.DataFrame,
                          keys: list[str], label: str) -> None:
    actual_keys = set(actual[keys].itertuples(index=False, name=None))
    expected_keys = set(expected[keys].itertuples(index=False, name=None))
    missing, unexpected = expected_keys - actual_keys, actual_keys - expected_keys
    if missing or unexpected:
        raise ValueError(f"{label}: {len(missing)} missing keys and {len(unexpected)} unexpected keys")


def completion_metadata(path: Path) -> dict | None:
    resolved = path.resolve()
    if resolved.parent == ARCHIVED_METADATASET_DIR.resolve():
        return None
    # An explicit external --data file is not a preparation-stage artifact.
    if METADATASET_DIR.resolve() not in resolved.parents:
        return None
    marker = completion_path(path)
    if not marker.is_file():
        raise ValueError(f"Incomplete or legacy build: {path} has no {marker.name}. Use a fresh build or archived inputs.")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("table_sha256") != file_sha256(path):
        raise ValueError(f"Table does not match its completion metadata: {path}")
    if metadata.get("context", {}).get("implementation") != implementation_context():
        raise ValueError(f"Preparation code or package versions changed since {path} was produced. Use a fresh build.")
    return metadata


def read_pipeline_table(path: Path, schema_name: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_frame(frame, schema_name or path.name)
    metadata = completion_metadata(path)
    if metadata and (metadata["rows"] != len(frame) or metadata["columns"] != list(frame.columns)):
        raise ValueError(f"Table dimensions do not match completion metadata: {path}")
    return frame


def require_preparation_context(path: Path, context: dict) -> None:
    metadata = completion_metadata(path)
    if metadata and metadata["context"] != context:
        raise ValueError(f"Raw datasets or preparation context differ from {path}. Use a fresh build.")


def require_input_lineage(path: Path, inputs: tuple[Path, ...]) -> None:
    metadata = completion_metadata(path)
    expected = {source.name: file_sha256(source) for source in inputs}
    if metadata and metadata.get("inputs") != expected:
        raise ValueError(f"{path} was generated from different input tables. Use a coherent build.")


def shared_preparation_context(paths: tuple[Path, ...]) -> dict:
    contexts = [metadata["context"] for path in paths if (metadata := completion_metadata(path))]
    if contexts and any(context != contexts[0] for context in contexts[1:]):
        raise ValueError("Preparation tables have different raw-data or code contexts.")
    return contexts[0] if contexts else {"implementation": implementation_context(), "datasets": {}}


def mark_complete(path: Path, frame: pd.DataFrame, *, inputs: tuple[Path, ...], context: dict) -> None:
    validate_frame(frame, path.name)
    atomic_json(completion_path(path), {
        "schema_version": 1, "table_sha256": file_sha256(path),
        "rows": len(frame), "columns": list(frame.columns),
        "inputs": {source.name: file_sha256(source) for source in inputs}, "context": context,
    })


def read_meta_dataset(path: Path) -> pd.DataFrame:
    frame = read_pipeline_table(path, "meta_dataset.csv")
    if set(frame.columns) != set(META_COLUMNS):
        raise ValueError("The meta-dataset must contain exactly the documented 24 columns.")
    return frame.loc[:, META_COLUMNS]
