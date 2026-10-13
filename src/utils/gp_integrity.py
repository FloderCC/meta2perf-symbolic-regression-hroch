"""Bind GP caches, checkpoints and summaries to their data and implementation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from src.paths import ARCHIVED_METADATASET_DIR, SRC_DIR
from src.utils.artifact_integrity import atomic_json, content_fingerprint, file_sha256, package_versions


def engine_fingerprint(files: tuple[str, ...]) -> str:
    """Ignore only budget, logging and resume controls when hashing GP stages."""
    mutable_controls = {
        "SEARCH_MAX_ITERATIONS", "RESUME_FROM_CHECKPOINT", "SAVE_CHECKPOINT_EVERY",
        "SAVE_HOF_CSV", "SAVE_FINAL_POPULATION_CSV", "LOG_METRICS_EVERY",
    }
    fingerprints = {}
    for name in files:
        path = SRC_DIR / name
        if not name.startswith("experiments/genetic_programming/"):
            fingerprints[name] = file_sha256(path)
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        body = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if targets & mutable_controls:
                    continue
                if targets & {"CONFIG", "FIXED_BASE_CONFIG"} and isinstance(node.value, ast.Dict):
                    entries = [
                        (key, value) for key, value in zip(node.value.keys, node.value.values)
                        if not (isinstance(key, ast.Constant)
                                and key.value in {"max_iterations", "LOG_METRICS_EVERY"})
                    ]
                    node.value.keys = [key for key, _ in entries]
                    node.value.values = [value for _, value in entries]
            body.append(node)
        tree.body = body
        fingerprints[name] = ast.dump(tree, include_attributes=False)
    return content_fingerprint(fingerprints)


def gp_context(data_path: Path, feature_names: list[str]) -> dict:
    # All stages hash the same engine files, allowing compatible screening
    # checkpoints to continue in refinement while detecting code changes.
    files = (
        "paths.py", "utils/artifact_integrity.py", "utils/pipeline_integrity.py",
        "utils/gp_integrity.py", "utils/gp_initialization.py",
        "utils/gp_operators.py", "experiments/genetic_programming/search_hyperparameters.py",
        "experiments/genetic_programming/refine_top_configurations.py",
        "experiments/genetic_programming/train_configuration_01.py",
        "experiments/genetic_programming/train_configuration_02.py",
        "experiments/genetic_programming/train_configuration_03.py",
    )
    context = {
        "schema_version": 1, "data_sha256": file_sha256(data_path),
        "feature_order": feature_names, "code_sha256": engine_fingerprint(files),
        "versions": package_versions(("numpy", "pandas", "scikit-learn", "deap", "pyBlindOpt", "EmptySpaceSearch", "torann")),
    }
    return {**context, "fingerprint": content_fingerprint(context)}


def summary_context_columns(context: dict) -> dict:
    return {
        "context_fingerprint": context["fingerprint"], "data_sha256": context["data_sha256"],
        "feature_order_json": json.dumps(context["feature_order"]),
        "code_sha256": context["code_sha256"], "versions_json": json.dumps(context["versions"], sort_keys=True),
    }


def prepare_gp_run(directory: Path, context: dict) -> None:
    manifest = directory / "run_context.json"
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8")) != context:
            raise ValueError(f"GP run context changed: {directory}. Choose a fresh META2PERF_RUN_ID.")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Legacy or incomplete GP run: {directory}. Preserve it and choose a fresh META2PERF_RUN_ID.")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, context)


def validate_summary_context(frame, path: Path, context: dict, *, allow_legacy_archive: bool = False) -> None:
    if "context_fingerprint" in frame.columns:
        if frame["context_fingerprint"].isna().any() or not frame["context_fingerprint"].eq(context["fingerprint"]).all():
            raise ValueError(f"GP summary belongs to different data, feature order, code or versions: {path}")
        return
    archived_data = ARCHIVED_METADATASET_DIR / "meta_dataset.csv"
    if (allow_legacy_archive and path.parent.name == "archived"
            and context["data_sha256"] == file_sha256(archived_data)):
        # Historical hyperparameters remain readable for the historical data.
        # Their old caches/checkpoints are never accepted as current artifacts.
        return
    raise ValueError(f"GP summary has no compatibility metadata: {path}. Use a fresh screening run.")
