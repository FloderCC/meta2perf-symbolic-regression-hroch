"""Select reference-regressor configurations on the training partition only.

Available archived configurations are reused with their recorded feature set.
Only missing joint DT/GB/MLP configurations use the four-fold negative-MSE
search from benchmark.py. Winners are frozen before the subsequent grouped
fits; this is not a nested LODO hyperparameter search.
"""

from __future__ import annotations

import copy
import ast
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
import warnings

from joblib import parallel_config
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from threadpoolctl import threadpool_limits

from src.equations.hroch import FEATURE_NAMES
from src.experiments.baselines.benchmark import PARAMETER_GRIDS
from src.experiments.baselines.benchmark_lodo import (
    DATASET_ONLY_SOURCE, JOINT_RF_SOURCE, MODEL_CLASSES, make_model,
)
from src.utils.artifact_integrity import (
    atomic_json, code_fingerprint, content_fingerprint, file_sha256, package_versions,
)
from src.utils.pipeline_integrity import DATASET_FEATURES


ROOT = Path(__file__).resolve().parents[3]
SELECTION_CODE_FILES = (
    "src/experiments/baselines/select_configurations.py",
    "src/experiments/baselines/benchmark.py",
    "src/experiments/baselines/benchmark_lodo.py",
    "src/equations/hroch.py", "src/utils/pipeline_integrity.py",
    "src/utils/artifact_integrity.py",
)
SEED = 42
SCORING = "neg_mean_squared_error"


def _json_ready(value):
    return json.loads(json.dumps(value, allow_nan=False))


def parameter_grids(models=None, *, smoke_test=False):
    names = list(MODEL_CLASSES) if models is None else list(models)
    if not names or len(names) != len(set(names)) or set(names) - set(MODEL_CLASSES):
        raise ValueError("Choose distinct documented reference regressors.")
    grids = {name: copy.deepcopy(PARAMETER_GRIDS[name]) for name in names}
    if smoke_test:
        grids = {name: {key: [values[0]] for key, values in grid.items()}
                 for name, grid in grids.items()}
    return grids


def historical_configuration_sources(feature_set):
    if feature_set not in ("joint", "dataset_only"):
        raise ValueError("feature_set must be joint or dataset_only.")
    source = JOINT_RF_SOURCE if feature_set == "joint" else DATASET_ONLY_SOURCE
    return {source: file_sha256(ROOT / source)}


def _read_archived_parameters(feature_set):
    source = JOINT_RF_SOURCE if feature_set == "joint" else DATASET_ONLY_SOURCE
    parameters, active_model = {}, None
    for line in (ROOT / source).read_text(encoding="utf-8").splitlines():
        model_match = re.search(r"Results for(?: the model)? ([A-Za-z]+):", line)
        if model_match:
            active_model = model_match.group(1)
        if "Selected hyperparameters: " in line:
            literal = line.split("Selected hyperparameters: ", 1)[1]
            if active_model is None or active_model in parameters:
                raise ValueError(f"Ambiguous archived reference selection in {source}.")
            selected = {} if literal == "Default parameters (no tuning)" else ast.literal_eval(literal)
            if not isinstance(selected, dict):
                raise ValueError(f"Invalid archived reference parameters in {source}.")
            parameters[active_model] = _json_ready(selected)
    expected = {"RandomForestRegressor"} if feature_set == "joint" else set(MODEL_CLASSES)
    if set(parameters) != expected:
        raise ValueError(f"Archived {feature_set} configurations do not cover the documented models.")
    return parameters, {"source": source, "source_sha256": file_sha256(ROOT / source),
                        "selection_features": feature_set,
                        "original_cv_score_available": False}


def selection_plan(models=None, *, feature_set="joint", cv=4, smoke_test=False):
    if cv != 4:
        raise ValueError("The documented reference-regressor protocol uses four-fold CV.")
    grids = parameter_grids(models, smoke_test=smoke_test)
    if feature_set not in ("joint", "dataset_only"):
        raise ValueError("feature_set must be joint or dataset_only.")
    counts = {}
    for name, grid in grids.items():
        reuse = feature_set == "dataset_only" or name == "RandomForestRegressor"
        search_grid = grid if grid and not reuse else {}
        counts[name] = {
            "n_candidates": len(ParameterGrid(search_grid)) if search_grid else 0,
            "cv_fits": cv * len(ParameterGrid(search_grid)) if search_grid else 0,
            "selection": "training_grid_search" if search_grid else
                         "default_parameters" if name == "LinearRegression" else "archived_configuration",
        }
    return {"models": counts, "total_cv_fits": sum(value["cv_fits"] for value in counts.values()),
            "feature_set": feature_set, "cv": cv, "scoring": SCORING, "smoke_test": smoke_test}


def _training_arrays(frame, feature_set):
    if feature_set not in ("joint", "dataset_only"):
        raise ValueError("feature_set must be joint or dataset_only.")
    columns = list(FEATURE_NAMES) if feature_set == "joint" else list(DATASET_FEATURES)
    if frame.columns.duplicated().any() or not {*columns, "MCC", "row_id"} <= set(frame.columns):
        raise ValueError("Training frame requires unique descriptor, MCC and row_id columns.")
    if len(frame) < 4 or frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError("Four-fold selection requires at least four unique training rows.")
    X = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    y = frame.MCC.to_numpy(dtype=float, copy=True)
    if not np.isfinite(X).all() or not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("Reference selection requires finite training descriptors and MCC > 0 targets.")
    digest = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()
    return X, y, columns, frame.row_id.tolist(), digest


def _prepare_output(directory, config, resume):
    manifest = directory / "selection_config.json"
    if resume:
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("Cannot resume reference selection: training rows, grids, code or settings differ.")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Reference selection output is not empty; use strict resume or a fresh directory.")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, config)
    (directory / "checkpoints").mkdir(exist_ok=True)


def _validate_checkpoint(payload, *, name, grid, config_sha256, directory,
                         archived_params=None, archive_source=None):
    if payload.get("selection_config_sha256") != config_sha256:
        raise ValueError(f"Reference selection checkpoint provenance differs: {name}.")
    result = payload.get("result", {})
    if result.get("model") != name or result.get("n_candidates") != (len(ParameterGrid(grid)) if grid else 0):
        raise ValueError(f"Reference selection checkpoint model/grid differs: {name}.")
    candidates = ([_json_ready(archived_params)] if archived_params is not None else
                  [_json_ready(parameters) for parameters in ParameterGrid(grid)] if grid else [{}])
    if result.get("best_params") not in candidates:
        raise ValueError(f"Saved {name} parameters are not in its documented selection grid.")
    method = ("training_grid_search" if grid else "default_parameters"
              if name == "LinearRegression" else "archived_configuration")
    if result.get("method") != method:
        raise ValueError(f"Saved {name} selection method differs from its provenance.")
    if method == "archived_configuration" and any(
        result.get(key) != archive_source.get(key)
        for key in ("source", "source_sha256", "selection_features")
    ):
        raise ValueError(f"Saved {name} archived source provenance differs.")
    if grid:
        score, mse = result.get("best_score", np.nan), result.get("best_cv_mse", np.nan)
        if not np.isfinite([score, mse]).all() or score > 0 or not np.isclose(mse, -score):
            raise ValueError(f"Invalid saved reference selection score: {name}.")
        cv_path = directory / result.get("cv_results_file", "missing")
        if not cv_path.is_file() or file_sha256(cv_path) != result.get("cv_results_sha256"):
            raise ValueError(f"Missing or changed reference CV results: {name}.")
        table = pd.read_csv(cv_path)
        index = result.get("best_index", -1)
        if not {"params", "mean_test_score", "std_test_score"} <= set(table.columns):
            raise ValueError(f"Saved {name} CV evidence is missing required fields.")
        scores = table.mean_test_score.to_numpy(dtype=float)
        if (not np.isfinite(scores).all()
                or not isinstance(index, int) or not 0 <= index < len(table)
                or index != int(np.argmax(scores))
                or len(table) != len(ParameterGrid(grid))
                or json.loads(table.iloc[index]["params"]) != result["best_params"]
                or not np.isclose(table.iloc[index]["mean_test_score"], score, rtol=1e-12, atol=1e-12)
                or not np.isclose(table.iloc[index]["std_test_score"], result.get("best_score_std", np.nan), rtol=1e-12, atol=1e-12)
                or result.get("cv_fits") != 4 * len(ParameterGrid(grid))):
            raise ValueError(f"Saved {name} winner does not match its cross-validation evidence.")
    return result


def select_parameters(frame_train, feature_set, *, n_jobs, seed=SEED, cv=4,
                      models=None, output_dir=None, context=None, resume=False,
                      smoke_test=False):
    """Return feature-specific parameter maps and provenance, without test input.

    Each model's completed search has an independent checkpoint. An interrupted
    unfinished search is repeated; completed searches are reused only after
    validating their training-only provenance and CV-result file fingerprint.
    Smoke mode uses one candidate per model and is explicitly non-publication.
    GridSearchCV does not refit a winner; the caller performs the final fit once.
    """
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero.")
    plan = selection_plan(models, feature_set=feature_set, cv=cv, smoke_test=smoke_test)
    grids = parameter_grids(models, smoke_test=smoke_test)
    archived, archive_source = _read_archived_parameters(feature_set)
    # Default LR requires no historical tuning or new search.
    for name in grids:
        if name in archived or name == "LinearRegression":
            grids[name] = {}
    X, y, columns, row_ids, training_digest = _training_arrays(frame_train, feature_set)
    folds = list(KFold(n_splits=cv, shuffle=False).split(X))
    config = {
        "schema_version": 1, "protocol": "verified_archived_and_training_only_joint_selection",
        "feature_set": feature_set, "feature_order": columns,
        "training_row_ids": row_ids, "n_train": len(y), "training_arrays_sha256": training_digest,
        "context": _json_ready(context or {}), "models": list(grids), "grids": _json_ready(grids),
        "cv": cv, "cv_splitter": "KFold(shuffle=False), in supplied training-row order",
        "scoring": SCORING, "refit": False, "seed": seed, "n_jobs": n_jobs,
        "forest_workers_per_cv_fit": 1, "inner_threads_per_worker": 1,
        "smoke_test": smoke_test,
        "archived_configuration_source": archive_source,
        "archived_parameters": {name: archived[name] for name in grids if name in archived},
        "code_sha256": code_fingerprint(ROOT, SELECTION_CODE_FILES),
        "versions": package_versions(("numpy", "pandas", "scikit-learn", "joblib", "threadpoolctl")),
        "cv_fold_sizes": [{"n_train": len(train), "n_validation": len(valid)} for train, valid in folds],
        "scope": "No held-out row-wise labels used; winners subsequently frozen for conditional LODO.",
    }
    config_sha256 = content_fingerprint(config)
    directory = None if output_dir is None else Path(output_dir)
    if resume and directory is None:
        raise ValueError("Resume requires an explicit reference-selection output directory.")
    if directory is not None:
        _prepare_output(directory, config, resume)
    selected, evidence = {}, {}
    for name, grid in grids.items():
        checkpoint = None if directory is None else directory / "checkpoints" / f"{name}.json"
        if resume and checkpoint.is_file():
            result = _validate_checkpoint(json.loads(checkpoint.read_text(encoding="utf-8")),
                                          name=name, grid=grid, config_sha256=config_sha256,
                                          directory=directory,
                                          archived_params=archived.get(name),
                                          archive_source=archive_source)
            print(f"[{feature_set}] {name}: reused completed parameter selection", flush=True)
        elif name in archived and name != "LinearRegression":
            result = {"model": name, "best_params": archived[name],
                      "best_score": None, "best_cv_mse": None,
                      "n_candidates": 0, "cv_fits": 0, "method": "archived_configuration",
                      "selection": "verified archived winner for this descriptor feature set",
                      "selection_seconds": 0., "smoke_test": smoke_test, **archive_source}
        elif not grid:
            result = {"model": name, "best_params": {}, "best_score": None, "best_cv_mse": None,
                      "n_candidates": 0, "cv_fits": 0, "selection": "default parameters; no tuning",
                      "selection_seconds": 0., "smoke_test": smoke_test,
                      "method": "default_parameters"}
        else:
            print(f"[{feature_set}] {name}: {len(ParameterGrid(grid))} candidates x {cv} training CV folds", flush=True)
            # CV parallelizes candidates/folds; a forest fit uses one worker.
            model = make_model(name, {name: {}}, n_jobs=1)
            model.set_params(random_state=seed)
            search = GridSearchCV(model, grid, scoring=SCORING, cv=folds,
                                  n_jobs=n_jobs, refit=False, error_score="raise",
                                  return_train_score=False)
            started = perf_counter()
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                with parallel_config(backend="loky", inner_max_num_threads=1), threadpool_limits(limits=1):
                    search.fit(X, y)
            score = float(search.best_score_)
            if not np.isfinite(score) or score > 0:
                raise ValueError(f"Invalid {name} cross-validation score.")
            result = {
                "model": name, "best_params": _json_ready(search.best_params_),
                "best_score": score, "best_cv_mse": -score,
                "best_index": int(search.best_index_),
                "best_score_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                "n_candidates": len(ParameterGrid(grid)), "cv_fits": cv * len(ParameterGrid(grid)),
                "cv": cv, "selection": "training-only four-fold negative-MSE grid search",
                "method": "training_grid_search", "selection_features": feature_set,
                "selection_seconds": perf_counter() - started, "smoke_test": smoke_test,
                "warnings": sorted({f"{item.category.__name__}: {item.message}" for item in observed}),
            }
            if directory is not None:
                cv_path = directory / f"{name}_cv_results.csv"
                table = pd.DataFrame(search.cv_results_)
                table["params"] = table["params"].map(lambda parameters: json.dumps(_json_ready(parameters), sort_keys=True))
                temporary = cv_path.with_suffix(".csv.tmp")
                table.to_csv(temporary, index=False)
                temporary.replace(cv_path)
                result.update(cv_results_file=cv_path.name, cv_results_sha256=file_sha256(cv_path))
        if checkpoint is not None:
            atomic_json(checkpoint, {"selection_config_sha256": config_sha256, "result": result})
        selected[name], evidence[name] = result["best_params"], result
    payload = {"parameters": selected, "selection": evidence, "feature_order": columns,
               "feature_set": feature_set, "n_train": len(y), "training_row_ids": row_ids,
               "selection_config_sha256": config_sha256, "plan": plan,
               "conditional_lodo": True}
    if directory is not None:
        atomic_json(directory / "selected_parameters.json", selected)
        atomic_json(directory / "selection_summary.json", payload)
    return payload
