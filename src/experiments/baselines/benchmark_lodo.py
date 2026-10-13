"""LODO of five reference regressors with previously selected hyperparameters.

Run from the repository root::

    python -m src.experiments.baselines.benchmark_lodo

Or run this file directly from any working directory. No hyperparameter search,
scaling, imputation or symbolic discovery is performed. Each regressor is fitted
from scratch on the other datasets, with seed 42 and the 19 SR descriptors.
Only MCC > 0 rows are retained, exactly as in the supplied SR LODO experiments.
Their configuration files are checked against the input checksum, rows, feature
order and dataset groups before any training starts.

Random Forest uses its historical joint-descriptor winner. The other models
reuse the historical dataset-only configurations on the joint descriptors;
Linear Regression had no tuned hyperparameters. These choices are fixed before
LODO, so evaluation is conditional on prior selection, not nested model selection.

Examples::

    python -m src.experiments.baselines.benchmark_lodo --dry-run
    python -m src.experiments.baselines.benchmark_lodo --max-folds 1 --models LinearRegression DecisionTreeRegressor
    python -m src.experiments.baselines.benchmark_lodo --output-dir src/outputs/baselines/lodo/my_run
    python -m src.experiments.baselines.benchmark_lodo --output-dir src/outputs/baselines/lodo/my_run --resume

Outputs: run_config.json, folds.csv, selected_rows.csv, per-fit JSON checkpoints,
oof_predictions.csv, fold_metrics.csv, and summary.csv. Predictions and metrics
are saved both raw and clipped to each training fold's MCC range. The summary
includes pooled errors, equally weighted dataset errors and approximate 95%
dataset-bootstrap intervals conditional on the saved out-of-fold predictions.
Partial runs are labelled explicitly and never reported as full-corpus LODO.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

# Support both `python -m ...` and direct execution of this single script.
ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from src.equations.hroch import FEATURE_NAMES
from src.paths import META_DATASET_PATH, output_run_directory
from src.utils.artifact_integrity import atomic_json, file_sha256
from src.utils.pipeline_integrity import read_meta_dataset


SEED = 42
IDENTIFIERS = ("Seed", "Dataset", "Sample Size", "Model")
MODEL_CLASSES = {
    "LinearRegression": LinearRegression,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "MLPRegressor": MLPRegressor,
}
FIXED_PARAMETERS = {
    "LinearRegression": {},
    "DecisionTreeRegressor": {
        "max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 1,
        "min_samples_split": 5,
    },
    "RandomForestRegressor": {
        "bootstrap": False, "max_depth": 16, "max_features": 0.5,
        "min_samples_leaf": 2, "min_samples_split": 5, "n_estimators": 100,
    },
    "GradientBoostingRegressor": {
        "learning_rate": 0.03, "max_depth": 5, "n_estimators": 100,
        "subsample": 1.0,
    },
    "MLPRegressor": {
        "activation": "tanh", "alpha": 0.001, "hidden_layer_sizes": (200,),
    },
}
DATASET_ONLY_SOURCE = "src/outputs/baselines/dataset_only/archived/run.log"
JOINT_RF_SOURCE = "src/outputs/random_forest_shap/archived/run.log"
DEFAULT_SR_CONFIGS = tuple(
    ROOT / "src" / "outputs" / "recalibration" / name / "archived" / "run_config.json"
    for name in ("hroch_equation", "extended_equation")
)
SCOPE = (
    "LODO refitting with fixed, previously selected hyperparameters. Historical "
    "selection used observations from the dataset collection later held out. "
    "This evaluation is conditional on those choices, not nested validation of "
    "hyperparameter selection. Random Forest uses the 19-descriptor winner; the "
    "other historical configurations were selected with 12 dataset descriptors "
    "and are reused here with the same 19 descriptors as SR."
)


def json_ready(value):
    """Normalize estimator parameters; JSON uses lists in place of tuples."""
    return json.loads(json.dumps(value, allow_nan=False))


def make_model(name: str, parameters: dict, n_jobs: int):
    settings = copy.deepcopy(parameters[name])
    if name != "LinearRegression":
        settings["random_state"] = SEED
    if name == "RandomForestRegressor":
        settings["n_jobs"] = n_jobs
    if name == "MLPRegressor":
        if "hidden_layer_sizes" in settings:
            settings["hidden_layer_sizes"] = tuple(settings["hidden_layer_sizes"])
        settings.setdefault("max_iter", 1000)
    return MODEL_CLASSES[name](**settings)


def load_data(path: Path) -> tuple[pd.DataFrame, int]:
    frame = read_meta_dataset(path)
    original_rows = len(frame)
    frame.insert(0, "row_id", np.arange(original_rows))
    frame = frame.loc[frame["MCC"] > 0].copy().reset_index(drop=True)
    if frame.empty or frame["Dataset"].nunique() < 2:
        raise ValueError("LODO requires at least two datasets after the MCC > 0 filter.")
    return frame, original_rows


def verify_sr_inputs(paths: list[Path], frame: pd.DataFrame, original_rows: int,
                     data_sha256: str, data_path: Path) -> list[dict]:
    """Require the exact SR data and fold population, not only matching counts."""
    expected = {
        "original_rows": original_rows,
        "included_rows": len(frame), "mcc_filter": "MCC > 0",
        "all_datasets": sorted(frame["Dataset"].unique().tolist()),
        "feature_order": list(FEATURE_NAMES), "seed": SEED,
    }
    # Git's Windows checkout policy can change LF to CRLF without changing CSV
    # observations. Accept precisely these byte variants, not arbitrary new data.
    normalized = data_path.read_bytes().replace(b"\r\n", b"\n")
    compatible_hashes = {
        "exact_bytes": data_sha256,
        "line_endings_only_LF": hashlib.sha256(normalized).hexdigest(),
        "line_endings_only_CRLF": hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest(),
    }
    references = []
    for path in paths:
        recorded = json.loads(path.read_text(encoding="utf-8"))
        mismatched = [key for key, value in expected.items() if recorded.get(key) != value]
        match = next((name for name, digest in compatible_hashes.items()
                      if recorded.get("data_sha256") == digest), None)
        if match is None:
            mismatched.append("data_sha256")
        if mismatched:
            raise ValueError(f"Input differs from SR configuration {path}: {', '.join(mismatched)}")
        references.append({"path": str(path.resolve()), "sha256": file_sha256(path),
                           "recorded_data_sha256": recorded["data_sha256"], "input_match": match})
    return references


def score(y: np.ndarray, prediction: np.ndarray) -> dict:
    if len(y) == 0 or not np.isfinite(prediction).all():
        raise ValueError("Cannot score empty or non-finite predictions.")
    residual = prediction - y
    squared_error = float(np.dot(residual, residual))
    denominator = float(np.sum(np.square(y - y.mean())))
    ratios = np.divide(2 * np.abs(residual), np.abs(y) + np.abs(prediction),
                       out=np.zeros_like(residual), where=(np.abs(y) + np.abs(prediction)) > 0)
    return {
        "n": len(y), "invalid_predictions": 0,
        "r2": 1 - squared_error / denominator if denominator > 0 and len(y) > 1 else np.nan,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(squared_error / len(y))),
        "smape_pct": float(100 * ratios.mean()),
    }


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def prepare_output(directory: Path, config: dict, resume: bool) -> None:
    manifest = directory / "run_config.json"
    if resume:
        if not manifest.is_file():
            raise ValueError("--resume requires an existing run_config.json.")
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("Cannot resume: data, code, versions, parameters or settings changed.")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Output directory is not empty. Use --resume or a new --output-dir.")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, config)
    (directory / "folds").mkdir(exist_ok=True)


def validate_checkpoint(fold: dict, label: str, name: str, row_ids: list[int],
                        effective_parameters: dict, expected_fold: dict) -> None:
    if (fold.get("dataset") != label or fold.get("method") != name
            or fold.get("row_ids") != row_ids or fold.get("seed") != SEED
            or any(fold.get(key) != value for key, value in expected_fold.items())
            or fold.get("effective_parameters") != effective_parameters):
        raise ValueError(f"Checkpoint metadata mismatch for {label} / {name}.")
    predictions = np.asarray(fold.get("predictions_raw"), dtype=float)
    if predictions.shape != (len(row_ids),) or not np.isfinite(predictions).all():
        raise ValueError(f"Invalid checkpoint predictions for {label} / {name}.")


def fit_fold(name: str, parameters: dict, n_jobs: int, X_train: np.ndarray,
             y_train: np.ndarray, X_test: np.ndarray) -> dict:
    # No target or identifier from the reserved dataset is passed to this function.
    model = make_model(name, parameters, n_jobs)
    random.seed(SEED)
    np.random.seed(SEED)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        # Avoid nested BLAS thread pools; Random Forest controls its own workers.
        with threadpool_limits(limits=1, user_api="blas"):
            model.fit(X_train, y_train)
            prediction = np.asarray(model.predict(X_test), dtype=float)
    if prediction.shape != (len(X_test),) or not np.isfinite(prediction).all():
        raise ValueError(f"{name} produced non-finite or incorrectly shaped predictions.")
    return {
        "predictions_raw": prediction.tolist(), "fit_predict_seconds": perf_counter() - started,
        "effective_parameters": json_ready(model.get_params(deep=False)),
        "warnings": sorted({f"{item.category.__name__}: {item.message}" for item in observed}),
        "n_iter": int(model.n_iter_) if hasattr(model, "n_iter_") else None,
    }


def dataset_intervals(rows: pd.DataFrame, mode: str, repetitions: int) -> dict:
    """Percentile cluster bootstrap: resample whole datasets, never individual rows.

    Fitted predictions are held fixed. These approximate intervals describe
    between-dataset uncertainty conditional on fitted folds and fixed settings;
    they do not include re-selection or re-training variability.
    """
    names = ("r2", "mae", "rmse", "macro_fold_mae", "macro_fold_rmse")
    result = {f"{name}_ci95_{bound}": np.nan for name in names for bound in ("low", "high")}
    pieces = []
    for _, group in rows.groupby("Dataset", sort=True):
        y = group["y_true"].to_numpy()
        error = group[f"y_pred_{mode}"].to_numpy() - y
        pieces.append((len(y), y.sum(), np.dot(y, y), np.abs(error).sum(), np.dot(error, error)))
    if repetitions == 0 or len(pieces) < 2:
        return {**result, "bootstrap_repetitions": 0}
    values = np.asarray(pieces, dtype=float)
    indices = np.random.default_rng(SEED).integers(0, len(values), size=(repetitions, len(values)))
    counts, totals, squares, absolute_error, squared_error = values[indices].sum(axis=1).T
    denominator = squares - totals ** 2 / counts
    r2 = np.full(repetitions, np.nan)
    np.divide(squared_error, denominator, out=r2, where=denominator > 0)
    samples = {
        "r2": 1 - r2, "mae": absolute_error / counts,
        "rmse": np.sqrt(squared_error / counts),
        "macro_fold_mae": (values[:, 3] / values[:, 0])[indices].mean(axis=1),
        "macro_fold_rmse": np.sqrt(values[:, 4] / values[:, 0])[indices].mean(axis=1),
    }
    for name, sample in samples.items():
        finite = sample[np.isfinite(sample)]
        if len(finite):
            result[f"{name}_ci95_low"], result[f"{name}_ci95_high"] = np.quantile(finite, [0.025, 0.975]).tolist()
    return {**result, "bootstrap_repetitions": repetitions}


def write_results(directory: Path, frame: pd.DataFrame, completed: dict,
                  selected: list[str], models: list[str], bootstrap: int) -> pd.DataFrame:
    prediction_tables, metrics = [], []
    for (label, name), fold in sorted(completed.items()):
        test = frame["Dataset"] == label
        y = frame.loc[test, "MCC"].to_numpy(dtype=float)
        raw = np.asarray(fold["predictions_raw"], dtype=float)
        clipped = np.clip(raw, fold["train_target_min"], fold["train_target_max"])
        rows = frame.loc[test, ["row_id", *IDENTIFIERS]].copy()
        rows["method"], rows["y_true"] = name, y
        rows["y_pred_raw"], rows["y_pred_clipped"] = raw, clipped
        rows["clipped"], rows["invalid_prediction"] = raw != clipped, False
        prediction_tables.append(rows)
        for mode, prediction in (("raw", raw), ("clipped", clipped)):
            metrics.append({
                "dataset": label, "method": name, "prediction_mode": mode,
                "n_train": fold["n_train"], "seed": SEED,
                "train_target_min": fold["train_target_min"], "train_target_max": fold["train_target_max"],
                "fit_predict_seconds": fold["fit_predict_seconds"], "warning_count": len(fold["warnings"]),
                "clipped_predictions": int((raw != clipped).sum()) if mode == "clipped" else 0,
                **score(y, prediction),
            })
    oof = pd.concat(prediction_tables, ignore_index=True).sort_values(["method", "row_id"])
    per_fold = pd.DataFrame(metrics)
    summaries = []
    for name in models:
        rows = oof.loc[oof["method"] == name]
        if rows.empty:
            continue
        done = rows["Dataset"].nunique()
        complete = done == len(selected)
        full_corpus = complete and len(selected) == frame["Dataset"].nunique()
        for mode in ("raw", "clipped"):
            details = per_fold.loc[(per_fold["method"] == name) & (per_fold["prediction_mode"] == mode)]
            summaries.append({
                "method": name, "prediction_mode": mode,
                "completed_folds": done, "expected_folds": len(selected), "complete": complete,
                "full_corpus_complete": full_corpus,
                "evaluation_scope": "all_datasets" if len(selected) == frame["Dataset"].nunique() else "selected_datasets",
                **score(rows["y_true"].to_numpy(), rows[f"y_pred_{mode}"].to_numpy()),
                "macro_fold_mae": float(details["mae"].mean()),
                "macro_fold_rmse": float(details["rmse"].mean()),
                "worst_fold_mae": float(details["mae"].max()),
                "median_fold_r2": float(details["r2"].median()),
                "clipped_predictions": int(rows["clipped"].sum()) if mode == "clipped" else 0,
                **dataset_intervals(rows, mode, bootstrap if complete else 0),
            })
    summary = pd.DataFrame(summaries)
    for filename, table in (("oof_predictions", oof), ("fold_metrics", per_fold), ("summary", summary)):
        atomic_csv(directory / f"{filename}.csv", table)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=META_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=output_run_directory("baselines", "lodo"))
    parser.add_argument("--sr-config", nargs="+", type=Path, default=list(DEFAULT_SR_CONFIGS),
                        help="SR run_config.json files whose inputs must match exactly (default: both archived runs)")
    parser.add_argument("--hyperparameters", type=Path,
                        help="Optional JSON mapping all five class names to their fixed parameter dictionaries")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CLASSES), default=list(MODEL_CLASSES))
    parser.add_argument("--datasets", nargs="+", help="Evaluate only these datasets; every other dataset still trains each fold")
    parser.add_argument("--max-folds", type=int, help="Limit evaluation for a smoke test; does not shrink fold training sets")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Random Forest workers; -1 uses all available CPUs")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Whole-dataset bootstrap repetitions; 0 disables intervals")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan without fitting or writing outputs")
    args = parser.parse_args(argv)
    if args.n_jobs == 0 or args.bootstrap < 0 or (args.max_folds is not None and args.max_folds < 1):
        parser.error("n-jobs must be nonzero, bootstrap nonnegative and max-folds positive.")
    if len(args.models) != len(set(args.models)):
        parser.error("Models must not be repeated.")
    return args


def run(args) -> pd.DataFrame | None:
    data_hash = file_sha256(args.data)
    frame, original_rows = load_data(args.data)
    if file_sha256(args.data) != data_hash:
        raise ValueError("Input file changed while being read.")
    references = verify_sr_inputs(args.sr_config, frame, original_rows, data_hash, args.data)
    labels = sorted(frame["Dataset"].unique().tolist())
    unknown = sorted(set(args.datasets or ()) - set(labels))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    selected = [label for label in labels if args.datasets is None or label in args.datasets]
    if args.max_folds is not None:
        selected = selected[:args.max_folds]
    parameters = copy.deepcopy(FIXED_PARAMETERS)
    if args.hyperparameters:
        parameters = json.loads(args.hyperparameters.read_text(encoding="utf-8"))
        if not isinstance(parameters, dict) or set(parameters) != set(MODEL_CLASSES):
            raise ValueError("Hyperparameter JSON must map exactly the five regressor class names.")
        if any(not isinstance(value, dict) for value in parameters.values()):
            raise ValueError("Each hyperparameter entry must be a parameter dictionary.")
    # Construct before training so unknown parameter names fail even in a dry run.
    effective = {name: json_ready(make_model(name, parameters, args.n_jobs).get_params(deep=False))
                 for name in args.models}
    provenance = {}
    for name in args.models:
        if args.hyperparameters:
            provenance[name] = {
                "source": str(args.hyperparameters.resolve()), "sha256": file_sha256(args.hyperparameters),
                "selection_features": "user-supplied fixed configuration",
            }
        else:
            source = JOINT_RF_SOURCE if name == "RandomForestRegressor" else DATASET_ONLY_SOURCE
            provenance[name] = {
                "source": source, "sha256": file_sha256(ROOT / source),
                "selection_features": "19 joint descriptors" if name == "RandomForestRegressor" else "12 dataset descriptors",
                "selection": "default parameters, no tuning" if name == "LinearRegression" else "historical row-wise selection",
            }
    sources = (Path(__file__), ROOT / "src/equations/hroch.py", ROOT / "src/utils/pipeline_integrity.py",
               ROOT / "src/utils/artifact_integrity.py", ROOT / "src/paths.py")
    config = {
        "schema_version": 1, "protocol": "LODO_refitting_with_previously_selected_hyperparameters",
        "scope": (SCOPE if not args.hyperparameters else
                  "LODO refitting conditional on user-supplied fixed hyperparameters; their original selection is outside this evaluation."),
        "data_path": str(args.data.resolve()), "data_sha256": data_hash,
        "code_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest(),
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "pandas", "threadpoolctl", "joblib")},
        "original_rows": original_rows, "included_rows": len(frame), "mcc_filter": "MCC > 0",
        "all_datasets": labels, "selected_datasets": selected, "models": args.models,
        "feature_order": list(FEATURE_NAMES), "excluded_predictors": list(IDENTIFIERS),
        "preprocessing": "none: same raw descriptors as SR; no scaling or imputation",
        "sr_references": references, "fixed_parameters": json_ready(parameters),
        "effective_parameters": effective, "parameter_provenance": provenance,
        "seed": SEED, "n_jobs": args.n_jobs, "blas_threads_per_fit": 1,
        "clipping": "training_fold_target_range_for_reporting_only",
        "bootstrap_repetitions": args.bootstrap, "bootstrap_seed": SEED,
        "interval_method": "approximate percentile 95% whole-dataset bootstrap of fixed OOF predictions; conditional on fitted folds and fixed hyperparameters",
    }
    print(f"Rows: {len(frame)}/{original_rows}; descriptors: {len(FEATURE_NAMES)}; datasets: {len(labels)}", flush=True)
    print(f"Evaluation: {len(selected)} folds x {len(args.models)} models = {len(selected) * len(args.models)} fresh fits; seed={SEED}", flush=True)
    print("SR input checks passed; no hyperparameter search or feature preprocessing.", flush=True)
    if args.dry_run:
        print(json.dumps({"parameters": effective, "provenance": provenance}, indent=2), flush=True)
        return None
    prepare_output(args.output_dir, config, args.resume)
    atomic_csv(args.output_dir / "selected_rows.csv", frame.loc[:, ["row_id", *IDENTIFIERS, "MCC"]])
    X = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["MCC"].to_numpy(dtype=float)
    groups = frame["Dataset"].to_numpy()
    folds = []
    for index, label in enumerate(labels):
        if label in selected:
            test = groups == label
            folds.append({"fold_index": index, "dataset": label, "n_train": int((~test).sum()),
                          "n_test": int(test.sum()), "train_target_min": float(y[~test].min()),
                          "train_target_max": float(y[~test].max())})
    atomic_csv(args.output_dir / "folds.csv", pd.DataFrame(folds))
    completed = {}
    summary = None
    for position, fold_info in enumerate(folds, start=1):
        label, index = fold_info["dataset"], fold_info["fold_index"]
        test = groups == label
        train = ~test
        # Keep identifiers only in result metadata; X has the explicit SR whitelist.
        row_ids = frame.loc[test, "row_id"].tolist()
        for name in args.models:
            checkpoint = args.output_dir / "folds" / f"fold_{index:03d}_{name}.json"
            if args.resume and checkpoint.is_file():
                fitted = json.loads(checkpoint.read_text(encoding="utf-8"))
                validate_checkpoint(fitted, label, name, row_ids, effective[name], fold_info)
                print(f"[{position}/{len(folds)}] {label} / {name}: reused completed fit", flush=True)
            else:
                print(f"[{position}/{len(folds)}] {label} / {name}: train={train.sum()}, test={test.sum()}", flush=True)
                fitted = fit_fold(name, parameters, args.n_jobs, X[train], y[train], X[test])
                fitted.update({**fold_info, "method": name, "row_ids": row_ids, "seed": SEED})
                atomic_json(checkpoint, fitted)
                print(f"  saved in {fitted['fit_predict_seconds']:.2f}s; warnings={len(fitted['warnings'])}", flush=True)
            completed[label, name] = fitted
        # Checkpoints persist after each fit; rebuild aggregate tables after each dataset.
        summary = write_results(args.output_dir, frame, completed, selected, args.models, 0)
    summary = write_results(args.output_dir, frame, completed, selected, args.models, args.bootstrap)
    print(summary[["method", "prediction_mode", "completed_folds", "full_corpus_complete", "r2", "mae", "rmse"]].to_string(index=False), flush=True)
    print(f"Saved results to {args.output_dir.resolve()}", flush=True)
    return summary


def main(argv=None) -> None:
    try:
        run(parse_args(argv))
    except (ValueError, OSError, TypeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
