"""Fixed-form leave-one-dataset-out coefficient recalibration with pyBlindOpt.

The equation AND its initial coefficients were selected on the full corpus.
This evaluates coefficient recalibration conditional on that supplied equation,
not a nested validation of symbolic equation discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyBlindOpt
from joblib import effective_n_jobs, parallel_config
from pyBlindOpt.de import differential_evolution
from pyBlindOpt.egwo import enhanced_grey_wolf_optimization
from pyBlindOpt.init import oblesa
from scipy.stats import spearmanr

from src.equations.extended import COEFFICIENT_NAMES, DEFAULT_COEFFICIENTS, FEATURE_NAMES, predict
from src.paths import META_DATASET_PATH, output_run_directory
from src.utils.pipeline_integrity import read_meta_dataset


RANDOM_SEED = 42
BAD_LOSS = 1e100
METHODS = ("original_constants", "train_mean", "DE", "EGWO")
LIMITATION = (
    "Fixed-form coefficient recalibration only: the supplied equation structure "
    "and warm-start coefficients were selected using the full metadataset, "
    "including the datasets later held out here. This is not nested validation "
    "of the full symbolic-regression discovery procedure."
)


def load_data(path: Path, include_nonpositive: bool = False) -> tuple[pd.DataFrame, int]:
    """Keep dataset identities and original row ids; never fit preprocessing."""
    frame = read_meta_dataset(path)
    required = ["Seed", "Dataset", "Sample Size", "Model", *FEATURE_NAMES, "MCC"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[["Dataset", "Model"]].isna().any().any():
        raise ValueError("Dataset and Model identities must not be missing.")
    numeric = frame.loc[:, [*FEATURE_NAMES, "MCC"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Features and MCC must be finite; no rows are silently dropped or imputed.")
    original_rows = len(frame)
    frame.insert(0, "row_id", np.arange(original_rows))
    if not include_nonpositive:
        frame = frame.loc[frame["MCC"] > 0].copy()
    if frame.empty or frame["Dataset"].nunique() < 2:
        raise ValueError("LODO requires at least two datasets after the MCC filter.")
    return frame.reset_index(drop=True), original_rows


def dataset_folds(frame: pd.DataFrame, selected: list[str]):
    """All seeds, sample sizes and models of the held-out dataset leave training."""
    groups = frame["Dataset"].to_numpy()
    for label in selected:
        test = groups == label
        yield label, ~test, test


def coefficient_bounds(lower_factor: float, upper_factor: float) -> np.ndarray:
    if not (np.isfinite([lower_factor, upper_factor]).all()
            and 0 < lower_factor <= 1 <= upper_factor
            and lower_factor < upper_factor):
        raise ValueError("Bounds must satisfy 0 < lower factor <= 1 <= upper factor, with lower < upper.")
    return np.sort(np.column_stack((DEFAULT_COEFFICIENTS * lower_factor,
                                    DEFAULT_COEFFICIENTS * upper_factor)), axis=1)


def training_mse(coefficients: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Penalize invalid final outputs without discarding training observations."""
    prediction = predict(X, coefficients)
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        return BAD_LOSS
    with np.errstate(over="ignore", invalid="ignore"):
        loss = float(np.mean(np.square(prediction - y)))
    return loss if np.isfinite(loss) and loss < BAD_LOSS else BAD_LOSS


def initial_population(objective, bounds: np.ndarray, n_pop: int,
                       *, n_jobs: int = 1, dataset_label: str = "fold") -> np.ndarray:
    """Warm start + (N-1) native Sobol points -> OBLESA -> N individuals."""
    # Reset in every fold, so standalone, resumed and full runs use seed 42.
    rng = np.random.default_rng(RANDOM_SEED)
    sampler = pyBlindOpt.utils.SobolSampler(rng)
    seeds = np.vstack((DEFAULT_COEFFICIENTS, sampler.sample(n_pop - 1, bounds)))
    print(f"  [{dataset_label}] Starting OBLESA: population={n_pop}, workers={n_jobs}, seed={RANDOM_SEED}",
          flush=True)
    population = oblesa(
        objective=objective, bounds=bounds, population=seeds, n_pop=n_pop,
        n_jobs=n_jobs, seed=rng, selection="best",
    )
    population = np.asarray(population, dtype=float).copy()
    if population.shape != seeds.shape or not np.isfinite(population).all():
        raise RuntimeError("OBLESA did not return the requested finite population.")
    if ((population < bounds[:, 0]) | (population > bounds[:, 1])).any():
        raise RuntimeError("OBLESA returned a coefficient outside its bounds.")
    # OBLESA may discard the warm start. Retain it explicitly for both methods.
    if not np.any(np.all(population == DEFAULT_COEFFICIENTS, axis=1)):
        worst = int(np.argmax(pyBlindOpt.utils.compute_objective(population, objective, n_jobs)))
        population[worst] = DEFAULT_COEFFICIENTS
    return population


def calibrate_fold(X_train: np.ndarray, y_train: np.ndarray, bounds: np.ndarray,
                   *, n_pop: int, n_iter: int, n_jobs: int,
                   dataset_label: str = "fold", methods=("DE", "EGWO")) -> dict:
    """No held-out features or targets enter initialization, selection or loss."""
    methods = tuple(methods)
    if not methods or len(methods) != len(set(methods)) or set(methods) - {"DE", "EGWO"}:
        raise ValueError("Select distinct optimizer names from DE and EGWO.")
    X_train, y_train = np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float)
    if (X_train.ndim != 2 or X_train.shape[1] != len(FEATURE_NAMES)
            or y_train.ndim != 1 or len(y_train) != len(X_train) or not len(y_train)
            or not np.isfinite(X_train).all() or not np.isfinite(y_train).all()):
        raise ValueError("Training requires finite nonempty X/y with the documented feature order.")
    # Each worker evaluates a complete candidate deterministically. The seeded
    # sampler and optimizer stay in the main process; workers draw no randomness.
    objective = partial(training_mse, X=X_train, y=y_train)
    workers = min(n_pop, effective_n_jobs(n_jobs))
    original_loss = objective(DEFAULT_COEFFICIENTS)
    if original_loss >= BAD_LOSS:
        raise ValueError("The supplied coefficients do not yield finite training predictions.")
    # Keep one process pool across the fold and avoid each worker spawning its
    # own BLAS/OpenMP thread pool. Joblib preserves candidate order in results.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        started = perf_counter()
        population = initial_population(objective, bounds, n_pop, n_jobs=workers,
                                        dataset_label=dataset_label)
        initialization_seconds = perf_counter() - started
        initial_best = float(np.min(pyBlindOpt.utils.compute_objective(population, objective, workers)))
        result = {"initialization_seconds": initialization_seconds, "initial_best_mse": initial_best,
                  "original_train_mse": original_loss, "seed": RANDOM_SEED,
                  "workers": workers, "methods": {}}
        optimizers = {"DE": differential_evolution, "EGWO": enhanced_grey_wolf_optimization}
        for name in methods:
            optimizer = optimizers[name]
            print(f"  [{dataset_label}] Starting {name} optimization: population={n_pop}, "
                  f"iterations={n_iter}, workers={workers}, seed={RANDOM_SEED}", flush=True)
            started = perf_counter()
            kwargs = {"variant": "best/1/bin", "F": 0.5, "cr": 0.7} if name == "DE" else {}
            best, _, history = optimizer(
                objective=objective, bounds=bounds, population=population.copy(),
                n_pop=n_pop, n_iter=n_iter, n_jobs=workers, seed=RANDOM_SEED,
                debug=True, verbose=False, cached=False, **kwargs,
            )
            best = np.asarray(best, dtype=float)
            loss = objective(best)
            # Compare using training loss only, never the held-out score.
            used_warm_start = loss > original_loss
            if used_warm_start:
                best, loss = DEFAULT_COEFFICIENTS.copy(), original_loss
            if len(history[0]) != n_iter:
                raise RuntimeError(f"{name} did not complete all {n_iter} iterations.")
            result["methods"][name] = {
                "coefficients": best.tolist(), "train_mse": loss,
                "optimization_seconds": perf_counter() - started,
                "used_warm_start": used_warm_start,
                "best_mse_history": np.asarray(history[0], dtype=float).tolist(),
            }
    return result


def validate_fold_checkpoint(fold, *, label, row_ids, n_train, bounds,
                             methods, n_iter, run_config_sha256):
    """Reject copied, truncated, or incompatible recovery checkpoints."""
    expected = {"dataset": label, "row_ids": row_ids, "n_train": n_train,
                "seed": RANDOM_SEED, "run_config_sha256": run_config_sha256}
    if any(fold.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint fold population or run provenance does not match.")
    if set(fold.get("methods", {})) != set(methods):
        raise ValueError("Checkpoint optimizer names do not match.")
    for key in ("original_train_mse", "initial_best_mse", "initialization_seconds"):
        value = fold.get(key, np.nan)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid checkpoint {key}.")
    for name in methods:
        detail = fold["methods"][name]
        coefficients = np.asarray(detail.get("coefficients", []), dtype=float)
        curve = np.asarray(detail.get("best_mse_history", []), dtype=float)
        if (coefficients.shape != (len(bounds),) or not np.isfinite(coefficients).all()
                or np.any(coefficients < bounds[:, 0]) or np.any(coefficients > bounds[:, 1])
                or curve.shape != (n_iter,) or not np.isfinite(curve).all()
                or not np.isfinite(detail.get("train_mse", np.nan))
                or not 0 <= detail.get("train_mse", BAD_LOSS) < BAD_LOSS):
            raise ValueError(f"Invalid {name} checkpoint coefficients, history, or loss.")


def score(y: np.ndarray, prediction: np.ndarray) -> dict:
    """Any invalid prediction invalidates the full score, rather than shrinking n."""
    invalid = int(np.count_nonzero(~np.isfinite(prediction)))
    values = {"n": len(y), "invalid_predictions": invalid}
    keys = ("r2", "mae", "rmse", "smape_pct", "spearman")
    if invalid:
        return {**values, **dict.fromkeys(keys, float("nan"))}
    residual = prediction - y
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean(np.square(residual)))
        total = float(np.sum(np.square(y - y.mean())))
        r2 = 1 - float(np.sum(np.square(residual))) / total if total > 0 and len(y) > 1 else float("nan")
        denominator = np.abs(y) + np.abs(prediction)
        ratios = np.divide(2 * np.abs(residual), denominator,
                           out=np.zeros_like(residual), where=denominator > 0)
    rho = float(spearmanr(y, prediction).statistic) if (
        len(y) > 1 and np.ptp(y) > 0 and np.ptp(prediction) > 0) else float("nan")
    return {**values, "r2": r2, "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(mse)), "smape_pct": float(100 * ratios.mean()), "spearman": rho}


def clip_to_training(prediction: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    # Clipping must not conceal an infinite or NaN equation output.
    return np.where(np.isfinite(prediction),
                    np.clip(prediction, float(y_train.min()), float(y_train.max())), np.nan)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def prepare_output(directory: Path, configuration: dict, resume: bool) -> None:
    manifest = directory / "run_config.json"
    if resume:
        if not manifest.is_file():
            raise ValueError("--resume requires an existing run_config.json.")
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != configuration:
            raise ValueError("Cannot resume: data, code, versions or experiment settings differ from run_config.json.")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Output directory is not empty. Use --resume or choose another --output-dir.")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, configuration)
    (directory / "folds").mkdir(exist_ok=True)


def write_results(directory: Path, frame: pd.DataFrame, completed: dict,
                  expected_folds: int) -> pd.DataFrame:
    """Rebuild portable results from fold checkpoints, also during a long run."""
    predictions, metrics, coefficients, convergence = [], [], [], []
    if not completed:
        raise ValueError("No completed folds to summarize.")
    first_methods = tuple(next(iter(completed.values()))["methods"])
    if any(tuple(fold["methods"]) != first_methods for fold in completed.values()):
        raise ValueError("Completed folds have inconsistent optimizers.")
    report_methods = ("original_constants", "train_mean", *first_methods)
    X = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["MCC"].to_numpy(dtype=float)
    for label, train, test in dataset_folds(frame, list(completed)):
        fold = completed[label]
        y_train, y_test = y[train], y[test]
        for method in report_methods:
            values = DEFAULT_COEFFICIENTS if method == "original_constants" else None
            if method in ("DE", "EGWO"):
                values = np.asarray(fold["methods"][method]["coefficients"])
            raw = (np.full(len(y_test), y_train.mean()) if method == "train_mean"
                   else predict(X[test], values))
            clipped = clip_to_training(raw, y_train)
            rows = frame.loc[test, ["row_id", "Seed", "Dataset", "Sample Size", "Model"]].copy()
            rows["method"], rows["y_true"] = method, y_test
            rows["y_pred_raw"], rows["y_pred_clipped"] = raw, clipped
            rows["clipped"] = np.isfinite(raw) & (raw != clipped)
            rows["invalid_prediction"] = ~np.isfinite(raw)
            predictions.append(rows)
            detail = fold["methods"].get(method, {})
            for mode, prediction in (("raw", raw), ("clipped", clipped)):
                metrics.append({
                    "dataset": label, "method": method, "prediction_mode": mode,
                    "n_train": len(y_train), "train_target_min": float(y_train.min()),
                    "train_target_max": float(y_train.max()), "seed": fold["seed"],
                    "train_mse": detail.get("train_mse", fold["original_train_mse"]
                                           if method == "original_constants" else float(np.var(y_train))),
                    "initial_best_mse": fold["initial_best_mse"],
                    "initialization_seconds": fold["initialization_seconds"],
                    "optimization_seconds": detail.get("optimization_seconds", 0),
                    "clipped_predictions": int(rows["clipped"].sum()) if mode == "clipped" else 0,
                    **score(y_test, prediction),
                })
            if values is not None:
                coefficients.append({"dataset": label, "method": method, "seed": fold["seed"],
                                     **dict(zip(COEFFICIENT_NAMES, values.tolist()))})
            if method in ("DE", "EGWO"):
                convergence.append({"dataset": label, "method": method, "iteration": 0,
                                    "best_train_mse": fold["initial_best_mse"]})
                convergence.extend({"dataset": label, "method": method, "iteration": i,
                                    "best_train_mse": value}
                                   for i, value in enumerate(detail["best_mse_history"], start=1))
    oof = pd.concat(predictions, ignore_index=True)
    per_fold = pd.DataFrame(metrics)
    summary = []
    for method in report_methods:
        rows = oof[oof["method"] == method]
        for mode in ("raw", "clipped"):
            fold_scores = per_fold[(per_fold["method"] == method)
                                    & (per_fold["prediction_mode"] == mode)]
            mae_values = fold_scores["mae"].to_numpy()
            summary.append({
                "method": method, "prediction_mode": mode, "completed_folds": len(completed),
                "expected_folds": expected_folds,
                "evaluation_scope": "all_datasets" if expected_folds == frame["Dataset"].nunique() else "selected_datasets",
                "complete": len(completed) == expected_folds,
                **score(rows["y_true"].to_numpy(), rows[f"y_pred_{mode}"].to_numpy()),
                "macro_fold_mae": float(mae_values.mean()),
                "worst_fold_mae": float(mae_values.max()),
                "median_fold_r2": float(np.median(fold_scores["r2"].to_numpy())),
                "clipped_predictions": int(rows["clipped"].sum()) if mode == "clipped" else 0,
            })
    summary = pd.DataFrame(summary)
    for name, table in (("oof_predictions", oof), ("fold_metrics", per_fold),
                        ("coefficients", pd.DataFrame(coefficients)),
                        ("convergence", pd.DataFrame(convergence)), ("summary", summary)):
        atomic_csv(directory / f"{name}.csv", table)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=META_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path,
                        default=output_run_directory("recalibration", "extended_equation"))
    parser.add_argument("--methods", nargs="+", choices=("DE", "EGWO"), default=["DE", "EGWO"],
                        help="Coefficient optimizers to run")
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="CPU workers for OBLESA, DE and EGWO: -1 uses all available CPUs; 1 runs serially")
    parser.add_argument("--lower-factor", type=float, default=0.1)
    parser.add_argument("--upper-factor", type=float, default=10.0)
    parser.add_argument("--include-nonpositive", action="store_true", help="Include MCC <= 0, unlike the original GP scripts")
    parser.add_argument("--datasets", nargs="+", help="Only evaluate these datasets; all other datasets still train each fold")
    parser.add_argument("--max-folds", type=int, help="Limit evaluated datasets for a smoke test; training still uses all other datasets")
    parser.add_argument("--resume", action="store_true", help="Reuse completed folds after checking data, code and settings")
    args = parser.parse_args(argv)
    if len(args.methods) != len(set(args.methods)):
        parser.error("Optimizer methods must not be repeated.")
    if args.population_size < 4 or args.iterations < 1:
        parser.error("Population size must be >= 4 and iterations must be >= 1.")
    if args.n_jobs == 0 or (args.max_folds is not None and args.max_folds < 1):
        parser.error("n-jobs must be nonzero and max-folds positive.")
    return args


def run(args) -> pd.DataFrame:
    bounds = coefficient_bounds(args.lower_factor, args.upper_factor)
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    frame, original_rows = load_data(args.data, args.include_nonpositive)
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != data_sha256:
        raise ValueError("Input data changed while being read.")
    labels = sorted(frame["Dataset"].unique().tolist())
    selected = labels if args.datasets is None else [label for label in labels if label in args.datasets]
    unknown = sorted(set(args.datasets or ()) - set(labels))
    if unknown:
        raise ValueError(f"Unknown or filtered-out datasets: {unknown}")
    if args.max_folds is not None:
        selected = selected[:args.max_folds]
    source_root = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), source_root / "equations" / "extended.py",
                    source_root / "utils" / "pipeline_integrity.py",
                    source_root / "utils" / "artifact_integrity.py", source_root / "paths.py"]
    configuration = {
        "schema_version": 2, "protocol": "LODO_fixed_form_coefficient_recalibration",
        "limitation": LIMITATION, "data_sha256": data_sha256,
        "code_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest(),
        "versions": {name: version(name) for name in ("pyBlindOpt", "EmptySpaceSearch", "torann", "joblib", "numpy", "pandas", "scipy")},
        "original_rows": original_rows, "included_rows": len(frame),
        "mcc_filter": "all" if args.include_nonpositive else "MCC > 0",
        "all_datasets": labels, "selected_datasets": selected,
        "feature_order": list(FEATURE_NAMES), "coefficient_order": list(COEFFICIENT_NAMES),
        "initial_coefficients": DEFAULT_COEFFICIENTS.tolist(), "bounds": bounds.tolist(),
        "objective": "raw_training_MSE", "clipping": "training_fold_target_range_for_reporting_only",
        "population_size": args.population_size, "iterations": args.iterations,
        "methods": list(args.methods),
        "seed": RANDOM_SEED, "n_jobs": args.n_jobs,
        "effective_workers": min(args.population_size, effective_n_jobs(args.n_jobs)),
        "parallelism": "candidate evaluation in loky processes; ordered results; one inner thread per worker",
        "rng_policy": "reset seed in every fold; SobolSampler and OBLESA share rng; DE and EGWO each start with the same seed",
        "initialization": "current_coefficients + (N-1) pyBlindOpt.utils.SobolSampler(rng) -> OBLESA defaults; retain warm start; shared by DE and EGWO",
        "DE": {"variant": "best/1/bin", "F": 0.5, "cr": 0.7}, "EGWO": "pyBlindOpt defaults",
    }
    prepare_output(args.output_dir, configuration, args.resume)
    run_config_sha256 = hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    X = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["MCC"].to_numpy(dtype=float)
    completed = {}
    label_indices = {label: i for i, label in enumerate(labels)}
    print(LIMITATION, flush=True)
    print(f"Rows: {len(frame)}/{original_rows}; datasets: {len(labels)}; evaluated folds: {len(selected)}", flush=True)
    print(f"CPU workers: {configuration['effective_workers']}; seed: {RANDOM_SEED}", flush=True)
    summary = None
    for position, (label, train, test) in enumerate(dataset_folds(frame, selected), start=1):
        index = label_indices[label]
        checkpoint = args.output_dir / "folds" / f"fold_{index:03d}.json"
        if args.resume and checkpoint.is_file():
            fold = json.loads(checkpoint.read_text(encoding="utf-8"))
            validate_fold_checkpoint(
                fold, label=label, row_ids=frame.loc[test, "row_id"].tolist(),
                n_train=int(train.sum()), bounds=bounds, methods=args.methods,
                n_iter=args.iterations, run_config_sha256=run_config_sha256,
            )
            print(f"[{position}/{len(selected)}] {label}: reused completed fold", flush=True)
        else:
            print(f"[{position}/{len(selected)}] {label}: train={train.sum()}, test={test.sum()}, seed={RANDOM_SEED}", flush=True)
            fold = calibrate_fold(X[train], y[train], bounds, n_pop=args.population_size,
                                  n_iter=args.iterations, n_jobs=args.n_jobs, dataset_label=label, methods=args.methods)
            fold.update({"dataset": label, "row_ids": frame.loc[test, "row_id"].tolist(),
                         "n_train": int(train.sum()), "run_config_sha256": run_config_sha256})
            atomic_json(checkpoint, fold)
            print("  " + "; ".join(f"{method} train MSE={detail['train_mse']:.6g}"
                                     for method, detail in fold["methods"].items()), flush=True)
        completed[label] = fold
        summary = write_results(args.output_dir, frame, completed, len(selected))
    print(summary[["method", "prediction_mode", "r2", "mae", "rmse", "invalid_predictions"]].to_string(index=False), flush=True)
    print(f"Saved results to {args.output_dir.resolve()}", flush=True)
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
