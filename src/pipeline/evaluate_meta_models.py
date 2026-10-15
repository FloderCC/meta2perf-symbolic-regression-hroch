"""Evaluate symbolic and reference meta-models on a prepared meta-dataset.

Run from the repository root with ``python -m src.pipeline.evaluate_meta_models``.
Only observations with MCC > 0 enter fitting and evaluation, preserving the
paper's analysis population. Reference hyperparameters are selected separately
for missing joint configurations using the row-wise training partition only.
Original preparation tables and historical results are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.equations import extended, hroch
from src.experiments.baselines import benchmark_lodo as baselines
from src.experiments.recalibration import (
    recalibrate_extended_equation_lodo as deap_lodo,
    recalibrate_hroch_equation_lodo as hroch_lodo,
)
from src.experiments.baselines.select_configurations import (
    historical_configuration_sources, select_parameters, selection_plan,
)
from src.paths import META_DATASET_PATH, output_run_directory
from src.utils.artifact_integrity import atomic_json, content_fingerprint, file_sha256
from src.utils.meta_regression import adjusted_r2, stratified_regression_split
from src.utils.pipeline_integrity import DATASET_FEATURES, read_meta_dataset, read_pipeline_table

SEED = 42
DESCRIPTOR_DEFINITION_VERSION = "corrected-complexity-regularization-v1"
DEFAULT_OUTPUT = output_run_directory("model_evaluation")
EQUATIONS = {"HROCH": hroch, "DEAP": extended}
IDENTIFIERS = ["Seed", "Dataset", "Sample Size", "Model"]
CODE_FILES = (
    "src/pipeline/evaluate_meta_models.py", "src/pipeline/build_meta_dataset.py",
    "src/utils/artifact_integrity.py",
    "src/utils/meta_regression.py", "src/utils/pipeline_integrity.py", "src/paths.py",
    "src/equations/hroch.py", "src/equations/extended.py",
    "src/experiments/recalibration/recalibrate_hroch_equation_lodo.py",
    "src/experiments/recalibration/recalibrate_extended_equation_lodo.py",
    "src/experiments/baselines/benchmark_lodo.py", "src/analysis/model_diagnostics.py",
    "src/experiments/baselines/select_configurations.py", "src/experiments/baselines/benchmark.py",
)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    atomic_json(path, clean_json(value))


def select_scope(frame, scope):
    if scope != "positive":
        raise ValueError("This evaluation retains only MCC > 0 observations (scope=positive).")
    selected = frame.copy()
    selected.insert(0, "row_id", np.arange(len(frame)))
    selected = selected.loc[selected.MCC > 0].copy()
    return selected


def make_split(selected):
    indexed = selected.set_index("row_id", drop=False)
    train, test, _, _ = stratified_regression_split(
        indexed.loc[:, list(hroch.FEATURE_NAMES)], indexed.MCC,
    )
    if set(train.index) & set(test.index) or set(train.index) | set(test.index) != set(indexed.index):
        raise ValueError("Invalid row-wise split coverage.")
    split = selected.loc[:, ["row_id", *IDENTIFIERS, "MCC"]].copy()
    split["partition"] = np.where(split.row_id.isin(train.index), "train", "test")
    # SHAP must explain the exact RF fit, including its training row order.
    split = split.set_index("row_id", drop=False).loc[list(train.index) + list(test.index)].reset_index(drop=True)
    return indexed.loc[train.index], indexed.loc[test.index], split


def metrics(y, predictions, count):
    predictions = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float)
    if predictions.shape != y.shape or not np.isfinite(predictions).all():
        raise ValueError("Invalid predictions cannot be hidden by clipping or row filtering.")
    values = baselines.score(y, predictions)
    if not np.isfinite([values["r2"], values["mae"], values["rmse"]]).all():
        raise ValueError("Row-wise metrics are undefined or overflowed; the run cannot be completed.")
    return {**values, "adjusted_r2": adjusted_r2(values["r2"], len(y), count),
            "smape": values["smape_pct"] / 100,
            "outside_mcc_range": int(((predictions < -1) | (predictions > 1)).sum())}


def checked_checkpoint(path, expected):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("context") != expected:
        raise ValueError(f"Checkpoint context mismatch: {path}")
    return payload["result"]


def supplied_equation_record(name, equation):
    source = Path(equation.__file__).resolve()
    return {"equation": name, "coefficient_order": list(equation.COEFFICIENT_NAMES),
            "coefficients": equation.DEFAULT_COEFFICIENTS.tolist(),
            "source_module": source.relative_to(ROOT).as_posix(), "source_sha256": file_sha256(source),
            "procedure": "evaluate_supplied_final_equation"}


def fit_reference(name, parameters, jobs, X_train, y_train, X_test):
    model = baselines.make_model(name, parameters, jobs)
    np.random.seed(SEED)
    random.seed(SEED)
    start = perf_counter()
    with warnings.catch_warnings(record=True) as observed, threadpool_limits(limits=1, user_api="blas"):
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
        train_prediction = np.asarray(model.predict(X_train), dtype=float)
        test_prediction = np.asarray(model.predict(X_test), dtype=float)
    if not np.isfinite(train_prediction).all() or not np.isfinite(test_prediction).all():
        raise ValueError(f"Non-finite {name} predictions.")
    return {"train_prediction": train_prediction.tolist(), "test_prediction": test_prediction.tolist(),
            "fit_predict_seconds": perf_counter() - start,
            "effective_parameters": baselines.json_ready(model.get_params(deep=False)),
            "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in observed})}


def reference_model_names(args, feature_set):
    """The joint RF must also be tuned when SHAP is requested in a partial run."""
    models = list(args.models)
    if feature_set == "joint" and not args.skip_shap and "RandomForestRegressor" not in models:
        models.append("RandomForestRegressor")
    return models


def plot_symbolic(targets, predictions, path, scope):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
    lower = min(0, float(np.min(targets)),
                *(float(np.min(p)) for p in predictions.values())) - .03
    upper = max(1, *(float(np.max(p)) for p in predictions.values())) + .03
    for ax, name in zip(axes, EQUATIONS):
        p = predictions[name]
        ax.scatter(targets, p, s=8, alpha=.23, edgecolors="none", rasterized=True)
        ax.plot([lower, upper], [lower, upper], "--", color="#B44135", lw=1.2)
        ax.set(xlabel="Observed MCC", title=name,
               xlim=(lower, upper), ylim=(lower, upper))
        ax.set_aspect("equal", adjustable="box")
        score = metrics(targets, p, len(hroch.FEATURE_NAMES))
        ax.text(.03, .97, f"$R^2$ = {score['r2']:.3f}\nMAE = {score['mae']:.3f}",
                transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": .9})
    axes[0].set_ylabel("Predicted MCC (raw)")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def run_rowwise(selected, directory, config, args, scope):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "checkpoints").mkdir(exist_ok=True)
    train, test, split = make_split(selected)
    baselines.atomic_csv(directory / "split.csv", split)
    Xtrain = train.loc[:, list(hroch.FEATURE_NAMES)].to_numpy(dtype=float)
    Xtest = test.loc[:, list(hroch.FEATURE_NAMES)].to_numpy(dtype=float)
    ytrain, ytest = train.MCC.to_numpy(), test.MCC.to_numpy()
    fitting_context = {"run_sha256": content_fingerprint(config), "scope": scope,
                       "train_row_ids": train.index.tolist(), "test_row_ids": test.index.tolist()}
    fitted, table, predictions = {}, [], []

    def append(name, optimizer, feature_set, values, count):
        for partition, rows, y, raw in (("train", train, ytrain, values[0]), ("test", test, ytest, values[1])):
            raw = np.asarray(raw, dtype=float)
            metrics(y, raw, count)  # Check raw before applying the reporting constraint.
            clipped = np.clip(raw, float(ytrain.min()), float(ytrain.max()))
            for mode, p in (("raw", raw), ("clipped", clipped)):
                table.append({"scope": scope, "model": name, "optimizer": optimizer,
                              "feature_set": feature_set, "partition": partition,
                              "prediction_mode": mode, **metrics(y, p, count)})
            detail = rows.loc[:, ["row_id", *IDENTIFIERS]].copy()
            detail["predictor"], detail["optimizer"], detail["feature_set"] = name, optimizer, feature_set
            detail["partition"], detail["y_true"] = partition, y
            detail["y_pred_raw"], detail["y_pred_clipped"] = raw, clipped
            predictions.append(detail)

    for name, equation in EQUATIONS.items():
        print(f"[{scope}] {name}: evaluating supplied final equation", flush=True)
        fitted[name] = supplied_equation_record(name, equation)
        append(name, "supplied_equation", "joint", (equation.predict(Xtrain), equation.predict(Xtest)), len(hroch.FEATURE_NAMES))
    save_json(directory / "coefficients.json", fitted)

    reference_selection = {}
    for feature_set, columns in (("joint", list(hroch.FEATURE_NAMES)), ("dataset_only", DATASET_FEATURES)):
        print(f"[{scope}] {feature_set}: reference configuration verification and training-only selection", flush=True)
        selection_directory = directory / "checkpoints" / "reference_selection" / feature_set
        selection = select_parameters(
            train, feature_set, n_jobs=args.n_jobs, seed=SEED, cv=4,
            models=reference_model_names(args, feature_set),
            output_dir=selection_directory,
            context={**fitting_context, "feature_set": feature_set},
            resume=bool(args.resume and (selection_directory / "selection_config.json").is_file()),
            smoke_test=args.smoke_test,
        )
        reference_selection[feature_set] = selection
        parameters = selection["parameters"]
        save_json(directory / f"reference_parameters_{feature_set}.json", parameters)
        for name in args.models:
            checkpoint = directory / "checkpoints" / f"{feature_set}_{name}.json"
            expected = {**fitting_context, "predictor": name, "feature_set": feature_set,
                        "parameters": baselines.json_ready(parameters[name])}
            if args.resume and checkpoint.is_file():
                result = checked_checkpoint(checkpoint, expected)
                print(f"[{scope}] {feature_set}/{name}: reused completed fit", flush=True)
            else:
                print(f"[{scope}] {feature_set}/{name}: selected configuration fit", flush=True)
                result = fit_reference(name, parameters, args.n_jobs,
                                       train.loc[:, columns].to_numpy(), ytrain, test.loc[:, columns].to_numpy())
                save_json(checkpoint, {"context": expected, "result": result})
            append(name, "training_cv", feature_set,
                   (result["train_prediction"], result["test_prediction"]), len(columns))
    append("TrainingMean", "training_only", "joint",
           (np.full(len(train), ytrain.mean()), np.full(len(test), ytrain.mean())), 0)
    values, prediction_table = pd.DataFrame(table), pd.concat(predictions, ignore_index=True)
    baselines.atomic_csv(directory / "metrics.csv", values)
    baselines.atomic_csv(directory / "predictions.csv", prediction_table)
    save_json(directory / "reference_selection.json", reference_selection)
    predicted = {name: equation.predict(Xtest) for name, equation in EQUATIONS.items()}
    plot_symbolic(ytest, predicted, directory / "symbolic_comparison_final", scope)
    save_json(directory / "rowwise_config.json", {
        "context": fitting_context, "n_train": len(train), "n_test": len(test), "seed": SEED,
        "split": "80/20 target-histogram stratification of the MCC > 0 population",
        "symbolic_evaluation": "supplied final equation defaults; no row-wise coefficient fitting",
        "deap_form": "grouped 19-coefficient form used consistently for row-wise and LODO",
        "baseline_selection": (
            "Verified historical dataset-only configurations and joint RF configuration; default LinearRegression; "
            "missing joint DT/GB/MLP configurations selected by 4-fold negative-MSE training-only GridSearchCV"
        ),
        "baseline_parameters_joint": reference_selection["joint"]["parameters"],
        "baseline_parameters_dataset_only": reference_selection["dataset_only"]["parameters"],
        "clipping": "training-partition target range; reporting only; raw outputs retained",
    })
    return fitted, values, reference_selection


def run_lodo(data_path, directory, args, scope, joint_parameters_path):
    configs = []
    summaries = []
    for name, module in (("HROCH", hroch_lodo), ("DEAP", deap_lodo)):
        output = directory / name.lower()
        argv = ["--data", str(data_path), "--output-dir", str(output), "--population-size", str(args.population_size),
                "--iterations", str(args.iterations), "--n-jobs", str(args.n_jobs),
                "--lower-factor", str(args.lower_factor), "--upper-factor", str(args.upper_factor),
                "--methods", *args.methods]
        if args.max_folds:
            argv += ["--max-folds", str(args.max_folds)]
        if args.resume and (output / "run_config.json").is_file():
            argv.append("--resume")
        summary = module.run(module.parse_args(argv))
        oof = pd.read_csv(output / "oof_predictions.csv")
        enriched = []
        for _, row in summary.iterrows():
            piece = oof.loc[oof.method == row["method"]]
            intervals = baselines.dataset_intervals(piece, row["prediction_mode"], args.bootstrap if bool(row["complete"]) else 0)
            enriched.append({"model": name, "scope": scope, **row.to_dict(), **intervals,
                             "full_corpus_complete": bool(row["complete"]) and row["evaluation_scope"] == "all_datasets"})
        interval_table = pd.DataFrame(enriched)
        baselines.atomic_csv(output / "summary_with_intervals.csv", interval_table)
        summaries.append(interval_table)
        configs.append(output / "run_config.json")

    output = directory / "references"
    argv = ["--data", str(data_path), "--output-dir", str(output), "--n-jobs", str(args.n_jobs),
            "--bootstrap", str(args.bootstrap), "--sr-config", *map(str, configs),
            "--models", *args.models, "--hyperparameters", str(joint_parameters_path)]
    if args.max_folds:
        argv += ["--max-folds", str(args.max_folds)]
    if args.resume and (output / "run_config.json").is_file():
        argv.append("--resume")
    reference_table = baselines.run(baselines.parse_args(argv))
    reference_table = reference_table.rename(columns={"method": "model"})
    reference_table["method"], reference_table["scope"] = "fixed_configuration", scope
    summaries.append(reference_table)
    combined = pd.concat(summaries, ignore_index=True)
    baselines.atomic_csv(directory / "comparison.csv", combined)
    return combined


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=META_DATASET_PATH)
    parser.add_argument("--hyperparameters", type=Path, help="Base-classifier tuning table; defaults to the input CSV's directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scopes", nargs="+", choices=("positive",), default=["positive"],
                        help="Retain MCC > 0 observations")
    parser.add_argument("--methods", nargs="+", choices=("DE", "EGWO"), default=["DE", "EGWO"],
                        help="Coefficient optimizers for LODO only")
    parser.add_argument("--models", nargs="+", choices=tuple(baselines.MODEL_CLASSES), default=list(baselines.MODEL_CLASSES))
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--lower-factor", type=float, default=.1)
    parser.add_argument("--upper-factor", type=float, default=10.)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--shap-background", type=int, default=100)
    parser.add_argument("--shap-samples", type=int, help="Limit SHAP explanation rows, leaving model fitting unchanged")
    parser.add_argument("--skip-shap", action="store_true", help="Mark the run as incomplete for the manuscript's SHAP update")
    parser.add_argument("--max-folds", type=int, help="Partial LODO only; results are explicitly marked incomplete")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate sources, splits, dependencies and finite reference predictions; no fits or outputs")
    parser.add_argument("--smoke-test", action="store_true", help="Two folds, population6/iteration1, limited baseline and SHAP checks; not publication results")
    args = parser.parse_args(argv)
    if args.population_size < 4 or args.iterations < 1 or args.n_jobs == 0 or args.bootstrap < 0:
        parser.error("Population >=4, iterations >=1, n-jobs nonzero and bootstrap >=0 are required.")
    if args.shap_background < 1 or (args.shap_samples is not None and args.shap_samples < 1) or (args.max_folds is not None and args.max_folds < 1):
        parser.error("SHAP counts and max-folds must be positive.")
    for name in ("scopes", "methods", "models"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"{name} must not contain duplicates.")
    if args.smoke_test:
        args.population_size, args.iterations, args.max_folds = 6, 1, 2
        args.bootstrap, args.shap_background, args.shap_samples = 20, 10, 3
        args.models = ["LinearRegression", "DecisionTreeRegressor"]
    args.hyperparameters = args.hyperparameters or args.data.parent / "base_classifier_hyperparameters.csv"
    return args


def evaluation_ready(args, split_details, completion, lodo_table):
    """Require all requested models and positive-population dataset folds."""
    required_columns = {"model", "scope", "completed_folds", "expected_folds", "n"}
    if (not completion.get("complete") or args.scopes != ["positive"]
            or split_details["positive"]["datasets"] != 65
            or set(args.models) != set(baselines.MODEL_CLASSES)
            or args.shap_samples is not None or args.bootstrap < 2000
            or not required_columns <= set(lodo_table.columns) or lodo_table.empty):
        return False
    expected_models = {*EQUATIONS, *baselines.MODEL_CLASSES}
    return bool(expected_models <= set(lodo_table.model)
                and set(lodo_table.scope) == {"positive"}
                and (lodo_table.completed_folds == 65).all()
                and (lodo_table.expected_folds == 65).all()
                and (lodo_table.n == split_details["positive"]["rows"]).all())


def run(args):
    from src.analysis.model_diagnostics import run_diagnostics
    if args.scopes != ["positive"]:
        raise ValueError("This evaluation supports exactly one scope: positive (MCC > 0).")
    output = args.output_dir.resolve()
    if "archived" in output.parts or output == args.data.resolve().parent or output in args.data.resolve().parents:
        raise ValueError("Choose a separate output directory outside historical or input directories.")
    source_hashes = {"meta_dataset": file_sha256(args.data), "base_hyperparameters": file_sha256(args.hyperparameters)}
    corrected = read_meta_dataset(args.data)
    read_pipeline_table(args.hyperparameters, "base_classifier_hyperparameters.csv")
    positive = corrected.MCC > 0
    if not positive.any():
        raise ValueError("No observations with MCC > 0 remain for this evaluation.")
    # Descriptor correction belongs to preparation. All stages read the same
    # validated input directly; its checksum is recorded without a second copy.
    data_path = args.data.resolve()
    audit = {
        "definition_version": DESCRIPTOR_DEFINITION_VERSION,
        "operation": "Read the already-corrected meta-dataset; no input values changed",
        "source_corpus": {"rows": len(corrected)},
        "evaluated_population": {
            "mcc_filter": "MCC > 0", "rows": int(positive.sum()),
            "datasets": int(corrected.loc[positive, "Dataset"].nunique()),
        },
    }
    split_details = {}
    for scope in args.scopes:
        frame = select_scope(corrected, scope)
        train, test, split = make_split(frame)
        for module in (hroch_lodo, deap_lodo):
            module.coefficient_bounds(args.lower_factor, args.upper_factor)
        for name, equation in EQUATIONS.items():
            values = equation.predict(frame.loc[:, list(hroch.FEATURE_NAMES)])
            if not np.isfinite(values).all():
                raise ValueError(f"Reference {name} form produces invalid outputs in {scope} data.")
        split_details[scope] = {"rows": len(frame), "datasets": int(frame.Dataset.nunique()),
                                "train_rows": len(train), "test_rows": len(test),
                                "split_sha256": hashlib.sha256(split.to_csv(index=False).encode()).hexdigest()}
    config = {
        "schema_version": 4, "protocol": "positive_mcc_supplied_final_equation_evaluation_and_conditional_lodo",
        "descriptor_version": DESCRIPTOR_DEFINITION_VERSION,
        "sources": {"meta_dataset": str(args.data.resolve()), "base_hyperparameters": str(args.hyperparameters.resolve())},
        "source_sha256": source_hashes, "corrected_data_sha256": source_hashes["meta_dataset"],
        "code_sha256": {name: file_sha256(ROOT / name) for name in CODE_FILES},
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyBlindOpt", "EmptySpaceSearch", "torann", "shap", "matplotlib", "threadpoolctl")},
        "scopes": args.scopes, "methods": args.methods, "models": args.models,
        "population_size": args.population_size, "iterations": args.iterations, "n_jobs": args.n_jobs,
        "coefficient_factors": [args.lower_factor, args.upper_factor], "bootstrap": args.bootstrap,
        "max_folds": args.max_folds, "smoke_test": args.smoke_test,
        "shap_background": args.shap_background, "shap_samples": args.shap_samples, "skip_shap": args.skip_shap,
        "splits": split_details, "seed": SEED,
        "rowwise_equations": {name: supplied_equation_record(name, equation) for name, equation in EQUATIONS.items()},
        "deap_form": "src/equations/extended.py: supplied final grouped19-coefficient expression",
        "mcc_filter": "MCC > 0",
        "evaluated_population": audit["evaluated_population"],
        "source_corpus": audit["source_corpus"],
        "selection": "Supplied final symbolic defaults evaluated directly; missing reference settings selected on row-wise training; LODO constants fitted within each training fold",
        "reference_selection": {
            feature_set: selection_plan(reference_model_names(args, feature_set), feature_set=feature_set, cv=4,
                                        smoke_test=args.smoke_test)
            for feature_set in ("joint", "dataset_only")
        },
        "reference_configuration_sources": {
            feature_set: historical_configuration_sources(feature_set)
            for feature_set in ("joint", "dataset_only")
        },
        "scope_statement": (
            "Supplied final equations are evaluated here; this runner does not perform symbolic discovery. "
            "LODO is conditional on fixed forms and selected reference configurations, without nested selection."
        ),
    }
    print(json.dumps({"source_corpus": audit["source_corpus"], "data": split_details,
                      "evaluated_population": audit["evaluated_population"],
                      "optimizers": args.methods, "reference_selection": config["reference_selection"],
                      "output": str(output), "smoke_test": args.smoke_test}, indent=2), flush=True)
    if {"meta_dataset": file_sha256(args.data), "base_hyperparameters": file_sha256(args.hyperparameters)} != source_hashes:
        raise ValueError("Source tables changed while preparing the run.")
    if args.dry_run:
        print("Dry run passed. No models fitted and no results written.", flush=True)
        return config
    manifest = output / "run_config.json"
    if args.resume:
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != clean_json(config):
            raise ValueError("Cannot resume: inputs, code, versions or experimental settings differ.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty. Use --resume with identical settings or choose a new directory.")
        output.mkdir(parents=True, exist_ok=True)
        save_json(manifest, config)
    lodo_tables = []
    for scope in args.scopes:
        selected = select_scope(corrected, scope)
        directory = output / scope
        directory.mkdir(exist_ok=True)
        fitted, _, reference_selection = run_rowwise(selected, directory / "rowwise", config, args, scope)
        lodo_tables.append(run_lodo(data_path, directory / "lodo", args, scope,
                                    directory / "rowwise" / "reference_parameters_joint.json"))
        diagnostics = directory / "diagnostics"
        marker = diagnostics / "diagnostics_complete.json"
        expected = {"run_sha256": content_fingerprint(config), "scope": scope}
        if args.resume and marker.is_file():
            completion = json.loads(marker.read_text(encoding="utf-8"))
            if completion.get("context") != expected:
                raise ValueError("Diagnostics recovery metadata differs.")
            for name, fingerprint in completion["files"].items():
                if file_sha256(diagnostics / name) != fingerprint:
                    raise ValueError(f"Changed or missing diagnostics artifact: {name}")
            print(f"[{scope}] Diagnostics: reused completed outputs", flush=True)
        else:
            print(f"[{scope}] Correlations, derivative audit and SHAP", flush=True)
            kwargs = {"n_jobs": args.n_jobs, "shap_background": args.shap_background,
                      "shap_samples": args.shap_samples, "seed": SEED,
                      "rf_parameters": (reference_selection["joint"]["parameters"]["RandomForestRegressor"]
                                        if not args.skip_shap else {}),
                      "hroch_coefficients": fitted["HROCH"]["coefficients"]}
            run_diagnostics(selected, directory / "rowwise", diagnostics,
                            complete_frame=corrected, include_shap=not args.skip_shap, **kwargs)
            files = {str(p.relative_to(diagnostics)): file_sha256(p) for p in diagnostics.rglob("*") if p.is_file() and p != marker}
            save_json(marker, {"context": expected, "files": files})
    lodo_table = pd.concat(lodo_tables, ignore_index=True)
    full_folds = bool(lodo_table.full_corpus_complete.fillna(False).all())
    valid = bool((lodo_table.invalid_predictions.fillna(0) == 0).all()
                 and np.isfinite(lodo_table[["r2", "mae", "rmse"]].to_numpy(dtype=float)).all())
    result = {"complete": full_folds and valid and not args.skip_shap and not args.smoke_test,
              "execution_finished": True, "smoke_test": args.smoke_test,
              "positive_population_lodo_complete": full_folds, "finite_lodo_predictions_and_metrics": valid,
              "shap_skipped": args.skip_shap, "source_tables_preserved": True,
              "scopes": args.scopes, "mcc_filter": "MCC > 0",
              "positive_mcc_datasets": split_details["positive"]["datasets"],
              "results": str(output), "config_sha256": content_fingerprint(config),
              "symbolic_rowwise_procedure": "direct evaluation of supplied final equations",
              "newly_fitted_stages": ["reference_selection_and_fits", "symbolic_LODO", "diagnostic_RF"]}
    result["evaluation_ready"] = evaluation_ready(args, split_details, result, lodo_table)
    if {"meta_dataset": file_sha256(args.data), "base_hyperparameters": file_sha256(args.hyperparameters)} != source_hashes:
        raise ValueError("Source tables changed during execution.")
    if {feature_set: historical_configuration_sources(feature_set)
        for feature_set in ("joint", "dataset_only")} != config["reference_configuration_sources"]:
        raise ValueError("Historical reference-configuration evidence changed during execution.")
    save_json(output / "completion.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    try:
        run(parse_args(argv))
    except (ValueError, OSError, RuntimeError, TypeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
