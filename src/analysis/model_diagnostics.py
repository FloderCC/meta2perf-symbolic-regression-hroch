"""Feature associations, symbolic sensitivity and reference-model diagnostics.

The caller supplies the prepared data, the saved row-wise split and the fixed
HROCH coefficients. Nothing in this module searches for a symbolic structure or
selects hyperparameters. Identifier columns are used only for matching and for
auditing partition membership, never as random-forest predictors.

All directional derivatives refer to the *raw* rational equation. They are
partial derivatives of its continuous arithmetic extension, not causal effects
or an assumption that category codes have intrinsic quantitative distances.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.equations.hroch import COEFFICIENT_NAMES, FEATURE_NAMES, predict
from src.utils.artifact_integrity import atomic_json


SHORT_NAMES = {
    "Processing Units Number": "proc_units", "Model Type": "model_type",
    "Training Complexity": "train_complexity",
    "Prediction Complexity": "inf_complexity", "Regularization": "regularization",
    "Robust to Outliers": "rob_outliers", "Representational Capacity": "repr_capacity",
}
CATEGORY_CODES = ["Model Type", "Regularization"]


def _validated_features(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(FEATURE_NAMES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing equation descriptors: {missing}")
    values = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Descriptors must be finite; no rows are silently removed.")
    return values


def _validated_frame(frame: pd.DataFrame) -> None:
    _validated_features(frame)
    if "row_id" not in frame or "MCC" not in frame:
        raise ValueError("Diagnostics require row_id and MCC columns.")
    if frame.empty or frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("row_id must identify each nonempty input row uniquely.")
    if not np.isfinite(frame["MCC"].to_numpy(dtype=float)).all():
        raise ValueError("MCC must be finite.")


def _finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _metrics(observed: np.ndarray, prediction: np.ndarray) -> dict:
    observed = np.asarray(observed, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if observed.shape != prediction.shape or not np.isfinite(observed).all():
        raise ValueError("Observed and predicted arrays must align and observations must be finite.")
    invalid = int(np.count_nonzero(~np.isfinite(prediction)))
    result = {"n": int(len(observed)), "invalid_predictions": invalid}
    if not len(observed) or invalid:
        return {**result, "r2": None, "mae": None, "rmse": None}
    with np.errstate(over="ignore", invalid="ignore"):
        residual = observed - prediction
        total = float(np.sum(np.square(observed - observed.mean())))
        squared_error = float(np.dot(residual, residual))
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(np.square(residual))))
    return {
        **result, "r2": _finite_number(1 - squared_error / total)
        if total > 0 and len(observed) > 1 else None,
        "mae": _finite_number(mae), "rmse": _finite_number(rmse),
    }


def split_frames(frame: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match a saved split by row id, rejecting omissions, duplicates and overlap."""
    _validated_frame(frame)
    if not {"row_id", "partition"} <= set(split.columns):
        raise ValueError("split.csv must contain row_id and partition.")
    if split["row_id"].isna().any() or split["row_id"].duplicated().any():
        raise ValueError("Every split row_id must occur exactly once.")
    if set(split["row_id"]) != set(frame["row_id"]):
        raise ValueError("Split and diagnostic input must contain exactly the same row ids.")
    if split["partition"].isna().any() or set(split["partition"]) != {"train", "test"}:
        raise ValueError("The split must contain nonempty train and test partitions only.")
    lookup = frame.set_index("row_id", drop=False)
    train_ids = split.loc[split["partition"] == "train", "row_id"]
    test_ids = split.loc[split["partition"] == "test", "row_id"]
    return lookup.loc[train_ids].reset_index(drop=True), lookup.loc[test_ids].reset_index(drop=True)


def hroch_directional_audit(features: pd.DataFrame, coefficients) -> pd.DataFrame:
    """Analytic partial derivatives for all 19 descriptors in x0..x18 order.

    Differentiation follows the same intermediate terms as hroch.predict and
    includes both numerator and denominator. Nonfinite derivatives at singular
    inputs stay visible and are counted separately by directional_report.
    """
    x = _validated_features(features)
    coef = np.asarray(coefficients, dtype=float)
    if coef.shape != (len(COEFFICIENT_NAMES),) or not np.isfinite(coef).all():
        raise ValueError("Expected six finite coefficients in a..f order.")
    a, b, c, d, e, f = coef
    unit = np.eye(len(FEATURE_NAMES))
    u = x[:, 13] - x[:, 15]
    du = unit[13] - unit[15]
    t1 = u - a
    t2 = x[:, 11] + b
    dt2 = unit[11]
    t3 = x[:, 11] + x[:, 9] + x[:, 1]
    dt3 = unit[11] + unit[9] + unit[1]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        q = 1 + x[:, 14] / t1
        dq = unit[14] / t1[:, None] - (x[:, 14] / t1**2)[:, None] * du
        t4 = q * x[:, 15] * t2
        dt4 = ((x[:, 15] * t2)[:, None] * dq
               + (q * t2)[:, None] * unit[15]
               + (q * x[:, 15])[:, None] * dt2)
        t5 = x[:, 10] * t4 / t3
        dt5 = ((t4 / t3)[:, None] * unit[10]
               + (x[:, 10] / t3)[:, None] * dt4
               - (x[:, 10] * t4 / t3**2)[:, None] * dt3)
        k = x[:, 4] - d / x[:, 6]
        dk = unit[4] + (d / x[:, 6]**2)[:, None] * unit[6]
        g = c + k * u
        dg = u[:, None] * dk + k[:, None] * du
        h = g * x[:, 0] / x[:, 2]
        dh = ((x[:, 0] / x[:, 2])[:, None] * dg
              + (g / x[:, 2])[:, None] * unit[0]
              - (g * x[:, 0] / x[:, 2]**2)[:, None] * unit[2])
        t6 = h**2 + x[:, 4] + t2
        dt6 = 2 * h[:, None] * dh + unit[4] + dt2
        numerator = ((x[:, 17] + 1) * t5 - t6
                     - (x[:, 5] + x[:, 6]) / e + x[:, 1])
        d_numerator = (t5[:, None] * unit[17]
                       + (x[:, 17] + 1)[:, None] * dt5 - dt6
                       - (unit[5] + unit[6]) / e + unit[1])
        v = t4 + x[:, 9] + x[:, 1]
        dv = dt4 + unit[9] + unit[1]
        denominator = f * t3 * v - t6
        d_denominator = f * (v[:, None] * dt3 + t3[:, None] * dv) - dt6
        derivative = (d_numerator * denominator[:, None]
                      - numerator[:, None] * d_denominator) / denominator[:, None]**2
    return pd.DataFrame(derivative, columns=list(FEATURE_NAMES), index=features.index)


def directional_report(frame: pd.DataFrame, coefficients, output_dir: Path,
                       *, complete_frame: pd.DataFrame | None = None,
                       zero_tolerance: float = 0.0) -> dict:
    """Save derivative signs and raw prediction range for the held-out rows."""
    _validated_frame(frame)
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("Derivative zero tolerance must be finite and nonnegative.")
    output_dir.mkdir(parents=True, exist_ok=True)
    derivatives = hroch_directional_audit(frame, coefficients)
    per_row = derivatives.rename(columns={name: f"d_{SHORT_NAMES.get(name, name)}"
                                           for name in FEATURE_NAMES})
    per_row.insert(0, "row_id", frame["row_id"].to_numpy())
    prediction = predict(frame, coefficients)
    per_row["prediction_raw"] = prediction
    per_row.to_csv(output_dir / "hroch_test_derivatives.csv", index=False)
    rows = []
    for name in FEATURE_NAMES:
        values = derivatives[name].to_numpy()
        finite = np.isfinite(values)
        n_finite = int(finite.sum())
        counts = {
            "positive": int(np.count_nonzero(finite & (values > zero_tolerance))),
            "negative": int(np.count_nonzero(finite & (values < -zero_tolerance))),
            "zero": int(np.count_nonzero(finite & (np.abs(values) <= zero_tolerance))),
            "invalid": int(np.count_nonzero(~finite)),
        }
        rows.append({
            "feature": name, "name": SHORT_NAMES.get(name, name), "n_test": len(frame),
            "n_finite": n_finite, **{f"n_{key}": value for key, value in counts.items()},
            **{f"share_{key}_all_test": value / len(frame) for key, value in counts.items()},
            "share_positive_finite": counts["positive"] / n_finite if n_finite else np.nan,
            "share_negative_finite": counts["negative"] / n_finite if n_finite else np.nan,
        })
    signs = pd.DataFrame(rows)
    signs.to_csv(output_dir / "hroch_derivative_signs.csv", index=False)
    corpus = frame if complete_frame is None else complete_frame
    _validated_features(corpus)
    finite_prediction = np.isfinite(prediction)
    gravity = corpus["gravity"].to_numpy(dtype=float)
    report = {
        "evaluation_partition": "row-wise test", "n_test": len(frame),
        "coefficients": np.asarray(coefficients, dtype=float).tolist(),
        "derivative_method": "analytic quotient and chain rules for all 19 descriptors",
        "derivative_zero_tolerance": zero_tolerance,
        "derivative_scope": "Raw equation; all other descriptors and coefficients held fixed.",
        "category_interpretation": (
            "Arithmetic derivatives for category and heuristic codes are conditional on the "
            "documented encoding, not intrinsic category distances or causal effects."
        ),
        "invalid_raw_predictions": int(np.count_nonzero(~finite_prediction)),
        "raw_above_mcc_upper_bound": int(np.count_nonzero(finite_prediction & (prediction > 1))),
        "raw_below_mcc_lower_bound": int(np.count_nonzero(finite_prediction & (prediction < -1))),
        "share_raw_above_one_all_test": float(np.count_nonzero(finite_prediction & (prediction > 1))
                                                / len(frame)),
        "minimum_test_gravity": float(frame["gravity"].min()),
        "gravity_audit_rows": len(corpus), "minimum_corpus_gravity": float(gravity.min()),
        "zero_corpus_gravity": int(np.count_nonzero(gravity == 0)),
        "zero_gravity_handling": "No substitute value. Singular outputs remain invalid predictions.",
        "test_metrics_raw": _metrics(frame["MCC"].to_numpy(), prediction),
        "outputs": ["hroch_test_derivatives.csv", "hroch_derivative_signs.csv"],
    }
    atomic_json(output_dir / "hroch_directional_report.json", report)
    return report


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def correlations(frame: pd.DataFrame, output_dir: Path) -> dict:
    """PCC and SRCC on the explicitly supplied analysis population, without IDs."""
    _validated_frame(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors = frame.loc[:, list(FEATURE_NAMES)]
    values = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "PCC": descriptors.corrwith(frame["MCC"], method="pearson").to_numpy(),
        "SRCC": descriptors.corrwith(frame["MCC"], method="spearman").to_numpy(),
    })
    values["name"] = values["feature"].map(lambda name: SHORT_NAMES.get(name, name))
    values.to_csv(output_dir / "feature_correlations.csv", index=False)
    ordered = values.sort_values("SRCC", na_position="first")
    colors = ["#377eb8" if name in SHORT_NAMES else "#e64042" for name in ordered["feature"]]
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for ax, column in zip(axes, ("PCC", "SRCC")):
        ax.barh(ordered["name"], ordered[column], color=colors)
        ax.set_xlabel(column)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"feature_correlations.{extension}", dpi=180)
    plt.close(fig)
    report = {
        "n": len(frame), "population": "Exactly the supplied experiment population",
        "feature_order": list(FEATURE_NAMES),
        "category_interpretation": "PCC and SRCC for category codes depend on the documented encoding.",
        "constant_features": [name for name in FEATURE_NAMES if frame[name].nunique() == 1],
        "outputs": ["feature_correlations.csv", "feature_correlations.png", "feature_correlations.pdf"],
    }
    atomic_json(output_dir / "correlations_report.json", report)
    return report


def shap_row_selection(train: pd.DataFrame, test: pd.DataFrame, *,
                       background_count: int, sample_count: int | None,
                       seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select only training rows for the background and only test rows to explain."""
    if background_count <= 0 or (sample_count is not None and sample_count <= 0):
        raise ValueError("SHAP background/sample counts must be positive.")
    if train.empty or test.empty:
        raise ValueError("SHAP needs nonempty training and test partitions.")
    if set(train["row_id"]) & set(test["row_id"]):
        raise ValueError("SHAP training and test rows overlap.")
    background_rng, explained_rng = [np.random.default_rng(part)
                                     for part in np.random.SeedSequence(seed).spawn(2)]
    background_positions = np.sort(background_rng.choice(len(train), min(background_count, len(train)),
                                                         replace=False))
    explained_positions = (np.arange(len(test)) if sample_count is None else
                           np.sort(explained_rng.choice(len(test), min(sample_count, len(test)), replace=False)))
    return train.iloc[background_positions].copy(), test.iloc[explained_positions].copy()


def explain_random_forest(frame: pd.DataFrame, split: pd.DataFrame, output_dir: Path,
                          parameters: dict, *, n_jobs: int = 1,
                          shap_background: int = 100, shap_samples: int | None = None,
                          seed: int = 42) -> dict:
    """Fit the fixed RF winner on train only, and explain test with TreeSHAP.

    This uses exact interventional TreeSHAP with a deterministic background drawn
    only from training. It replaces the archived approximate KernelSHAP routine;
    the method and complete row memberships are recorded explicitly.
    """
    from sklearn.ensemble import RandomForestRegressor
    import shap

    train, test = split_frames(frame, split)
    background, explained = shap_row_selection(
        train, test, background_count=shap_background, sample_count=shap_samples, seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(parameters)
    settings.update(random_state=seed, n_jobs=n_jobs)
    model = RandomForestRegressor(**settings)
    model.fit(train.loc[:, list(FEATURE_NAMES)], train["MCC"])
    background_values = background.loc[:, list(FEATURE_NAMES)]
    # Preserve the requested deterministic background instead of the masker's
    # default secondary subsample of 100 rows.
    masker = shap.maskers.Independent(background_values, max_samples=len(background_values))
    explainer = shap.TreeExplainer(model, data=masker, feature_perturbation="interventional",
                                   model_output="raw", feature_names=list(FEATURE_NAMES))
    values = np.asarray(explainer.shap_values(explained.loc[:, list(FEATURE_NAMES)],
                                            check_additivity=True), dtype=float)
    if values.shape != (len(explained), len(FEATURE_NAMES)) or not np.isfinite(values).all():
        raise RuntimeError("TreeSHAP returned nonfinite or misaligned explanation values.")
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    explanation_prediction = model.predict(explained.loc[:, list(FEATURE_NAMES)])
    additivity_error = float(np.max(np.abs(values.sum(axis=1) + expected_value - explanation_prediction)))
    per_row = pd.DataFrame(values, columns=list(FEATURE_NAMES))
    per_row.insert(0, "row_id", explained["row_id"].to_numpy())
    per_row.to_csv(output_dir / "random_forest_shap_values.csv", index=False)
    relevance = pd.DataFrame({"feature": FEATURE_NAMES, "mean_absolute_shap": np.mean(np.abs(values), axis=0)})
    relevance["name"] = relevance["feature"].map(lambda name: SHORT_NAMES.get(name, name))
    relevance.to_csv(output_dir / "joint_random_forest_shap.csv", index=False)
    ordered = relevance.sort_values("mean_absolute_shap")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = ["#377eb8" if name in SHORT_NAMES else "#e64042" for name in ordered["feature"]]
    ax.barh(ordered["name"], ordered["mean_absolute_shap"], color=colors)
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"joint_random_forest_shap.{extension}", dpi=180)
    plt.close(fig)
    report = {
        "method": "interventional TreeSHAP", "approximate": False,
        "archived_method": "KernelSHAP (approximate); these outputs explicitly use TreeSHAP",
        "shap_version": shap.__version__, "feature_order": list(FEATURE_NAMES),
        "hyperparameter_search_performed": False,
        "random_forest_parameters": model.get_params(deep=False), "seed": seed,
        "training_row_ids": train["row_id"].tolist(),
        "background_row_ids": background["row_id"].tolist(),
        "test_row_ids": test["row_id"].tolist(),
        "explained_row_ids": explained["row_id"].tolist(),
        "background_source": "training partition only", "explanation_source": "test partition only",
        "expected_value": expected_value, "maximum_additivity_error": additivity_error,
        "test_metrics_raw": _metrics(test["MCC"].to_numpy(), model.predict(test.loc[:, list(FEATURE_NAMES)])),
        "outputs": ["random_forest_shap_values.csv", "joint_random_forest_shap.csv",
                    "joint_random_forest_shap.png", "joint_random_forest_shap.pdf"],
    }
    atomic_json(output_dir / "random_forest_shap_report.json", report)
    return report


def _read_hroch_coefficients(path: Path) -> np.ndarray:
    """Read the parent's row-wise coefficients schema without implicit fallback."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["HROCH"]
    if isinstance(entry, dict) and "coefficients" in entry:
        return np.asarray(entry["coefficients"], dtype=float)
    if isinstance(entry, dict) and "DE" in entry:
        entry = entry["DE"]
        return np.asarray(entry["coefficients"] if isinstance(entry, dict) else entry, dtype=float)
    return np.asarray(entry, dtype=float)


def run_diagnostics(frame: pd.DataFrame, rowwise_dir: Path, output_dir: Path,
                    *, n_jobs: int = 1, shap_background: int = 100,
                    shap_samples: int | None = None, seed: int = 42,
                    rf_parameters: dict | None = None, hroch_coefficients=None,
                    complete_frame: pd.DataFrame | None = None,
                    include_shap: bool = True) -> dict:
    """Run the saved split's diagnostic suite without symbolic or HP searches.

    frame is exactly the row-wise population; complete_frame optionally supplies
    the complete corpus for the gravity-domain audit. split.csv must
    cover every frame row_id exactly once. Coefficients and fixed RF parameters
    can be supplied explicitly, avoiding dependency on JSON layout details.
    """
    if rf_parameters is None:
        from src.experiments.baselines.benchmark_lodo import FIXED_PARAMETERS
        rf_parameters = FIXED_PARAMETERS["RandomForestRegressor"]
    output_dir.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(rowwise_dir / "split.csv")
    train, test = split_frames(frame, split)
    coefficients = (_read_hroch_coefficients(rowwise_dir / "coefficients.json")
                    if hroch_coefficients is None else hroch_coefficients)
    report = {
        "status": "complete", "n": len(frame), "n_train": len(train), "n_test": len(test),
        "seed": seed, "feature_order": list(FEATURE_NAMES),
        "correlations": correlations(frame, output_dir / "correlations"),
        "hroch_directional": directional_report(test, coefficients, output_dir / "directional",
                                                  complete_frame=complete_frame if complete_frame is not None else frame),
    }
    if include_shap:
        report["random_forest_shap"] = explain_random_forest(
            frame, split, output_dir / "shap", rf_parameters, n_jobs=n_jobs,
            shap_background=shap_background, shap_samples=shap_samples, seed=seed,
        )
    else:
        report["random_forest_shap"] = {"status": "not run", "reason": "include_shap=False"}
    atomic_json(output_dir / "diagnostics_report.json", report)
    return report
