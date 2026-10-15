"""Compare the final HROCH and grouped DEAP equations on corrected descriptors."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.equations.hroch import FEATURE_NAMES, predict as hroch_predict
from src.equations.extended import predict as deap_predict
from src.paths import META_DATASET_PATH, output_run_directory
from src.utils.meta_regression import adjusted_r2, smape, stratified_regression_split
from src.utils.pipeline_integrity import read_meta_dataset


# Preserve the original figure's page size when LaTeX scales it to text width.
FIGURE_PAGE_SIZE = (8.253510357480556, 4.092259114583333)


def load_deap_predictor(summary_path: Path) -> Callable[[pd.DataFrame], np.ndarray]:
    """Load a single prefix tree and restrict it to the documented primitives."""
    summary = pd.read_csv(summary_path)
    if len(summary) != 1 or "best_expression" not in summary.columns:
        raise ValueError("The DEAP summary must contain exactly one best_expression.")
    expression = summary.iloc[0]["best_expression"]
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("The DEAP expression is missing or empty.")
    operations = {
        "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
        "sq2": lambda value: value * value,
        "sq3": lambda value: value * value * value,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "identity": lambda value: value,
    }
    arities = {name: 2 if name in {"add", "sub", "mul", "div"} else 1
               for name in operations}
    allowed_names = set(operations) | {f"X{i}" for i in range(len(FEATURE_NAMES))}
    tree = ast.parse(expression.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.Call, ast.Name, ast.Load,
                                 ast.Constant, ast.UnaryOp, ast.USub, ast.UAdd)):
            raise ValueError(f"Unexpected expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unexpected expression name: {node.id}")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in operations or node.keywords
            or len(node.args) != arities[node.func.id]
        ):
            raise ValueError("Unexpected expression call or primitive arity.")
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            raise ValueError("Only numerical expression constants are allowed.")
    code = compile(tree, str(summary_path), "eval")

    def predict(features: pd.DataFrame) -> np.ndarray:
        ordered = features.loc[:, list(FEATURE_NAMES)]
        environment = {
            **operations,
            **{f"X{i}": ordered.iloc[:, i].to_numpy() for i in range(len(FEATURE_NAMES))},
        }
        with np.errstate(all="ignore"):
            predictions = np.asarray(
                eval(code, {"__builtins__": {}}, environment), dtype=float
            )
        if predictions.shape != (len(features),):
            raise ValueError("The DEAP tree must predict one value per observation.")
        return predictions

    return predict


def plot_comparison(targets, predictions: dict[str, np.ndarray], output_dir: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
    lower = min(0.0, min(float(values.min()) for values in predictions.values())) - .03
    upper = max(1.0, max(float(values.max()) for values in predictions.values())) + .03
    titles = {"HROCH": "HROCH: simpler expression", "DEAP": "DEAP: more complex expression"}
    for axis, name in zip(axes, ("HROCH", "DEAP")):
        values = predictions[name]
        axis.scatter(targets, values, s=8, alpha=.23, edgecolors="none",
                     color="#145980", rasterized=True)
        axis.plot([lower, upper], [lower, upper], "--", color="#B44135",
                  lw=1.3, label="Ideal agreement")
        axis.set(xlabel="Observed MCC", title=titles[name],
                 xlim=(lower, 1.0), ylim=(lower, upper))
        axis.set_aspect("equal", adjustable="box")
        axis.text(
            .04, .96,
            f"$R^2$ = {r2_score(targets, values):.3f}\n"
            f"MAE = {mean_absolute_error(targets, values):.3f}",
            transform=axis.transAxes, va="top",
            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"},
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted MCC")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    figure.tight_layout()
    figure.canvas.draw()
    bounds = figure.get_tightbbox(figure.canvas.get_renderer()).padded(.1)
    width, height = FIGURE_PAGE_SIZE
    export_bounds = Bbox.from_bounds(
        (bounds.x0 + bounds.x1 - width) / 2,
        (bounds.y0 + bounds.y1 - height) / 2,
        width, height,
    )
    figure.savefig(output_dir / "Fig5_symbolic_comparison.pdf", bbox_inches=export_bounds)
    figure.savefig(output_dir / "Fig5_symbolic_comparison.png", bbox_inches=export_bounds, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=META_DATASET_PATH)
    parser.add_argument("--deap-summary", type=Path,
                        help="Explicitly evaluate a historical DEAP prefix tree instead of the final equation")
    parser.add_argument("--output-dir", type=Path,
                        default=output_run_directory("figures", "symbolic_comparison"))
    args = parser.parse_args()
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("Choose a fresh output directory to preserve existing files.")

    frame = read_meta_dataset(args.data)
    frame = frame[frame["MCC"] > 0]
    features = frame.loc[:, list(FEATURE_NAMES)]
    targets = frame["MCC"]
    X_train, X_test, y_train, y_test = stratified_regression_split(features, targets)
    predictors = {"HROCH": hroch_predict,
                  "DEAP": load_deap_predictor(args.deap_summary) if args.deap_summary else deap_predict}
    metrics = []
    test_predictions = {}
    for name, predictor in predictors.items():
        for partition, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
            values = predictor(X)
            if not np.isfinite(values).all():
                raise ValueError(f"Invalid {name} predictions in the {partition} partition.")
            r2 = float(r2_score(y, values))
            metrics.append({
                "model": name, "partition": partition, "n": len(y), "r2": r2,
                "adjusted_r2": adjusted_r2(r2, len(y), len(FEATURE_NAMES)),
                "smape": smape(y, values),
                "mae": float(mean_absolute_error(y, values)),
            })
            if partition == "test":
                test_predictions[name] = values

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(y_test, test_predictions, args.output_dir)
    table = pd.DataFrame(metrics)
    table.to_csv(args.output_dir / "symbolic_comparison_metrics.csv", index=False)
    print(table.to_string(index=False))
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
