"""Plot archived or regenerated KernelSHAP feature relevance values."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.paths import (
    ensure_output_directories,
    latest_output_path,
    output_run_directory,
)


RENAME_MAP = {
    "Processing Units Number": "proc_units",
    "Model Type": "model_type",
    "Training Complexity": "train_complexity",
    "Prediction Complexity": "inf_complexity",
    "Regularization": "regularization",
    "Robust to Outliers": "rob_outliers",
    "Representational Capacity": "repr_capacity",
}
MODEL_FEATURES = set(RENAME_MAP.values())
DATASET_COLOR = "#e64042"
MODEL_COLOR = "#377eb8"


def main() -> None:
    figures_dir = output_run_directory("figures", "shap_relevance")
    ensure_output_directories(figures_dir)
    relevance = pd.read_csv(latest_output_path(
        "random_forest_shap", filename="joint_random_forest_shap.csv"
    ))
    relevance["feature"] = relevance["feature"].replace(RENAME_MAP)
    relevance["group"] = np.where(
        relevance["feature"].isin(MODEL_FEATURES),
        "model",
        "dataset",
    )
    relevance = relevance.sort_values("mean_absolute_shap")
    colors = [
        DATASET_COLOR if group == "dataset" else MODEL_COLOR
        for group in relevance["group"]
    ]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
        }
    )
    fig, axis = plt.subplots(figsize=(6, 7))
    axis.barh(
        relevance["feature"],
        relevance["mean_absolute_shap"],
        color=colors,
        edgecolor="none",
    )
    axis.set_xlabel("Feature relevance (mean absolute SHAP value)")
    axis.set_ylabel("Meta-features")
    axis.set_xlim(0, float(relevance["mean_absolute_shap"].max()) * 1.18)
    axis.grid(axis="x", linestyle="--", alpha=0.25)
    for spine in ("top", "right", "left", "bottom"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(
        handles=[
            Patch(facecolor=DATASET_COLOR, label="Dataset features"),
            Patch(facecolor=MODEL_COLOR, label="Algorithm features"),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(
        figures_dir / "shap_feature_relevance.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        figures_dir / "shap_feature_relevance.png",
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
