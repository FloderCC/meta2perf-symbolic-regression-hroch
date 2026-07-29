"""Plot Pearson and Spearman associations with MCC."""

import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.paths import (
    PLOTS_PDF_DIR,
    PLOTS_PNG_DIR,
    RESULTS_DIR,
    ensure_output_directories,
)

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
warnings.filterwarnings("ignore")

ensure_output_directories()
df_meta = pd.read_csv(RESULTS_DIR / "meta_dataset.csv")
df_meta = df_meta.drop(columns=['Seed','Dataset','Sample Size','Model'])

# Rename model-related columns
model_related_columns_map = {
    'Model Type': 'model_type',
    'Processing Units Number': 'proc_units',
    'Training Complexity': 'train_complexity',
    'Prediction Complexity': 'inf_complexity',
    'Regularization': 'regularization',
    'Robust to Outliers': 'rob_outliers',
    'Representational Capacity': 'repr_capacity',
}
df_meta = df_meta.rename(columns=model_related_columns_map)

# Drop columns with NaNs
cols_with_nan = df_meta.columns[df_meta.isnull().any()].tolist()
print(f"Columns with NaN values: {cols_with_nan}")
df = df_meta.drop(columns=cols_with_nan)

# --- Features & target ---
meta_features = df.drop(columns=["MCC"])
target = df["MCC"]

# --- Correlations ---
pcc = meta_features.corrwith(target, method="pearson")
srcc = meta_features.corrwith(target, method="spearman")

correlations = pd.DataFrame({"PCC": pcc, "SRCC": srcc}).sort_values("SRCC", ascending=True)

# --- Define groups ---
MODEL_FEATURES_CANON = {
    'model_type', 'proc_units', 'train_complexity', 'inf_complexity',
    'regularization', 'rob_outliers', 'repr_capacity'
}
dataset_features = [f for f in correlations.index if f not in MODEL_FEATURES_CANON]

def is_dataset(feat: str) -> bool:
    return feat in dataset_features

# --- Colors: just two flat colors ---
DATASET_COLOR = "#e64042"  # red
MODEL_COLOR = "#377eb8"    # blue

def color_for(feat: str):
    return DATASET_COLOR if is_dataset(feat) else MODEL_COLOR

pcc_colors = [color_for(f) for f in correlations.index]
srcc_colors = [color_for(f) for f in correlations.index]

# --- Plot ---
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 16
})

fig, axes = plt.subplots(ncols=2, figsize=(12.5, 6), sharey=True)

# Left: PCC
axes[0].barh(correlations.index, correlations["PCC"],
             color=pcc_colors, edgecolor='none', linewidth=0)
axes[0].set_xlabel("PCC")
axes[0].axvline(0, color='gray', linestyle='--')
axes[0].set_ylabel("Meta-features")

# Right: SRCC
axes[1].barh(correlations.index, correlations["SRCC"],
             color=srcc_colors, edgecolor='none', linewidth=0)
axes[1].set_xlabel("SRCC")
axes[1].axvline(0, color='gray', linestyle='--')

# Shared y-ticks
y_pos = np.arange(len(correlations.index))
for ax in axes:
    ax.set_yticks(y_pos)
    ax.set_yticklabels(correlations.index)

# Dynamic x-ticks/limits
for ax, col in zip(axes, ["PCC", "SRCC"]):
    min_val = correlations[col].min()
    max_val = correlations[col].max()
    ax.set_xticks(np.arange(np.floor(min_val * 5) / 5, np.ceil(max_val * 5) / 5 + 0.2, 0.2))
    ax.set_xlim(min_val - 0.025, max_val + 0.025)
    ax.grid(axis='x', linestyle='--', alpha=0.25)

# --- Figure-level legend ---
legend_handles = [
    Patch(facecolor=DATASET_COLOR, edgecolor="none", label="Dataset features"),
    Patch(facecolor=MODEL_COLOR, edgecolor="none", label="Model features"),
]

plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.565, 0.03),
    frameon=False
)

# Save
plt.savefig(
    PLOTS_PDF_DIR / "feature_correlations.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.savefig(
    PLOTS_PNG_DIR / "feature_correlations.png",
    format="png",
    bbox_inches="tight",
    dpi=300,
)
plt.close(fig)
