# results based on dataset features


import random
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from src.utils.pipeline_integrity import read_meta_dataset
from src.paths import META_DATASET_PATH, ensure_output_directories, output_run_directory

OUTPUT_DIR = output_run_directory("figures", "extended_equation_predictions")

def plot_regression_results(y_true, y_pred, title, target='', set_name='train'):
    """
    Plots true vs. predicted values for regression results, with R² and MAE metrics.

    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values
    - title: str, plot title

    Saves pred_vs_true.pdf in this figure run's output directory.
    """
    # Compute metrics
    # r2 = r2_score(y_true, y_pred)
    # mae = mean_absolute_error(y_true, y_pred)

    # print(min(y_pred))

    # Create plot
    plt.figure(figsize=(5, 5))
    plt.scatter(
        y_true,
        y_pred,
        s=16,
        alpha=0.3,
        edgecolors='none',
        label='Predictions',
        # color='#054f82'
        color='#054f82'
    )
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
        linestyle='--',
        color='#c42525',
        label='Ideal Fit (y = x)')

    # limit the x and y axis to the range of the data
    x_margin = 0.03 * (max(y_true) - min(y_true))
    y_margin = 0.03 * (max(y_pred) - min(y_pred))  # Similarly for y_pred if needed

    plt.xlim(min(y_true) - x_margin, max(y_true) + x_margin)
    plt.ylim(min(y_pred)- y_margin, max(y_pred) + y_margin)
    # plt.ylim(0, 1)

    # Labels and title
    plt.xlabel(f'True {target} Values', fontsize=12)
    plt.ylabel(f'Predicted {target} Values', fontsize=12)
    # plt.title(title, fontsize=14)

    # Annotate metrics
    # plt.text(0.05, 0.95,
    #          f'$R^2$: {r2:.2f}\nMAE: {mae:.2f}',
    #          transform=plt.gca().transAxes,
    #          fontsize=10,
    #          verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Final touches
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save and show
    plt.savefig(OUTPUT_DIR / 'pred_vs_true.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def train_test_split_regression(X, y, test_size=0.2, b='auto', random_state=42):
    # print(f'y = {y}')
    if isinstance(b, str):
        bins = np.histogram_bin_edges(y, bins=b)
        # remove the last index (end point)
        bins = bins[:-1]
    elif isinstance(b, int):
        bins = np.linspace(min(y), max(y), num=b, endpoint=False)
    else:
        raise Exception(f'Undefined bins {b}')

    # print(f'Bins: {bins}')
    groups = np.digitize(y, bins)
    # print(f'Group: {groups}')
    return train_test_split(X, y, test_size=test_size, stratify=groups, random_state=random_state)

random_seed = 42

# Set seeds
np.random.seed(random_seed)
random.seed(random_seed)

warnings.filterwarnings("ignore")

ensure_output_directories(OUTPUT_DIR)
df_meta = read_meta_dataset(META_DATASET_PATH)
df_meta = df_meta.drop(columns=["Seed", "Dataset", "Sample Size", "Model"])
target_column = 'MCC'

# now for the filtered version
df_meta = df_meta[df_meta['MCC'] > 0]

# # Get columns names with NaN values
# cols_with_nan = df_meta.columns[df_meta.isnull().any()].tolist()
# print(f"Columns with NaN values: {cols_with_nan}")
# # Remove columns with NaN values
# df_meta = df_meta.drop(columns=cols_with_nan)

# Defining the regression score
def smape_score(true, pred):
    return np.mean(np.abs(pred - true) / ((np.abs(true) + np.abs(pred)) / 2))

# split into train and test sets
X = df_meta.iloc[:, :-1]  # Features
y = df_meta.iloc[:, -1]  # Target variable
# Apply the function
X_train, X_test, y_train, y_test = train_test_split_regression(X, y, test_size=0.2, b='auto')
# Recombine into train/test DataFrames
df_train = pd.DataFrame(X_train, columns=X.columns)
df_train[target_column] = y_train
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test[target_column] = y_test

# /// Model test begin \\\ #

# --- Symbolic regression expression ---

import numpy as np


def predict(
    df,
    C_4_44=4.44,
    C_20_65=20.65,
    C_0_29=0.29,
    C_0_31=0.31,
    C_16_53=16.53,
    C_0_62=0.62,
    C_3_92=3.92,
    C_48_25=48.25,
    C_49_93=49.93,
    C_37_97=37.97,
    C_1_75=1.75,
    C_0_98=0.98,
    C_31_51=31.51,
    C_33_56=33.56,
    C_12_03=12.03,
    C_9_23=9.23,
    C_39_37=39.37,
    C_33_34=33.34,
    C_77_11=77.11
):
    # Make a copy so the original dataframe is not modified
    df = df.copy()

    # parsing the features names to x0, x1, x2, ...
    original_cols = list(df.columns)
    for i, col in enumerate(original_cols):
        df = df.rename(columns={col: f'x{i}'})
        print(col, '->', f'x{i}')

    X0  = df['x0']
    X1  = df['x1']
    X2  = df['x2']
    X3  = df['x3']
    X4  = df['x4']
    X5  = df['x5']
    X6  = df['x6']
    X7  = df['x7']
    X8  = df['x8']
    X9  = df['x9']
    X10 = df['x10']
    X11 = df['x11']
    X12 = df['x12']
    X13 = df['x13']
    X14 = df['x14']
    X15 = df['x15']
    X16 = df['x16']
    X17 = df['x17']
    X18 = df['x18']

    # -------------------------
    # T1
    # -------------------------
    T1 = (
        X7
        + (
            X6 / (
                X14
                + X4 * X18
                * (np.sqrt(np.exp(np.exp(X15))) - X6 + X13)
                * np.log(C_4_44)
            )
        )
        + np.sqrt(X11 / X3)
    )

    # -------------------------
    # T2
    # -------------------------
    T2 = (
        X11 / (
            X4
            + np.exp(X9)
            * (
                -C_20_65
                + X4 * np.exp(np.sqrt(X3))
                * (
                    np.exp(X5 / C_0_29) ** 3
                    / (X2 ** 2 - C_0_31)
                )
            )
            * (C_16_53 / (X3 * X8))
        )
        + X11 / (
            X18 * np.log(X3)
            + 2 * (X13 + X14)
        )
    )

    # -------------------------
    # T3
    # -------------------------
    T3 = (
        X11 / (
            X6
            + (
                X1 / X10
                + X18
                * (
                    X16
                    + X18 ** 3
                    * np.exp(np.sqrt(X3))
                    * (C_0_62 / np.exp(np.exp(X18)))
                )
                * (X2 / (np.sqrt(X1) ** 3))
                * (X18 ** 3 / X1)
            ) / X13
        )
        + X1 / (X12 + X10)
        + X15
        * (
            (
                np.sqrt(X2)
                / (
                    (
                        np.sqrt(X3)
                        / (
                            (
                                X0 ** 6
                                + (np.sqrt(np.log(X1) ** 9)) ** 3
                            )
                            / (
                                C_0_29
                                + (
                                    (X4 * np.exp(X18))
                                    / (X2 * X11)
                                ) ** 162
                            )
                            + X17
                        )
                    )
                    / (
                        X16 ** 2
                        + (X0 * X1) ** 3
                        * X17
                        * np.log(X2) ** 6
                        + X16
                    )
                )
            ) ** (1 / 8)
        )
    )

    # -------------------------
    # T4
    # -------------------------
    T4 = (
        np.sqrt(X9 / X11)
        + (
            X11 / (
                np.exp(
                    X5
                    / (
                        (X6 - C_3_92) * np.exp(X15)
                        + X1
                    )
                )
                + X1 / X8
            )
            + C_48_25 * X15
            + X10
        )
        / (
            X4
            + C_49_93 * (X3 / X2)
        )
        + C_3_92 / (np.exp(X6) - X5)
    )

    # -------------------------
    # T5
    # -------------------------
    T5 = (
        X1 / (
            np.exp(
                3 * X5 / (
                    X11 / (
                        np.exp(
                            X11 / (X4 - C_37_97)
                        )
                        + X7
                    )
                    + X9 * X15
                    + X18
                )
            )
            + C_1_75
            * (
                X3
                / (
                    np.exp(3 * X0 ** 3)
                    + np.exp(X5)
                )
            )
        )
        + X1 / (
            np.exp(3 * np.sqrt(X10 / X3))
            + np.exp(3 * X0 ** 3) * X18
        )
        + X7
        + X1 / (
            X5
            + X14
            + C_0_98
            * (
                X11
                + (
                    X12 ** 2 * np.exp(X12)
                ) / (X11 ** 2)
            )
            * (X11 / X8)
        )
        - X4 / C_31_51
    )

    # -------------------------
    # T6
    # -------------------------
    T6 = (
        X11 / (
            X10
            + (
                X2 / X10
                + C_33_56
                * (X2 + np.exp(X12))
                * (C_12_03 / X8)
            )
            / (
                (
                    X10 ** 2
                    * X14
                    * np.exp(X18 ** 3)
                ) / C_37_97
                + X6 ** 3
            )
        )
        + C_77_11 / (
            X4
            + (
                -C_9_23 / X10
                + X6
                * (
                    np.exp(np.exp(X0))
                    + np.log(X2)
                    * np.exp(np.sqrt(X3))
                    * (X2 / X8)
                )
                * ((X2 - C_9_23) / X8)
            )
            / (X3 - X13)
        )
        + (X9 - X4 - C_39_37)
        / (
            X4
            + X17 * (
                (X11 - X5) ** 2 / X0
            )
        )
        + C_33_34
    )

    F = C_33_56 / (T1 + T2 + T3 + T4 + T5 + T6)

    return F

    # original full equation:
    # return div(33.5638701915741, add(X7, add(div(X6, add(X14, mul(X4, mul(X18, mul(add(sub(sqrt(exp(exp(X15))), X6), X13), log(4.443030977881925)))))), add(sqrt(div(X11, X3)), add(div(X11, add(X4, mul(mul(exp(X9), add(-20.653032127428073, mul(mul(X4, exp(sqrt(X3))), div(sq3(exp(identity(div(X5, identity(0.2931779161181822))))), sub(sq2(X2), 0.3120032833018058))))), div(identity(div(16.527167914853564, X3)), X8)))), add(div(X11, add(add(mul(X18, log(X3)), add(X13, X14)), add(X13, X14))), add(div(X1, add(X12, X10)), add(add(div(X11, add(X6, div(add(div(X1, X10), mul(mul(mul(X18, add(X16, mul(mul(sq3(X18), exp(sqrt(X3))), div(0.6167516132111217, exp(exp(X18)))))), div(X2, sq3(sqrt(X1)))), div(sq3(X18), X1))), X13))), mul(sqrt(sqrt(sqrt(div(sqrt(X2), div(div(sqrt(X3), add(div(add(sq3(sq2(X0)), sq3(sqrt(sq3(sq3(log(X1)))))), add(0.2931779161181822, sq3(sq3(sq3(sq3(div(sq2(mul(div(X4, X2), exp(X18))), sq2(X11)))))))), X17)), add(add(sq2(X16), mul(mul(sq3(mul(X0, X1)), X17), sq2(sq3(log(X2))))), X16)))))), X15)), add(sqrt(div(X9, X11)), add(div(add(add(div(X11, add(exp(identity(div(X5, add(mul(sub(X6, 3.922823513181264), exp(X15)), X1)))), div(X1, X8))), mul(48.251286620685036, X15)), X10), add(X4, mul(49.92924580592823, div(X3, X2)))), add(div(3.922823513181264, sub(exp(X6), X5)), add(div(X1, add(sq3(exp(identity(div(X5, add(add(div(X11, add(exp(identity(div(X11, sub(X4, 37.97287145620216)))), X7)), mul(X9, X15)), X18))))), mul(1.7488105760395811, div(X3, add(sq3(exp(sq3(X0))), exp(X5)))))), add(div(X1, add(sq3(exp(identity(div(sqrt(X10), sqrt(X3))))), mul(sq3(exp(sq3(X0))), X18))), add(X7, add(div(X1, add(add(X5, X14), mul(mul(0.9773512415938796, add(X11, mul(mul(X12, exp(X12)), div(X12, sq2(X11))))), div(X11, X8)))), add(div(X4, -31.513873970639562), add(div(X11, add(X10, div(add(div(X2, X10), mul(mul(33.5638701915741, add(X2, exp(X12))), div(12.032890547823584, X8))), add(div(mul(sq2(X10), X14), div(37.97287145620216, exp(sq3(X18)))), sq3(X6))))), add(div(add(33.5638701915741, 43.55355368057943), add(X4, div(add(div(-9.22508859226695, X10), mul(mul(X6, add(exp(exp(X0)), mul(mul(log(X2), exp(sqrt(X3))), div(X2, X8)))), div(add(X2, -9.22508859226695), X8))), sub(X3, X13)))), add(div(add(-39.368376250880644, sub(X9, X4)), add(X4, mul(X17, div(identity(sq2(sub(X11, X5))), X0)))), 33.33746993153734)))))))))))))))))))



# --- Rename columns to match expression variable names ---
# Converts 'nr_attr' -> 'nrattr', etc.
df_train.columns = [col.lower().replace('_', '') for col in df_train.columns]
df_test.columns = [col.lower().replace('_', '') for col in df_test.columns]

# inferencing the train dataset
y_pred_train = predict(df_train)



# \\\ Model test end/// #

y_train = df_train.iloc[:, -1].values  # Target variable

# Evaluating training set
train_r2 = r2_score(y_train, y_pred_train)
train_mape = smape_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)

n = len(y_train)  # Total samples
k = df_train.shape[1] - 1  # Number of predictors
train_adj_r2 = 1 - (1 - train_r2) * ((n - 1) / (n - k - 1))

# inferencing the test dataset
y_pred_test = predict(df_test)

y_test = df_test.iloc[:, -1].values  # Target variable
test_r2 = r2_score(y_test, y_pred_test)
test_mape = smape_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

n = len(y_test)
test_adj_r2 = 1 - (1 - test_r2) * ((n - 1) / (n - k - 1))

# Logging results
print(f"Final results:")
print(f"Train dataset ({len(y_train)} rows): R^2: {round(train_r2, 3)}, Adjusted R^2: {round(train_adj_r2, 3)}, sMAPE: {round(train_mape, 3)}, MAE: {round(train_mae, 3)}")
print(f"Test dataset ({len(y_test)} rows): R^2: {round(test_r2, 3)}, Adjusted R^2: {round(test_adj_r2, 3)}, sMAPE: {round(test_mape, 3)}, MAE: {round(test_mae, 3)}")

# plot_regression_results(y_train, y_pred_train, 'Regression Predictions vs. True Values (Train Set)', 'MCC', 'train')
plot_regression_results(y_test, y_pred_test, 'Regression Predictions vs. True Values (Test Set)', 'MCC', 'test')

# Train dataset (20668 rows): R^2: 0.378, Adjusted R^2: 0.377, sMAPE: 0.512, MAE: 0.206
# Test dataset (5167 rows): R^2: 0.361, Adjusted R^2: 0.36, sMAPE: 0.515, MAE: 0.209

# exporting a csv with the real and predicted values
df_test['y_pred'] = y_pred_test
df_test['y_true'] = y_test

#remove the columns that are not y_pred or y_true
df_test = df_test[['y_true', 'y_pred']]

df_test.to_csv(OUTPUT_DIR / 'regression_results_test.csv', index=False)

# doing the same with train
df_train['y_pred'] = y_pred_train
df_train['y_true'] = y_train
#remove the columns that are not y_pred or y_true
df_train = df_train[['y_true', 'y_pred']]
df_train.to_csv(OUTPUT_DIR / 'regression_results_train.csv', index=False)
