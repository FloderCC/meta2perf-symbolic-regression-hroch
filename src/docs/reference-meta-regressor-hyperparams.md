# Hyperparameter search space explored for the reference meta-regressors

| Regressor | Hyperparameters (explored values) |
|-----------|-----------------------------------|
| **Linear Regression** | No tunable hyperparameters (default configuration). |
| **Decision Tree (DT)** | **max_depth**: {8, 16, 64, 128} <br> **min_samples_split**: {5, 10} <br> **min_samples_leaf**: {1, 2, 5} <br> **max_features**: {`sqrt`, `log2`} |
| **Random Forest (RF)** | **n_estimators**: {50, 100} <br> **max_depth**: {8, 16, 64, 128} <br> **min_samples_split**: {2, 5} <br> **min_samples_leaf**: {1, 2} <br> **max_features**: {`sqrt`, `log2`, 0.5} <br> **bootstrap**: {True, False} |
| **Gradient Boosting (GB)** | **n_estimators**: {100, 300} <br> **learning_rate**: {0.03, 0.1} <br> **max_depth**: {3, 5} <br> **subsample**: {1.0, 0.8} |
| **MLP Regressor** | **hidden_layer_sizes**: {(50,), (100,), (200,)} <br> **activation**: {relu, tanh} <br> **alpha**: {0.0001, 0.001} <br> **max_iter**: 1000 (fixed) |

The same grids are used for the joint-descriptor analysis and the marginal
dataset-only baseline. The withdrawn min–max experiments are not part of this
publication repository.
