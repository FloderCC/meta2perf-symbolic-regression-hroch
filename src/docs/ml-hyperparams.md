# Hyperparameter space explored for the ML models

| Model | Hyperparameters (explored values) |
|-------|-----------------------------------|
| **LR** | **solver**: {saga} <br> **C**: {0.1, 1, 10} <br> **penalty**: {l1, l2, None} <br> **class_weight**: {None, balanced} |
| **Ridge** | **alpha**: {0.1, 1, 10} <br> **class_weight**: {None, balanced} |
| **SGD** | **loss**: {hinge, log_loss, squared_hinge} <br> **alpha**: {0.0001, 0.001, 0.01, 0.1} <br> **class_weight**: {None, balanced} |
| **Perceptron** | **penalty**: {None, l1, l2, elasticnet} <br> **alpha**: {0.0001, 0.001, 0.01, 0.1} <br> **class_weight**: {None, balanced} |
| **LinearSVC** | **C**: {0.1, 1, 10} <br> **penalty**: {l2} <br> **class_weight**: {None, balanced} |
| **DT** | **criterion**: {gini, entropy, log_loss} <br> **max_depth**: {10, 20, 40, 80, 160, 320} <br> **max_features**: {None, sqrt, log2} <br> **class_weight**: {None, balanced} |
| **ET** | **criterion**: {gini, entropy, log_loss} <br> **max_features**: {None, sqrt, log2} <br> **max_depth**: {10, 20, 40, 80, 160, 320} <br> **class_weight**: {None, balanced} |
| **RF** | **criterion**: {gini, entropy, log_loss} <br> **max_features**: {None, sqrt, log2} <br> **class_weight**: {None, balanced} <br> **n_estimators**: {25, 50, 100, 200} <br> **max_depth**: {10, 20, 40, 80, 160, 320} |
| **ETs** | **criterion**: {gini, entropy, log_loss} <br> **max_features**: {None, sqrt, log2} <br> **class_weight**: {None, balanced} <br> **n_estimators**: {25, 50, 100, 200} <br> **max_depth**: {10, 20, 40, 80, 160, 320} |
| **AB** | **n_estimators**: {25, 50, 100, 200} <br> **learning_rate**: {0.1, 0.5, 1} |
| **HGB** | **max_iter**: {50, 100, 200} <br> **learning_rate**: {0.1, 0.5, 1} <br> **max_depth**: {10, 20, 40, 80, 160, 320} <br> **l2_regularization**: {0.0, 0.1, 1.0} |
| **Bagging** | **n_estimators**: {10, 50, 100} <br> **max_samples**: {1.0} |
| **GaussianNB** | **var_smoothing**: {1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1} |
| **BernoulliNB** | **alpha**: {0.1, 1, 10} <br> **fit_prior**: {True} |
| **MLP** | **hidden_layer_sizes**: {(50,), (100,)} <br> **activation**: {logistic, relu} <br> **alpha**: {0.0001, 0.001, 0.01} |
| **DNN** | **hidden_layer_sizes**: {(16,16,16), (8,16,8), (32,16,8)} <br> **activation**: {logistic, tanh, relu} <br> **solver**: {adam, sgd} <br> **alpha**: {0.0001, 0.001, 0.01} <br> **batch_size**: {10, 50} <br> **max_iter**: {100, 500} |
