"""Model construction, hyperparameter search, and fitted-model helpers."""

import ast
import copy
import gc
import random
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn_genetic import GASearchCV, ConsecutiveStopping, ThresholdStopping
from sklearn_genetic.callbacks import ProgressBar
from sklearn_genetic.callbacks.base import BaseCallback
from sklearn_genetic.space import Categorical

from src.experiment_config import model_list


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

sklearn_mcc = make_scorer(matthews_corrcoef, greater_is_better=True)

class GarbageCollector(BaseCallback):
    """
    Callback that calls the garbage collector after each generation
    """

    def on_step(self, record=None, logbook=None, estimator=None):
        gc.collect()
        print("Garbage collector called")


def get_number_of_combinations(param_grid: dict) -> int:
    """
    Get the number of combinations for a given param_grid
    :param param_grid: the parameter grid
    :return: the number of combinations
    """
    num_combinations = 1
    for value in param_grid.values():
        num_combinations *= len(value)
    return num_combinations


def __init_models(global_random_seed, x_train, y_train):
    return {
        'LR': LogisticRegression(random_state=global_random_seed),
        'Ridge': RidgeClassifier(random_state=global_random_seed),
        'SGD': SGDClassifier(random_state=global_random_seed),
        'Perceptron': Perceptron(random_state=global_random_seed),
        'DT': DecisionTreeClassifier(random_state=global_random_seed),
        'ExtraTree': ExtraTreeClassifier(random_state=global_random_seed),
        'LinearSVC': LinearSVC(random_state=global_random_seed),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'RF': RandomForestClassifier(random_state=global_random_seed),
        'ExtraTrees': ExtraTreesClassifier(random_state=global_random_seed),
        'AdaBoost': AdaBoostClassifier(random_state=global_random_seed),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=global_random_seed),
        'Bagging': BaggingClassifier(random_state=global_random_seed, estimator=DecisionTreeClassifier(max_depth=160)),
        'MLP': MLPClassifier(random_state=global_random_seed),
        'DNN': MLPClassifier(random_state=global_random_seed),
    }


def create_models(global_random_seed, x_train,
                  y_train) -> list:  #list with returns model_name, model, parameters_desc
    models = __init_models(global_random_seed, x_train, y_train)

    # Define parameter grids for each classifier
    param_grids = {
        'LR': {'solver': ['saga'],
               'C': [0.1, 1, 10],
               'penalty': ['l1', 'l2', None],
               # penalty elasticnet removed because ValueError: l1_ratio must be specified when penalty is elasticnet
               'class_weight': [None, 'balanced']},

        'Ridge': {'alpha': [0.1, 1, 10],
                  'class_weight': [None, 'balanced']},

        'SGD': {'loss': ['hinge', 'log_loss', 'squared_hinge'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'class_weight': [None, 'balanced']},

        'Perceptron': {'penalty': [None, 'l1', 'l2', 'elasticnet'],
                       'alpha': [0.0001, 0.001, 0.01, 0.1],
                       'class_weight': [None, 'balanced']},

        'DT': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [10, 20, 40, 80, 160, 320],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced']},

        'ExtraTree': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': [None, 'sqrt', 'log2'],
            'max_depth': [10, 20, 40, 80, 160, 320],
            'class_weight': [None, 'balanced']},

        'LinearSVC': {'C': [0.1, 1, 10],
                      'penalty': ['l2'],
                      # penalty l1 removed because ValueError: Unsupported set of arguments: The combination of penalty='l1' and loss='squared_hinge' are not supported when dual=True, Parameters: penalty='l1', loss='squared_hinge', dual=True
                      'class_weight': [None, 'balanced']},

        'GaussianNB': {'var_smoothing': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1],
                       'priors': [None]},

        'BernoulliNB': {'alpha': [0.1, 1, 10],
                        'fit_prior': [True]},

        'RF': {'criterion': ['gini', 'entropy', 'log_loss'],
               'max_features': [None, 'sqrt', 'log2'],
               'class_weight': [None, 'balanced'],
               'n_estimators': [25, 50, 100, 200],
               'max_depth': [10, 20, 40, 80, 160, 320]},

        'ExtraTrees': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced'],
            'n_estimators': [25, 50, 100, 200],
            'max_depth': [10, 20, 40, 80, 160, 320]},

        'AdaBoost': {'n_estimators': [25, 50, 100, 200],
                     'learning_rate': [0.1, 0.5, 1]},

        'HistGradientBoosting': {
            'max_iter': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1],
            'max_depth': [10, 20, 40, 80, 160, 320],
            'l2_regularization': [0.0, 0.1, 1.0]
        },

        'Bagging': {'n_estimators': [10, 50, 100],
                    'max_samples': [1.0]},

        'MLP': {'hidden_layer_sizes': [(50,), (100,), ],
                'activation': ['logistic', 'relu'],
                'alpha': [0.0001, 0.001, 0.01]},
        'DNN': {
            'hidden_layer_sizes': [
                (16, 16, 16),
                (8, 16, 8),
                (32, 16, 8)
            ],
            'activation': ['logistic', 'tanh', 'relu'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'batch_size': [10, 50],
            'max_iter': [100, 500],
        },
    }

    # Hyperparameter selection uses four-fold cross-validation on the
    # base-task training partition.
    cv = 4
    population_size = 20
    generations = 10

    # Search the predefined candidate configurations for each classifier.
    model_configurations = []

    for model_name, model in models.items():
        if model_name not in model_list:
            continue

        print(f"Searching for best parameters for {model_name}")

        # getting the parameter grid
        param_grid = param_grids[model_name]
        number_of_combinations = get_number_of_combinations(param_grid)

        # parsing param_grid to the format expected by GASearchCV
        param_grid_o = copy.deepcopy(param_grids[model_name])
        if param_grid == {}:
            model_configurations.append((model_name, model, "unique"))
        else:
            for key, value in param_grid.items():
                param_grid[key] = Categorical(choices=value, random_state=global_random_seed if key != 'hidden_layer_sizes' else None)

            tic = time.time()

            if number_of_combinations > population_size:

                evolved_estimator = GASearchCV(model,
                                               cv=cv,
                                               scoring=sklearn_mcc,
                                               param_grid=param_grid,
                                               population_size=population_size,
                                               generations=generations,
                                               n_jobs=-1,
                                               verbose=False,
                                               )
                evolved_estimator.fit(x_train,
                                      y_train,
                                      callbacks=[ConsecutiveStopping(generations=2, metric='fitness'),
                                                 ProgressBar(),
                                                 GarbageCollector(),
                                                 ThresholdStopping(threshold=1.0, metric='fitness_max')]
                                      )
            else:
                evolved_estimator = GridSearchCV(model,
                                                 cv=cv,
                                                 scoring=sklearn_mcc,
                                                 param_grid=param_grid_o,
                                                 n_jobs=-1,
                                                 verbose=False
                                                 )
                evolved_estimator.fit(x_train,
                                      y_train
                                      )

            toc = time.time()
            print(f"Search elapsed time: {toc - tic} seconds. Model: {model_name}")

            best_params = evolved_estimator.best_params_


            model_configurations.append((model_name, str(best_params)[1:-1]))

            # cleaning up memory
            del evolved_estimator
            gc.collect()

    return model_configurations


def train_best_model(model_name: str, global_random_seed: int, x_train, y_train, config: str):
    """
    Train the best model with the best parameters
    Returns:
        trained model
    """

    # get the base model
    model = __init_models(global_random_seed, x_train, y_train)[model_name]

    if config != 'unique':
        # Configurations are stored as Python dictionary contents, including
        # tuples (MLP/DNN layer sizes), None and boolean values.
        try:
            parameters = ast.literal_eval("{" + config + "}")
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Invalid stored hyperparameters for {model_name}: {config!r}"
            ) from exc
        if not isinstance(parameters, dict):
            raise ValueError(f"Hyperparameters for {model_name} must be a dictionary.")
        # set the parameters of the model from the config
        model.set_params(**parameters)

    model.fit(x_train, y_train)

    return model
