# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42

    # time budget per algorithm (seconds)
    time_budget_s: int = 600
    print_every_s: int = 10

    # CV folds used inside HPO (on TRAIN ONLY)
    cv_folds: int = 3

    # Adult dataset split
    test_size: float = 0.25

    # NOTE: bounds are used mainly for Bayes normalization
    # HistGradientBoosting hyperparameter ranges
    max_depth_min: int = 3
    max_depth_max: int = 12

    max_leaf_nodes_min: int = 15
    max_leaf_nodes_max: int = 255

    # log10(learning_rate) range: 1e-3 to ~3e-1
    log10_lr_min: float = -3.0
    log10_lr_max: float = -0.5

    max_bins_min: int = 32
    max_bins_max: int = 255

    min_samples_leaf_min: int = 5
    min_samples_leaf_max: int = 200

    # log10(l2_regularization) range: 1e-6 to 10
    log10_l2_min: float = -6.0
    log10_l2_max: float = 1.0

    validation_fraction_min: float = 0.05
    validation_fraction_max: float = 0.30

    n_estimators_min: int = 100
    n_estimators_max: int = 1000
