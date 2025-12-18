# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42

    # time budget per algorithm (seconds)
    time_budget_s: int = 600
    print_every_s: int = 10

    # dataset / split
    dataset: str = "fashion_mnist"   # "fashion_mnist" or "cifar10"
    data_dir: str = "./data"
    val_fraction: float = 0.15

    # training basics
    num_workers: int = 2
    pin_memory: bool = True
    max_batches_per_epoch: int | None = None  # set e.g. 200 to speed up; None = full epoch

    # search ranges
    batch_min: int = 32
    batch_max: int = 256

    log10_lr_min: float = -4.0
    log10_lr_max: float = -1.0

    log10_wd_min: float = -7.0
    log10_wd_max: float = -2.0

    hidden_min: int = 64
    hidden_max: int = 1024

    dropout_min: float = 0.0
    dropout_max: float = 0.6

    max_epochs_min: int = 3
    max_epochs_max: int = 20

    patience_min: int = 1
    patience_max: int = 6

    # SGD momentum range (only used if optimizer == "sgd")
    momentum_min: float = 0.0
    momentum_max: float = 0.95
