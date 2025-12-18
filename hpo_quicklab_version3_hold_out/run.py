# run.py
from __future__ import annotations

import argparse
import torch
from torch.utils.data import ConcatDataset

from config import ExperimentConfig
from data_torch import load_datasets, split_train_val
from objective_torch import TorchNNObjective
from searchers import ParamSpace, grid_search, random_search, bayes_opt_gp_ei, genetic_algorithm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "cifar10"])
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--algos", type=str, default="grid,random,bayes,genetic")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-batches-per-epoch", type=int, default=0,
                    help="If >0, cap batches per epoch to speed up HPO. 0 means full epoch.")
    args = ap.parse_args()

    cfg = ExperimentConfig(
        seed=args.seed,
        time_budget_s=args.budget,
        print_every_s=args.print_every,
        dataset=args.dataset,
        data_dir=args.data_dir,
        val_fraction=args.val_frac,
        max_batches_per_epoch=(args.max_batches_per_epoch if args.max_batches_per_epoch > 0 else None),
    )

    # data
    train_full, test_ds, input_shape, num_classes = load_datasets(cfg.dataset, cfg.data_dir, cfg.seed)
    train_ds, val_ds = split_train_val(train_full, cfg.val_fraction, cfg.seed)
    trainval_ds = ConcatDataset([train_ds, val_ds])

    # objective
    objective = TorchNNObjective(cfg, train_ds, val_ds, input_shape, num_classes)
    space = ParamSpace(cfg)

    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    results = {}

    if "grid" in algos:
        results["grid"] = grid_search(cfg, objective, space)
    if "random" in algos:
        results["random"] = random_search(cfg, objective, space)
    if "bayes" in algos:
        results["bayes"] = bayes_opt_gp_ei(cfg, objective, space)
    if "genetic" in algos:
        results["genetic"] = genetic_algorithm(cfg, objective, space)

    print("\n=== SUMMARY (best VAL accuracy within budget) ===")
    for name, best in results.items():
        print(f"{name:>8}: best_val={best.score:.5f}  evals={best.evals}  params={best.params}")

    print("\n=== TEST SET PERFORMANCE (retrain on TRAIN+VAL, evaluate once) ===")
    for name, best in results.items():
        test_acc = objective.retrain_and_test(best.params, trainval_ds, test_ds)
        print(f"{name:>8}: test_acc={test_acc:.5f}")

if __name__ == "__main__":
    main()
