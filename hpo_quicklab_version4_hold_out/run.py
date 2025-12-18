# run.py
from __future__ import annotations

import argparse
from typing import Dict

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from config import ExperimentConfig
from data_adult import load_adult
from objective_gbdt import GBDTObjective
from searchers import ParamSpace, grid_search, random_search, bayes_opt_gp_ei, genetic_algorithm

def evaluate_on_test(best_params: Dict, X_train, y_train, X_test, y_test, seed: int) -> float:
    """
    Retrain the best-found config on FULL TRAIN and evaluate on held-out TEST.
    Note: HistGradientBoostingClassifier uses max_iter, we store it as n_estimators.
    """
    model = HistGradientBoostingClassifier(
        max_depth=int(best_params["max_depth"]),
        max_leaf_nodes=int(best_params["max_leaf_nodes"]),
        learning_rate=float(best_params["learning_rate"]),
        max_bins=int(best_params["max_bins"]),
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        l2_regularization=float(best_params["l2_regularization"]),
        early_stopping=bool(best_params["early_stopping"]),
        validation_fraction=float(best_params["validation_fraction"]),
        max_iter=int(best_params["n_estimators"]),
        random_state=seed,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return float(accuracy_score(y_test, y_pred))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600, help="Time budget per algorithm in seconds.")
    ap.add_argument("--print-every", type=int, default=10, help="Print best-so-far every N seconds.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--algos", type=str, default="grid,random,bayes,genetic",
                    help="Comma-separated: grid,random,bayes,genetic")
    ap.add_argument("--cv", type=int, default=3, help="CV folds used inside HPO on TRAIN only.")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--data-home", type=str, default=None, help="Optional OpenML cache directory.")
    args = ap.parse_args()

    cfg = ExperimentConfig(
        seed=args.seed,
        time_budget_s=args.budget,
        print_every_s=args.print_every,
        cv_folds=args.cv,
        test_size=args.test_size,
    )

    # Load data (train/test split)
    X_train, X_test, y_train, y_test = load_adult(test_size=cfg.test_size, seed=cfg.seed, data_home=args.data_home)

    # Objective uses TRAIN ONLY (CV inside)
    objective = GBDTObjective(X_train, y_train, cv_folds=cfg.cv_folds, seed=cfg.seed)

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

    print("\n=== SUMMARY (TRAIN-CV best within budget) ===")
    for name, best in results.items():
        print(f"{name:>8}: best_cv={best.score:.5f}  evals={best.evals}  params={best.params}")

    print("\n=== TEST SET PERFORMANCE (retrain on full TRAIN, evaluate once) ===")
    for name, best in results.items():
        test_acc = evaluate_on_test(best.params, X_train, y_train, X_test, y_test, seed=cfg.seed)
        print(f"{name:>8}: test_acc={test_acc:.5f}")

if __name__ == "__main__":
    main()
