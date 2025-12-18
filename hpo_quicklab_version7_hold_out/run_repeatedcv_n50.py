#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from config import ExperimentConfig
from data_adult_utils import load_adult_full, stratified_subsample, train_test_split_stratified
from objective_gbdt_repeatedcv import GBDTObjectiveRepeatedCV
from searchers import ParamSpace, random_search


def evaluate_on_test(best_params: Dict, X_train, y_train, X_test, y_test, seed: int) -> float:
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


def mean_std_sem(xs: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(xs, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sem = float(std / math.sqrt(len(arr))) if len(arr) > 0 else float("nan")
    return mean, std, sem


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_hpo_with_countdown(cfg: ExperimentConfig, objective, space: ParamSpace, countdown_every: int) -> Dict:
    """
    Runs random search in a worker thread and prints a countdown every N seconds
    based on the per-run time budget.
    Returns the BestSoFar object fields as a dict.
    """
    holder = {}

    def worker():
        best = random_search(cfg, objective, space)
        holder["best"] = best

    t0 = time.time()
    thr = threading.Thread(target=worker, daemon=True)
    thr.start()

    while thr.is_alive():
        thr.join(timeout=countdown_every)
        elapsed = time.time() - t0
        remaining = max(0.0, cfg.time_budget_s - elapsed)
        rem_m, rem_s = divmod(int(remaining), 60)
        el_m, el_s = divmod(int(elapsed), 60)
        print(f"[countdown] elapsed {el_m:02d}:{el_s:02d} | remaining budget ~ {rem_m:02d}:{rem_s:02d}")

    best = holder["best"]
    return {"score": float(best.score), "params": best.params, "evals": int(best.evals)}


def main():
    ap = argparse.ArgumentParser(description="n=50 experiment: repeated k-fold CV inside HPO objective.")
    ap.add_argument("--budget", type=int, default=180, help="Time budget per HPO run (seconds).")
    ap.add_argument("--cv", type=int, default=3, help="k folds for CV inside HPO (train only).")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--n-total", type=int, default=50, help="Total subsample size (fixed at 50 for this experiment).")
    ap.add_argument("--repetitions", type=str, default="1,3,5,10", help="Comma-separated repetition counts R.")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated outer seeds.")
    ap.add_argument("--countdown-every", type=int, default=30)
    ap.add_argument("--data-home", type=str, default=None, help="Optional OpenML cache directory.")
    ap.add_argument("--out-csv", type=str, default="n50_repeatedcv_results.csv")
    args = ap.parse_args()

    reps = parse_int_list(args.repetitions)
    seeds = parse_int_list(args.seeds)

    print("Loading Adult dataset once (same preprocessing pipeline)...")
    X_full, y_full = load_adult_full(data_home=args.data_home)
    print(f"Full dataset: n={len(y_full)} | pos_rate={float(np.mean(y_full)):.3f} | X_shape={X_full.shape}")

    total_runs = len(reps) * len(seeds)
    finished_runs = 0
    durations: List[float] = []

    rows = []

    for R in reps:
        test_scores: List[float] = []
        print(f"\n==============================")
        print(f"Repetitions R = {R}")
        print(f"==============================")

        for seed in seeds:
            t_run = time.time()

            # fixed tiny dataset n=50
            X_sub, y_sub = stratified_subsample(X_full, y_full, n_total=args.n_total, seed=seed)
            X_train, X_test, y_train, y_test = train_test_split_stratified(
                X_sub, y_sub, test_size=args.test_size, seed=seed
            )

            cfg = ExperimentConfig(
                seed=seed,
                time_budget_s=args.budget,
                print_every_s=max(1, args.countdown_every),  # searcher prints every N seconds too
                cv_folds=args.cv,
                test_size=args.test_size,
            )

            objective = GBDTObjectiveRepeatedCV(
                X_train, y_train,
                cv_folds=cfg.cv_folds,
                seed=cfg.seed,
                repetitions=R,
            )
            space = ParamSpace(cfg)

            print(f"\n--- Run {finished_runs+1}/{total_runs} | n={len(y_sub)} | seed={seed} | R={R} ---")
            best = run_hpo_with_countdown(cfg, objective, space, countdown_every=args.countdown_every)

            test_acc = evaluate_on_test(best["params"], X_train, y_train, X_test, y_test, seed=seed)
            test_scores.append(test_acc)

            dur = time.time() - t_run
            durations.append(dur)
            finished_runs += 1

            avg = float(np.mean(durations))
            remaining = total_runs - finished_runs
            eta_s = remaining * avg
            eta_m, eta_sec = divmod(int(eta_s), 60)

            print(f"[done] seed={seed} R={R} | best_cv={best['score']:.5f} evals={best['evals']} | test_acc={test_acc:.5f} | run_time={dur:.1f}s")
            print(f"[ETA] finished {finished_runs}/{total_runs} | est remaining ~ {eta_m:02d}:{eta_sec:02d}")

        m, s, se = mean_std_sem(test_scores)
        rows.append({
            "n_total": int(args.n_total),
            "test_size": float(args.test_size),
            "cv_folds": int(args.cv),
            "repetitions_R": int(R),
            "n_repeats_outer_seed": int(len(seeds)),
            "test_acc_mean": m,
            "test_acc_std": s,
            "test_acc_sem": se,
        })

        print("\n[aggregate]")
        print(f"R={R} | test_acc mean={m:.5f} std={s:.5f} sem={se:.5f}")

    # print summary
    print("\n==============================")
    print("FINAL SUMMARY (TEST performance after HPO)")
    print("==============================")
    header = f"{'R':>4}  {'mean':>8}  {'std':>8}  {'sem':>8}  {'outer_seeds':>11}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['repetitions_R']:>4d}  {r['test_acc_mean']:>8.5f}  {r['test_acc_std']:>8.5f}  {r['test_acc_sem']:>8.5f}  {r['n_repeats_outer_seed']:>11d}")

    # save CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote results to: {args.out_csv}")


if __name__ == "__main__":
    main()
