#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

from config import ExperimentConfig
from objective_gbdt import GBDTObjective
from searchers import ParamSpace, random_search, grid_search


# -----------------------------
# Data: same pipeline as data_adult.py, but return full X,y (no split)
# -----------------------------
def load_adult_full(seed: int = 42, data_home: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    data = fetch_openml("adult", version=2, as_frame=True, data_home=data_home)

    X_df = data.data
    y_raw = data.target
    y = (y_raw == ">50K").astype(int).values

    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns
    num_cols = X_df.select_dtypes(exclude=["category", "object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X = preprocessor.fit_transform(X_df)
    return np.asarray(X), np.asarray(y)


def stratified_subsample(X: np.ndarray, y: np.ndarray, n_total: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to size n_total (or full if n_total >= len(y))."""
    n_total = int(n_total)
    if n_total >= len(y):
        return X, y

    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    p1 = float(np.mean(y))
    n1 = int(round(n_total * p1))
    n0 = n_total - n1

    n0 = min(n0, len(idx0))
    n1 = min(n1, len(idx1))
    # if we clipped one class, fill remainder from the other
    remainder = n_total - (n0 + n1)
    if remainder > 0:
        if len(idx0) - n0 >= remainder:
            n0 += remainder
        else:
            n1 += remainder

    sel0 = rng.choice(idx0, size=n0, replace=False)
    sel1 = rng.choice(idx1, size=n1, replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return X[sel], y[sel]


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


@dataclass
class HPOResult:
    best_params: Dict
    best_cv: float
    evals: int


def run_hpo_with_countdown(
    algo_name: str,
    cfg: ExperimentConfig,
    objective: GBDTObjective,
    space: ParamSpace,
    print_every_s: int = 30,
) -> HPOResult:
    """
    Run an HPO algorithm in a worker thread and print countdown every `print_every_s`.
    Countdown is exact for the per-algorithm time budget, plus a rough total ETA across experiments.
    """
    out_holder = {}

    def worker():
        if algo_name == "random":
            best = random_search(cfg, objective, space)
        elif algo_name == "grid":
            best = grid_search(cfg, objective, space)
        else:
            raise ValueError(f"Unsupported algo: {algo_name}")
        out_holder["best"] = best

    t0 = time.time()
    thr = threading.Thread(target=worker, daemon=True)
    thr.start()

    # periodic countdown
    while thr.is_alive():
        thr.join(timeout=print_every_s)
        elapsed = time.time() - t0
        remaining = max(0.0, cfg.time_budget_s - elapsed)
        # Format as mm:ss
        rem_m = int(remaining) // 60
        rem_s = int(remaining) % 60
        el_m = int(elapsed) // 60
        el_s = int(elapsed) % 60
        print(f"[countdown] HPO '{algo_name}': elapsed {el_m:02d}:{el_s:02d} | remaining budget ~ {rem_m:02d}:{rem_s:02d}")

    best = out_holder["best"]
    return HPOResult(best_params=best.params, best_cv=float(best.score), evals=int(best.evals))


def mean_std_sem(xs: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(xs, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sem = float(std / math.sqrt(len(arr))) if len(arr) > 0 else float("nan")
    return mean, std, sem


def parse_int_list(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_seed_list(s: str) -> List[int]:
    return parse_int_list(s)


def main():
    ap = argparse.ArgumentParser(description="Observe how sample size impacts TEST performance after HPO (random/grid).")
    ap.add_argument("--algo", type=str, default="random", choices=["random", "grid"])
    ap.add_argument("--budget", type=int, default=180, help="Time budget per HPO run (seconds).")
    ap.add_argument("--cv", type=int, default=3, help="CV folds used inside HPO on TRAIN only.")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--sample-sizes", type=str, default="500,1000,2000,5000,10000,20000,40000",
                    help="Comma-separated total dataset sizes to subsample from Adult.")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                    help="Comma-separated random seeds; repeats experiment per sample size.")
    ap.add_argument("--data-home", type=str, default=None, help="Optional OpenML cache directory.")
    ap.add_argument("--countdown-every", type=int, default=30, help="Print countdown every N seconds.")
    ap.add_argument("--out-csv", type=str, default="sample_size_hpo_results.csv")
    args = ap.parse_args()

    sample_sizes = parse_int_list(args.sample_sizes)
    seeds = parse_seed_list(args.seeds)

    print("Loading Adult dataset once (same preprocessing pipeline)...")
    X_full, y_full = load_adult_full(seed=42, data_home=args.data_home)
    n_full = len(y_full)
    print(f"Full dataset: n={n_full}, positive_rate={float(np.mean(y_full)):.3f}, X_shape={X_full.shape}")

    rows = []
    global_start = time.time()
    total_runs = len(sample_sizes) * len(seeds)
    finished_runs = 0
    durations: List[float] = []

    for n_total in sample_sizes:
        test_scores: List[float] = []

        for seed in seeds:
            run_start = time.time()

            # build a new dataset at this sample size (stratified subsample)
            X_sub, y_sub = stratified_subsample(X_full, y_full, n_total=n_total, seed=seed)

            # train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y_sub, test_size=args.test_size, random_state=seed, stratify=y_sub
            )

            # config for this run
            cfg = ExperimentConfig(
                seed=seed,
                time_budget_s=args.budget,
                print_every_s=args.countdown_every,  # HPO internal logging
                cv_folds=args.cv,
                test_size=args.test_size,
            )

            objective = GBDTObjective(X_train, y_train, cv_folds=cfg.cv_folds, seed=cfg.seed)
            space = ParamSpace(cfg)

            # HPO + countdown prints
            print(f"\n=== Run {finished_runs+1}/{total_runs} | sample_size={len(y_sub)} | seed={seed} | algo={args.algo} ===")
            hpo_res = run_hpo_with_countdown(
                algo_name=args.algo,
                cfg=cfg,
                objective=objective,
                space=space,
                print_every_s=args.countdown_every,
            )

            test_acc = evaluate_on_test(hpo_res.best_params, X_train, y_train, X_test, y_test, seed=seed)
            test_scores.append(test_acc)

            dur = time.time() - run_start
            durations.append(dur)
            finished_runs += 1

            # global ETA (rough) after each run
            avg = float(np.mean(durations))
            remaining_runs = total_runs - finished_runs
            eta_s = remaining_runs * avg
            eta_m = int(eta_s) // 60
            eta_sec = int(eta_s) % 60
            print(f"[done] sample_size={len(y_sub)} seed={seed} | best_cv={hpo_res.best_cv:.5f} | test_acc={test_acc:.5f} | run_time={dur:.1f}s")
            print(f"[ETA] finished {finished_runs}/{total_runs} | est remaining ~ {eta_m:02d}:{eta_sec:02d}")

        m, s, se = mean_std_sem(test_scores)
        rows.append({
            "sample_size_total": int(min(n_total, n_full)),
            "n_train": int(round((1 - args.test_size) * min(n_total, n_full))),
            "n_test": int(round(args.test_size * min(n_total, n_full))),
            "n_repeats": int(len(seeds)),
            "test_acc_mean": m,
            "test_acc_std": s,
            "test_acc_sem": se,
        })

        print("\n--- Aggregate for this sample size ---")
        print(f"sample_size={int(min(n_total, n_full))} | test_acc mean={m:.5f} std={s:.5f} sem={se:.5f}")

    # print final table
    print("\n==============================")
    print("FINAL SUMMARY (TEST performance after HPO)")
    print("==============================")
    header = f"{'sample_size':>10}  {'n_train':>8}  {'n_test':>7}  {'repeats':>7}  {'mean':>8}  {'std':>8}  {'sem':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['sample_size_total']:>10d}  {r['n_train']:>8d}  {r['n_test']:>7d}  {r['n_repeats']:>7d}  "
              f"{r['test_acc_mean']:>8.5f}  {r['test_acc_std']:>8.5f}  {r['test_acc_sem']:>8.5f}")

    # save CSV
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total_elapsed = time.time() - global_start
    print(f"\nWrote results to: {args.out_csv}")
    print(f"Total elapsed: {total_elapsed/60.0:.1f} min")


if __name__ == "__main__":
    main()
