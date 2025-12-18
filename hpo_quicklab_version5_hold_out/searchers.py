# searchers.py
from __future__ import annotations

import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

@dataclass
class BestSoFar:
    score: float = float("-inf")
    params: Optional[Dict] = None
    evals: int = 0

class ParamSpace:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample_random(self, rng: random.Random) -> Dict:
        opt = rng.choice(["adam", "sgd"])
        p = {
            "batch_size": rng.choice([32, 64, 96, 128, 192, 256]),
            "lr": 10 ** rng.uniform(self.cfg.log10_lr_min, self.cfg.log10_lr_max),
            "weight_decay": 10 ** rng.uniform(self.cfg.log10_wd_min, self.cfg.log10_wd_max),
            "optimizer": opt,
            "momentum": rng.uniform(self.cfg.momentum_min, self.cfg.momentum_max) if opt == "sgd" else 0.0,
            "h1": rng.randint(self.cfg.hidden_min, self.cfg.hidden_max),
            "h2": rng.randint(self.cfg.hidden_min, self.cfg.hidden_max),
            "dropout": rng.uniform(self.cfg.dropout_min, self.cfg.dropout_max),
            "max_epochs": rng.randint(self.cfg.max_epochs_min, self.cfg.max_epochs_max),
            "patience": rng.randint(self.cfg.patience_min, self.cfg.patience_max),
        }
        return p

    def grid(self) -> List[Dict]:
        """
        Very coarse grid baseline (intentionally limited).
        Still demonstrates grid weakness as dimensionality grows.
        """
        batch_vals = [64, 128]
        lr_vals = [1e-4, 3e-4, 1e-3]
        wd_vals = [0.0, 1e-5, 1e-4]
        opt_vals = ["adam", "sgd"]
        mom_vals = [0.0, 0.9]   # only meaningful for sgd
        h_vals = [128, 256, 512]
        drop_vals = [0.0, 0.3, 0.5]
        max_epochs_vals = [5, 10, 15]
        patience_vals = [2, 4]

        out: List[Dict] = []
        for bs in batch_vals:
            for lr in lr_vals:
                for wd in wd_vals:
                    for opt in opt_vals:
                        for mom in mom_vals:
                            for h1 in h_vals:
                                for h2 in h_vals:
                                    for dr in drop_vals:
                                        for me in max_epochs_vals:
                                            for pat in patience_vals:
                                                out.append({
                                                    "batch_size": int(bs),
                                                    "lr": float(lr),
                                                    "weight_decay": float(wd),
                                                    "optimizer": str(opt),
                                                    "momentum": float(mom) if opt == "sgd" else 0.0,
                                                    "h1": int(h1),
                                                    "h2": int(h2),
                                                    "dropout": float(dr),
                                                    "max_epochs": int(me),
                                                    "patience": int(pat),
                                                })
        return out

def run_with_time_budget(name: str, cfg, objective, propose_fn):
    start = time.time()
    last_print = start
    best = BestSoFar()

    while True:
        now = time.time()
        if now - start >= cfg.time_budget_s:
            break

        params = propose_fn(best)
        res = objective.evaluate(params)
        best.evals += 1

        if res.score > best.score:
            best.score = res.score
            best.params = params

        if now - last_print >= cfg.print_every_s:
            elapsed = int(now - start)
            print(f"[{name:>10}] t={elapsed:>4}s  evals={best.evals:>5}  best_val={best.score:.5f}  params={best.params}")
            last_print = now

    elapsed = time.time() - start
    print(f"[{name:>10}] DONE  t={elapsed:.1f}s  evals={best.evals}  best_val={best.score:.5f}  params={best.params}")
    return best

def grid_search(cfg, objective, space: ParamSpace):
    grid = space.grid()
    idx = 0
    def propose(best):
        nonlocal idx
        p = grid[idx % len(grid)]
        idx += 1
        return p
    return run_with_time_budget("grid", cfg, objective, propose)

def random_search(cfg, objective, space: ParamSpace):
    rng = random.Random(cfg.seed)
    def propose(best):
        return space.sample_random(rng)
    return run_with_time_budget("random", cfg, objective, propose)

def bayes_opt_gp_ei(cfg, objective, space: ParamSpace, init_points: int = 12, cand_points: int = 200):
    rng = random.Random(cfg.seed)
    X: List[List[float]] = []
    y: List[float] = []

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=cfg.seed)

    def to_vec(p: Dict) -> List[float]:
        # normalize to [0,1]
        bs = (p["batch_size"] - cfg.batch_min) / (cfg.batch_max - cfg.batch_min)
        lr = (math.log10(p["lr"]) - cfg.log10_lr_min) / (cfg.log10_lr_max - cfg.log10_lr_min)
        wd = (math.log10(max(p["weight_decay"], 1e-12)) - cfg.log10_wd_min) / (cfg.log10_wd_max - cfg.log10_wd_min)
        opt = 1.0 if p["optimizer"] == "adam" else 0.0
        mom = (p["momentum"] - cfg.momentum_min) / (cfg.momentum_max - cfg.momentum_min) if p["optimizer"] == "sgd" else 0.0
        h1 = (p["h1"] - cfg.hidden_min) / (cfg.hidden_max - cfg.hidden_min)
        h2 = (p["h2"] - cfg.hidden_min) / (cfg.hidden_max - cfg.hidden_min)
        dr = (p["dropout"] - cfg.dropout_min) / (cfg.dropout_max - cfg.dropout_min)
        me = (p["max_epochs"] - cfg.max_epochs_min) / (cfg.max_epochs_max - cfg.max_epochs_min)
        pat = (p["patience"] - cfg.patience_min) / (cfg.patience_max - cfg.patience_min)
        return [bs, lr, wd, opt, mom, h1, h2, dr, me, pat]

    start = time.time()
    last_print = start
    best = BestSoFar()

    while True:
        now = time.time()
        if now - start >= cfg.time_budget_s:
            break

        if len(X) < init_points:
            params = space.sample_random(rng)
        else:
            try:
                gp.fit(np.array(X), np.array(y))
            except Exception:
                params = space.sample_random(rng)
            else:
                best_y = max(y) if y else -1e9
                best_ei = -1e9
                best_p = None

                for _ in range(cand_points):
                    p = space.sample_random(rng)
                    xv = np.array([to_vec(p)])
                    mu, std = gp.predict(xv, return_std=True)
                    mu = float(mu[0]); std = float(std[0])
                    std = max(std, 1e-9)

                    xi = 0.01
                    z = (mu - best_y - xi) / std
                    ei = (mu - best_y - xi) * normal_cdf(z) + std * normal_pdf(z)
                    if ei > best_ei:
                        best_ei = ei
                        best_p = p

                params = best_p if best_p is not None else space.sample_random(rng)

        res = objective.evaluate(params)
        best.evals += 1
        if res.score > best.score:
            best.score = res.score
            best.params = params

        X.append(to_vec(params))
        y.append(res.score)

        if now - last_print >= cfg.print_every_s:
            elapsed = int(now - start)
            print(f"[{'bayes':>10}] t={elapsed:>4}s  evals={best.evals:>5}  best_val={best.score:.5f}  params={best.params}")
            last_print = now

    elapsed = time.time() - start
    print(f"[{'bayes':>10}] DONE  t={elapsed:.1f}s  evals={best.evals}  best_val={best.score:.5f}  params={best.params}")
    return best

def genetic_algorithm(cfg, objective, space: ParamSpace, pop_size: int = 18, tournament_k: int = 3):
    rng = random.Random(cfg.seed)

    def mutate(p: Dict, rate: float = 0.25) -> Dict:
        q = dict(p)
        if rng.random() < rate:
            q["batch_size"] = rng.choice([32, 64, 96, 128, 192, 256])
        if rng.random() < rate:
            lr = math.log10(q["lr"]) + rng.uniform(-0.6, 0.6)
            lr = clamp(lr, cfg.log10_lr_min, cfg.log10_lr_max)
            q["lr"] = 10 ** lr
        if rng.random() < rate:
            wd = math.log10(max(q["weight_decay"], 1e-12)) + rng.uniform(-0.8, 0.8)
            wd = clamp(wd, cfg.log10_wd_min, cfg.log10_wd_max)
            q["weight_decay"] = max(0.0, 10 ** wd)
        if rng.random() < rate:
            q["optimizer"] = "adam" if q["optimizer"] == "sgd" else "sgd"
            if q["optimizer"] == "adam":
                q["momentum"] = 0.0
        if rng.random() < rate and q["optimizer"] == "sgd":
            q["momentum"] = float(clamp(q["momentum"] + rng.uniform(-0.2, 0.2), cfg.momentum_min, cfg.momentum_max))
        if rng.random() < rate:
            q["h1"] = int(clamp(q["h1"] + rng.randint(-128, 128), cfg.hidden_min, cfg.hidden_max))
        if rng.random() < rate:
            q["h2"] = int(clamp(q["h2"] + rng.randint(-128, 128), cfg.hidden_min, cfg.hidden_max))
        if rng.random() < rate:
            q["dropout"] = float(clamp(q["dropout"] + rng.uniform(-0.15, 0.15), cfg.dropout_min, cfg.dropout_max))
        if rng.random() < rate:
            q["max_epochs"] = int(clamp(q["max_epochs"] + rng.randint(-3, 3), cfg.max_epochs_min, cfg.max_epochs_max))
        if rng.random() < rate:
            q["patience"] = int(clamp(q["patience"] + rng.randint(-2, 2), cfg.patience_min, cfg.patience_max))
        return q

    def crossover(a: Dict, b: Dict) -> Dict:
        child = {}
        for k in a.keys():
            child[k] = a[k] if rng.random() < 0.5 else b[k]
        # enforce conditional
        if child["optimizer"] == "adam":
            child["momentum"] = 0.0
        return child

    population = [space.sample_random(rng) for _ in range(pop_size)]
    fitness: Dict[Tuple, float] = {}

    def fit(p: Dict) -> float:
        key = (
            int(p["batch_size"]),
            round(float(p["lr"]), 12),
            round(float(p["weight_decay"]), 12),
            str(p["optimizer"]),
            round(float(p.get("momentum", 0.0)), 12),
            int(p["h1"]), int(p["h2"]),
            round(float(p["dropout"]), 6),
            int(p["max_epochs"]), int(p["patience"]),
        )
        if key in fitness:
            return fitness[key]
        s = objective.evaluate(p).score
        fitness[key] = s
        return s

    def tournament_select() -> Dict:
        cand = rng.sample(population, k=min(tournament_k, len(population)))
        return max(cand, key=fit)

    start = time.time()
    last_print = start
    best = BestSoFar()

    # warm eval
    i = 0
    while i < len(population) and (time.time() - start) < cfg.time_budget_s:
        p = population[i]
        s = fit(p)
        best.evals += 1
        if s > best.score:
            best.score, best.params = s, p
        i += 1

    while True:
        now = time.time()
        if now - start >= cfg.time_budget_s:
            break

        p1 = tournament_select()
        p2 = tournament_select()
        child = mutate(crossover(p1, p2))
        s = fit(child)

        best.evals += 1
        if s > best.score:
            best.score, best.params = s, child

        worst_idx = min(range(len(population)), key=lambda j: fit(population[j]))
        population[worst_idx] = child

        if now - last_print >= cfg.print_every_s:
            elapsed = int(now - start)
            print(f"[{'genetic':>10}] t={elapsed:>4}s  evals={best.evals:>5}  best_val={best.score:.5f}  params={best.params}")
            last_print = now

    elapsed = time.time() - start
    print(f"[{'genetic':>10}] DONE  t={elapsed:.1f}s  evals={best.evals}  best_val={best.score:.5f}  params={best.params}")
    return best
