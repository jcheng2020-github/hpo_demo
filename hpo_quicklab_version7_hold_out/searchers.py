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

# ---------- helpers ----------
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

# ---------- search space ----------
class ParamSpace:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample_random(self, rng: random.Random) -> Dict:
        return {
            "max_depth": rng.randint(self.cfg.max_depth_min, self.cfg.max_depth_max),
            "max_leaf_nodes": rng.randint(self.cfg.max_leaf_nodes_min, self.cfg.max_leaf_nodes_max),
            "learning_rate": 10 ** rng.uniform(self.cfg.log10_lr_min, self.cfg.log10_lr_max),
            "max_bins": rng.randint(self.cfg.max_bins_min, self.cfg.max_bins_max),
            "min_samples_leaf": rng.randint(self.cfg.min_samples_leaf_min, self.cfg.min_samples_leaf_max),
            "l2_regularization": 10 ** rng.uniform(self.cfg.log10_l2_min, self.cfg.log10_l2_max),
            "early_stopping": rng.choice([True, False]),
            "validation_fraction": rng.uniform(self.cfg.validation_fraction_min, self.cfg.validation_fraction_max),
            "n_estimators": rng.randint(self.cfg.n_estimators_min, self.cfg.n_estimators_max),
        }

    def grid(self) -> List[Dict]:
        """
        Coarse grid baseline for GBDT.
        Intentionally small-ish but still shows grid weakness under a time budget.
        """
        lr_vals = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        depth_vals = [3, 6, 9, 12]
        leaf_vals = [31, 63, 127, 255]
        bins_vals = [32, 64, 128, 255]
        min_leaf_vals = [5, 20, 80, 200]
        l2_vals = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
        early_vals = [True, False]
        val_frac_vals = [0.1, 0.2]
        n_estimators_vals = [200, 500, 1000]

        out: List[Dict] = []
        for lr in lr_vals:
            for md in depth_vals:
                for mln in leaf_vals:
                    for mb in bins_vals:
                        for msl in min_leaf_vals:
                            for l2 in l2_vals:
                                for es in early_vals:
                                    for vf in val_frac_vals:
                                        for ne in n_estimators_vals:
                                            out.append({
                                                "learning_rate": float(lr),
                                                "max_depth": int(md),
                                                "max_leaf_nodes": int(mln),
                                                "max_bins": int(mb),
                                                "min_samples_leaf": int(msl),
                                                "l2_regularization": float(l2),
                                                "early_stopping": bool(es),
                                                "validation_fraction": float(vf),
                                                "n_estimators": int(ne),
                                            })
        return out

# ---------- shared runner ----------
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
            print(f"[{name:>10}] t={elapsed:>4}s  evals={best.evals:>6}  best_cv={best.score:.5f}  params={best.params}")
            last_print = now

    elapsed = time.time() - start
    print(f"[{name:>10}] DONE  t={elapsed:.1f}s  evals={best.evals}  best_cv={best.score:.5f}  params={best.params}")
    return best

# ---------- algorithms ----------
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

def bayes_opt_gp_ei(cfg, objective, space: ParamSpace, init_points: int = 15, cand_points: int = 250):
    rng = random.Random(cfg.seed)

    X: List[List[float]] = []
    y: List[float] = []

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=cfg.seed)

    def to_vec(p: Dict) -> List[float]:
        # normalize to [0,1] for GP stability
        md = (p["max_depth"] - cfg.max_depth_min) / (cfg.max_depth_max - cfg.max_depth_min)
        mln = (p["max_leaf_nodes"] - cfg.max_leaf_nodes_min) / (cfg.max_leaf_nodes_max - cfg.max_leaf_nodes_min)
        lr = (math.log10(p["learning_rate"]) - cfg.log10_lr_min) / (cfg.log10_lr_max - cfg.log10_lr_min)
        mb = (p["max_bins"] - cfg.max_bins_min) / (cfg.max_bins_max - cfg.max_bins_min)
        msl = (p["min_samples_leaf"] - cfg.min_samples_leaf_min) / (cfg.min_samples_leaf_max - cfg.min_samples_leaf_min)
        l2 = (math.log10(p["l2_regularization"]) - cfg.log10_l2_min) / (cfg.log10_l2_max - cfg.log10_l2_min)
        es = 1.0 if p["early_stopping"] else 0.0
        vf = (p["validation_fraction"] - cfg.validation_fraction_min) / (cfg.validation_fraction_max - cfg.validation_fraction_min)
        ne = (p["n_estimators"] - cfg.n_estimators_min) / (cfg.n_estimators_max - cfg.n_estimators_min)
        return [md, mln, lr, mb, msl, l2, es, vf, ne]

    start = time.time()
    last_print = start
    best = BestSoFar()

    while True:
        now = time.time()
        if now - start >= cfg.time_budget_s:
            break

        # propose
        if len(X) < init_points:
            params = space.sample_random(rng)
        else:
            # fit GP
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
                    mu = float(mu[0])
                    std = float(std[0])
                    std = max(std, 1e-9)

                    xi = 0.01
                    z = (mu - best_y - xi) / std
                    ei = (mu - best_y - xi) * normal_cdf(z) + std * normal_pdf(z)

                    if ei > best_ei:
                        best_ei = ei
                        best_p = p

                params = best_p if best_p is not None else space.sample_random(rng)

        # evaluate
        res = objective.evaluate(params)
        best.evals += 1
        if res.score > best.score:
            best.score = res.score
            best.params = params

        X.append(to_vec(params))
        y.append(res.score)

        if now - last_print >= cfg.print_every_s:
            elapsed = int(now - start)
            print(f"[{'bayes':>10}] t={elapsed:>4}s  evals={best.evals:>6}  best_cv={best.score:.5f}  params={best.params}")
            last_print = now

    elapsed = time.time() - start
    print(f"[{'bayes':>10}] DONE  t={elapsed:.1f}s  evals={best.evals}  best_cv={best.score:.5f}  params={best.params}")
    return best


def genetic_algorithm(cfg, objective, space: ParamSpace, pop_size: int = 20, tournament_k: int = 3):
    rng = random.Random(cfg.seed)

    def mutate(p: Dict, rate: float = 0.25) -> Dict:
        q = dict(p)
        if rng.random() < rate:
            q["max_depth"] = int(clamp(q["max_depth"] + rng.randint(-2, 2), cfg.max_depth_min, cfg.max_depth_max))
        if rng.random() < rate:
            q["max_leaf_nodes"] = int(clamp(q["max_leaf_nodes"] + rng.randint(-40, 40), cfg.max_leaf_nodes_min, cfg.max_leaf_nodes_max))
        if rng.random() < rate:
            # multiplicative-ish mutation in log space
            lr = math.log10(q["learning_rate"]) + rng.uniform(-0.4, 0.4)
            lr = clamp(lr, cfg.log10_lr_min, cfg.log10_lr_max)
            q["learning_rate"] = 10 ** lr
        if rng.random() < rate:
            q["max_bins"] = int(clamp(q["max_bins"] + rng.randint(-40, 40), cfg.max_bins_min, cfg.max_bins_max))
        if rng.random() < rate:
            q["min_samples_leaf"] = int(clamp(q["min_samples_leaf"] + rng.randint(-30, 30), cfg.min_samples_leaf_min, cfg.min_samples_leaf_max))
        if rng.random() < rate:
            l2 = math.log10(q["l2_regularization"]) + rng.uniform(-0.6, 0.6)
            l2 = clamp(l2, cfg.log10_l2_min, cfg.log10_l2_max)
            q["l2_regularization"] = 10 ** l2
        if rng.random() < rate:
            q["early_stopping"] = (not q["early_stopping"])
        if rng.random() < rate:
            q["validation_fraction"] = float(clamp(q["validation_fraction"] + rng.uniform(-0.05, 0.05),
                                                 cfg.validation_fraction_min, cfg.validation_fraction_max))
        if rng.random() < rate:
            q["n_estimators"] = int(clamp(q["n_estimators"] + rng.randint(-200, 200),
                                          cfg.n_estimators_min, cfg.n_estimators_max))
        return q

    def crossover(a: Dict, b: Dict) -> Dict:
        child = {}
        for k in a.keys():
            child[k] = a[k] if rng.random() < 0.5 else b[k]
        return child

    population = [space.sample_random(rng) for _ in range(pop_size)]
    fitness: Dict[Tuple, float] = {}

    def fit(p: Dict) -> float:
        key = (
            int(p["max_depth"]),
            int(p["max_leaf_nodes"]),
            round(float(p["learning_rate"]), 12),
            int(p["max_bins"]),
            int(p["min_samples_leaf"]),
            round(float(p["l2_regularization"]), 12),
            bool(p["early_stopping"]),
            round(float(p["validation_fraction"]), 12),
            int(p["n_estimators"]),
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

    # evaluate initial population as time allows
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

        # replace worst (steady-state)
        worst_idx = min(range(len(population)), key=lambda j: fit(population[j]))
        population[worst_idx] = child

        if now - last_print >= cfg.print_every_s:
            elapsed = int(now - start)
            print(f"[{'genetic':>10}] t={elapsed:>4}s  evals={best.evals:>6}  best_cv={best.score:.5f}  params={best.params}")
            last_print = now

    elapsed = time.time() - start
    print(f"[{'genetic':>10}] DONE  t={elapsed:.1f}s  evals={best.evals}  best_cv={best.score:.5f}  params={best.params}")
    return best
