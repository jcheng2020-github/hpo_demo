# objective_gbdt.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

@dataclass
class ObjectiveResult:
    score: float
    elapsed_s: float

class GBDTObjective:
    """
    Objective: CV accuracy of HistGradientBoostingClassifier on TRAIN set only.
    Params expected:
      max_depth, max_leaf_nodes, learning_rate, max_bins,
      min_samples_leaf, l2_regularization, early_stopping,
      validation_fraction, n_estimators
    """
    def __init__(self, X_train, y_train, cv_folds: int, seed: int):
        self.X_train = X_train
        self.y_train = y_train
        self.cv_folds = cv_folds
        self.seed = seed
        self.cache: Dict[Tuple, ObjectiveResult] = {}

    @staticmethod
    def _key(p: Dict) -> Tuple:
        # stable cache key (rounded floats)
        return (
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

    def evaluate(self, p: Dict) -> ObjectiveResult:
        k = self._key(p)
        if k in self.cache:
            return self.cache[k]

        t0 = time.time()

        model = HistGradientBoostingClassifier(
            max_depth=int(p["max_depth"]),
            max_leaf_nodes=int(p["max_leaf_nodes"]),
            learning_rate=float(p["learning_rate"]),
            max_bins=int(p["max_bins"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            l2_regularization=float(p["l2_regularization"]),
            early_stopping=bool(p["early_stopping"]),
            validation_fraction=float(p["validation_fraction"]),
            max_iter=int(p["n_estimators"]),  # IMPORTANT: estimator uses max_iter
            random_state=self.seed,
        )

        scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=self.cv_folds,
            scoring="accuracy",
            n_jobs=-1,
        )

        res = ObjectiveResult(score=float(np.mean(scores)), elapsed_s=(time.time() - t0))
        self.cache[k] = res
        return res
