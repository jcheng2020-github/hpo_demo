from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class ObjectiveResult:
    score: float
    elapsed_s: float


class GBDTObjectiveRepeatedCV:
    """
    Objective: repeated k-fold CV accuracy of HistGradientBoostingClassifier on TRAIN set only.

    For each hyperparameter setting p:
      - repeat the k-fold CV 'repetitions' times
      - each repetition uses a different CV split random_state and model random_state
      - objective score = mean over all folds and repetitions

    This reduces evaluation noise when n is tiny (e.g., n=50).
    """
    def __init__(self, X_train, y_train, cv_folds: int, seed: int, repetitions: int = 1):
        self.X_train = X_train
        self.y_train = y_train
        self.cv_folds = int(cv_folds)
        self.seed = int(seed)
        self.repetitions = int(repetitions)

        # Cache to avoid reevaluating same params during search
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
        k = (self._key(p), self.cv_folds, self.repetitions, self.seed)
        if k in self.cache:
            return self.cache[k]

        t0 = time.time()
        scores_all = []

        for r in range(self.repetitions):
            model = HistGradientBoostingClassifier(
                max_depth=int(p["max_depth"]),
                max_leaf_nodes=int(p["max_leaf_nodes"]),
                learning_rate=float(p["learning_rate"]),
                max_bins=int(p["max_bins"]),
                min_samples_leaf=int(p["min_samples_leaf"]),
                l2_regularization=float(p["l2_regularization"]),
                early_stopping=bool(p["early_stopping"]),
                validation_fraction=float(p["validation_fraction"]),
                max_iter=int(p["n_estimators"]),
                random_state=self.seed + 10_000 * r,  # different model seed per repetition
            )

            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.seed + 1_000 * r,  # different split seed per repetition
            )

            scores = cross_val_score(
                model,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            scores_all.append(scores)

        scores_all = np.concatenate(scores_all, axis=0)
        res = ObjectiveResult(score=float(np.mean(scores_all)), elapsed_s=(time.time() - t0))
        self.cache[k] = res
        return res
