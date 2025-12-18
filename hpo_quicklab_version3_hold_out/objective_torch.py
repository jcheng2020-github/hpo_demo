# objective_torch.py
from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from data_torch import make_loader
from models import MLP, SmallCNN

@dataclass
class ObjectiveResult:
    score: float        # best val accuracy
    elapsed_s: float

def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def _accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

class TorchNNObjective:
    """
    HPO objective: train NN on TRAIN split, early-stop on VAL, return best VAL accuracy.
    """
    def __init__(self, cfg, train_ds, val_ds, input_shape, num_classes):
        self.cfg = cfg
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache: dict[Tuple, ObjectiveResult] = {}

    @staticmethod
    def _key(p: Dict) -> Tuple:
        # rounded floats to keep cache sane
        return (
            int(p["batch_size"]),
            round(float(p["lr"]), 12),
            round(float(p["weight_decay"]), 12),
            str(p["optimizer"]),
            round(float(p.get("momentum", 0.0)), 12),
            int(p["h1"]),
            int(p["h2"]),
            round(float(p["dropout"]), 6),
            int(p["max_epochs"]),
            int(p["patience"]),
        )

    def _build_model(self, p: Dict):
        if self.cfg.dataset == "fashion_mnist":
            return MLP(self.input_shape, self.num_classes, p["h1"], p["h2"], p["dropout"])
        if self.cfg.dataset == "cifar10":
            # base_channels tied to h1 loosely (keeps "neurons" concept)
            base = max(16, min(64, p["h1"] // 16))
            return SmallCNN(self.num_classes, base_channels=base, dropout=p["dropout"])
        raise ValueError("Unknown dataset in cfg.dataset")

    def _build_optimizer(self, p: Dict, params):
        opt = p["optimizer"]
        lr = float(p["lr"])
        wd = float(p["weight_decay"])

        if opt == "adam":
            return Adam(params, lr=lr, weight_decay=wd)
        if opt == "sgd":
            mom = float(p.get("momentum", 0.9))
            return SGD(params, lr=lr, momentum=mom, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {opt}")

    @torch.no_grad()
    def _eval_loader(self, model, loader):
        model.eval()
        total_acc = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            logits = model(xb)
            total_acc += _accuracy(logits, yb)
            n_batches += 1
        return total_acc / max(1, n_batches)

    def evaluate(self, p: Dict) -> ObjectiveResult:
        k = self._key(p)
        if k in self.cache:
            return self.cache[k]

        t0 = time.time()
        _seed_everything(self.cfg.seed)

        batch_size = int(p["batch_size"])
        train_loader = make_loader(self.train_ds, batch_size, shuffle=True,
                                   num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory)
        val_loader = make_loader(self.val_ds, batch_size, shuffle=False,
                                 num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory)

        model = self._build_model(p).to(self.device)
        opt = self._build_optimizer(p, model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        best_val = -1.0
        best_epoch = -1
        patience = int(p["patience"])
        max_epochs = int(p["max_epochs"])

        bad_epochs = 0

        for epoch in range(max_epochs):
            model.train()
            n_seen = 0
            for i, (xb, yb) in enumerate(train_loader):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

                n_seen += 1
                if self.cfg.max_batches_per_epoch is not None and n_seen >= self.cfg.max_batches_per_epoch:
                    break

            val_acc = self._eval_loader(model, val_loader)

            if val_acc > best_val + 1e-4:
                best_val = val_acc
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                break

        res = ObjectiveResult(score=float(best_val), elapsed_s=time.time() - t0)
        self.cache[k] = res
        return res

    def retrain_and_test(self, best_params: Dict, trainval_ds, test_ds) -> float:
        """
        Retrain on TRAIN+VAL using the best config, then evaluate on TEST once.
        """
        _seed_everything(self.cfg.seed)
        batch_size = int(best_params["batch_size"])

        train_loader = make_loader(trainval_ds, batch_size, shuffle=True,
                                   num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory)
        test_loader = make_loader(test_ds, batch_size, shuffle=False,
                                  num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory)

        model = self._build_model(best_params).to(self.device)
        opt = self._build_optimizer(best_params, model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        max_epochs = int(best_params["max_epochs"])
        for epoch in range(max_epochs):
            model.train()
            n_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

                n_seen += 1
                if self.cfg.max_batches_per_epoch is not None and n_seen >= self.cfg.max_batches_per_epoch:
                    break

        test_acc = self._eval_loader(model, test_loader)
        return float(test_acc)
