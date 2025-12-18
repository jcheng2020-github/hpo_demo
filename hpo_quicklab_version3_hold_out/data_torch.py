# data_torch.py
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_datasets(dataset: str, data_dir: str, seed: int):
    """
    Returns: train_full, test, input_shape, num_classes
    """
    _seed_everything(seed)

    if dataset == "fashion_mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),  # common FashionMNIST normalization
        ])
        train_full = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tfm)
        input_shape = (1, 28, 28)
        num_classes = 10
        return train_full, test, input_shape, num_classes

    if dataset == "cifar10":
        tfm_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm_train)
        test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm_test)
        input_shape = (3, 32, 32)
        num_classes = 10
        return train_full, test, input_shape, num_classes

    raise ValueError(f"Unknown dataset: {dataset}. Use 'fashion_mnist' or 'cifar10'.")

def split_train_val(train_full, val_fraction: float, seed: int):
    n = len(train_full)
    n_val = int(round(n * val_fraction))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=gen)
    return train_ds, val_ds

def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
