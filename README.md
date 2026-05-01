# HPO QuickLab: Hold-Out Hyperparameter Optimization Experiments

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Short Introduction

This repository contains a set of hold-out validation experiments for comparing hyperparameter optimization methods in machine learning. The experiments evaluate how different search strategies perform under the same time budget when tuning model hyperparameters.

The repository includes multiple experiment versions, from `hpo_quicklab_version2_hold_out` through `hpo_quicklab_version7_hold_out`.

## Background

Hyperparameter optimization is an important step in building effective machine learning models. Parameters such as learning rate, model size, dropout rate, and batch size can strongly affect model accuracy, training stability, and computational cost.

Instead of manually selecting these values, this project compares several automated search methods:

- Grid search
- Random search
- Bayesian optimization
- Genetic algorithm

Each method explores the hyperparameter search space differently, which can lead to different trade-offs between accuracy, runtime, and number of evaluations.

## Problem

The main problem addressed in this repository is:

**Which hyperparameter optimization method finds better model configurations under the same computational budget?**

A fair comparison requires each algorithm to be tested using the same validation setup, time budget, and evaluation metrics.

## Method

Each experiment uses a hold-out validation strategy.

The dataset is divided into:

- Training set
- Validation set
- Test set

The training set is used to train candidate models. The validation set is used to compare hyperparameter configurations during the search. The test set is used only after the best configuration has been selected.

Experiments are executed on an HPC cluster using SLURM. Each run uses a batch script that activates the `ml_exp` conda environment and launches `run.py` with a fixed budget and random seed.

## Significance

This repository provides a reproducible framework for comparing hyperparameter optimization algorithms. It helps show how different search methods perform when they are given the same computational resources.

The results can be used to understand:

- Which algorithm reaches strong validation performance fastest
- Which algorithm completes more evaluations within the same budget
- Which algorithm finds the best final model configuration
- How search strategy affects model performance and computational efficiency

# Maintenance Status

**Repository Name:** HPO QuickLab Hold-Out Experiments  
**Maintainer(s):** [Junfu Cheng, Department of Electrical and Computer Engineering, University of Florida, junfu.cheng@ufl.edu]  
**Status:** Active  
**Last Updated:** [2026-05-01]

## Status Definitions

- **Active**: Actively developed, bugs fixed, new features added.
- **Maintenance Mode**: Only critical bug fixes and security updates.
- **Deprecated**: No longer recommended; no new features; may be removed in future.
- **Archived**: Read-only repository; no longer maintained.

## Notes

This repository is currently an MVP for hyperparameter optimization experiments using hold-out validation and SLURM-based execution.

# Support Policy

**Repository Name:** HPO QuickLab Hold-Out Experiments  
**Maintainer(s):** [Junfu Cheng, Department of Electrical and Computer Engineering, University of Florida, junfu.cheng@ufl.edu]  

## Supported Versions

- Active branches: `main`
- Unsupported branches: [Nan]

## Issue Handling

- Bugs or issues in active branches will be addressed within [10] business days.
- Pull requests from the community will be reviewed based on relevance and priority.
- Deprecated or archived repositories may not receive support.

## Contribution Guidelines

- Please refer to `CONTRIBUTING.md` for pull request and issue submission guidelines.
- When reporting a bug, include steps to reproduce, environment details, logs, and error messages.

## Contact

For questions or clarifications, contact [Junfu Cheng, Department of Electrical and Computer Engineering, University of Florida, junfu.cheng@ufl.edu].
