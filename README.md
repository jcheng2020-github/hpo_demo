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

## Experiment Summary

This repository contains a sequence of HPO QuickLab hold-out experiments. Each experiment folder includes its own `README.md` file describing the problem setup, optimization methods, evaluation metrics, and execution commands. Together, these experiments compare hyperparameter optimization methods across classical machine learning models, neural network models, different compute budgets, sample-size settings, and repeated cross-validation strategies.

| Experiment Folder | Main Setup | Model / Dataset | HPO Methods Covered | Main Measurements |
|---|---|---|---|---|
| `hpo_quicklab_version2_hold_out` | Baseline hold-out HPO experiment for tabular classification | HistGradientBoostingClassifier on Adult income dataset | Grid search, random search, Bayesian optimization, genetic algorithm | Best CV accuracy, number of evaluations, best hyperparameters, held-out test accuracy |
| `hpo_quicklab_version3_hold_out` | Neural-network HPO with CPU/GPU execution | MLP on Fashion-MNIST; Small CNN on CIFAR-10 | Grid search, random search, Bayesian optimization, genetic algorithm | Best validation accuracy, number of evaluations, best hyperparameters, final test accuracy |
| `hpo_quicklab_version4_hold_out` | Hold-out HPO experiment for gradient boosting | HistGradientBoostingClassifier on Adult income dataset | Grid search, random search, Bayesian optimization, genetic algorithm | Best training-CV accuracy, number of evaluations, best hyperparameters, held-out test accuracy |
| `hpo_quicklab_version5_hold_out` | Neural-network HPO with longer time budget | MLP on Fashion-MNIST; Small CNN on CIFAR-10 | Grid search, random search, Bayesian optimization, genetic algorithm | Best validation accuracy, number of evaluations, best hyperparameters, final test accuracy under a longer budget |
| `hpo_quicklab_version6_hold_out` | Sample-size study after HPO | HistGradientBoostingClassifier on stratified Adult subsamples | Random search and grid search | Test accuracy mean, standard deviation, SEM, best CV accuracy, effect of sample size |
| `hpo_quicklab_version7_hold_out` | Repeated-CV study for very small sample HPO | HistGradientBoostingClassifier on Adult dataset with default `n=50` | Random search with repeated k-fold CV objective | Best repeated-CV accuracy, evaluations, test accuracy mean, standard deviation, SEM across repetition counts |

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
