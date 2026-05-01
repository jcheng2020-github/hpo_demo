# HPO QuickLab Version 4: Hold-Out Gradient Boosting Experiment

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Short Introduction

This experiment compares different hyperparameter optimization methods for a binary classification task.

The experiment uses the UCI Adult income dataset and trains a `HistGradientBoostingClassifier`. The purpose is to compare how well different search methods find strong hyperparameter configurations under the same time budget.

## Problem Setup

The task is binary classification on the Adult income dataset.

The model predicts whether a person's income is:

- `>50K`
- `<=50K`

The dataset is loaded from OpenML. Categorical features are transformed with one-hot encoding, and numerical features are passed through directly. The encoded feature matrix is kept dense because `HistGradientBoostingClassifier` requires dense input.

The default data split is:

- 75% training data
- 25% held-out test data

The training set is used during hyperparameter optimization. The test set is held out and used only once at the end for final evaluation.

## Method Covered

This experiment tunes a `HistGradientBoostingClassifier`.

The hyperparameters searched include:

- `max_depth`
- `max_leaf_nodes`
- `learning_rate`
- `max_bins`
- `min_samples_leaf`
- `l2_regularization`
- `early_stopping`
- `validation_fraction`
- `n_estimators`

The configured search ranges include tree depth from 3 to 12, max leaf nodes from 15 to 255, learning rate from `1e-3` to about `3e-1`, max bins from 32 to 255, and number of estimators from 100 to 1000. :contentReference[oaicite:2]{index=2}

## Hyperparameter Optimization Methods

This experiment compares four search methods:

### 1. Grid Search

Grid search evaluates configurations from a fixed coarse grid.

It is simple and systematic, but it can be inefficient when the search space is large.

### 2. Random Search

Random search samples configurations randomly from the search space.

It can explore a wider range of values than grid search under the same time budget.

### 3. Bayesian Optimization

Bayesian optimization uses a Gaussian Process surrogate model and expected improvement to propose promising configurations after initial random evaluations.

### 4. Genetic Algorithm

The genetic algorithm maintains a population of configurations and improves them using:

- Tournament selection
- Crossover
- Mutation
- Replacement of weak candidates

The implementation includes grid search, random search, Bayesian optimization with GP expected improvement, and a genetic algorithm. :contentReference[oaicite:3]{index=3}

## What Is Measured

The main optimization objective is cross-validation accuracy on the training set.

Each candidate configuration is evaluated using `cross_val_score` with accuracy as the scoring metric. The default number of CV folds is 3. :contentReference[oaicite:4]{index=4}

The experiment measures:

- Best training cross-validation accuracy
- Number of evaluations completed
- Best hyperparameter configuration
- Runtime progress under the time budget
- Final held-out test accuracy

After each search method finishes, the best configuration is retrained on the full training set and evaluated on the held-out test set. :contentReference[oaicite:5]{index=5}

## How to Run the Code

### 1. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
````

The required packages are:

```text
numpy
scikit-learn
pandas
```

### 2. Run the Experiment

Run all four search methods with the default settings:

```bash
python run.py \
  --budget 600 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42
```

The default command gives each search method a 600-second time budget and prints progress every 10 seconds. The command-line interface supports budget, print frequency, seed, algorithm list, CV folds, test size, and OpenML cache directory. 

### 3. Run a Subset of Methods

Example: run only random search and Bayesian optimization:

```bash
python run.py \
  --budget 600 \
  --print-every 10 \
  --algos random,bayes \
  --seed 42
```

### 4. Change CV Folds or Test Split

Example:

```bash
python run.py \
  --budget 600 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42 \
  --cv 3 \
  --test-size 0.25
```

### 5. Use a Custom OpenML Cache Directory

```bash
python run.py \
  --budget 600 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42 \
  --data-home ./openml_cache
```

## Command-Line Options

| Option          | Description                                            |
| --------------- | ------------------------------------------------------ |
| `--budget`      | Time budget per algorithm in seconds                   |
| `--print-every` | Print best-so-far progress every N seconds             |
| `--seed`        | Random seed for reproducibility                        |
| `--algos`       | Comma-separated list of algorithms to run              |
| `--cv`          | Number of CV folds used during HPO on the training set |
| `--test-size`   | Fraction of data reserved for the held-out test set    |
| `--data-home`   | Optional OpenML cache directory                        |

## Output

The experiment prints progress for each method during the search.

At the end, it prints two final sections:

```text
=== SUMMARY (TRAIN-CV best within budget) ===
    grid: best_cv=...  evals=...  params={...}
  random: best_cv=...  evals=...  params={...}
   bayes: best_cv=...  evals=...  params={...}
 genetic: best_cv=...  evals=...  params={...}

=== TEST SET PERFORMANCE (retrain on full TRAIN, evaluate once) ===
    grid: test_acc=...
  random: test_acc=...
   bayes: test_acc=...
 genetic: test_acc=...
```

The summary reports the best cross-validation result found within the time budget. The test performance section reports the final held-out test accuracy after retraining the best configuration for each method. 

## Significance

This experiment compares HPO methods under equal compute budgets on the same dataset, model, and evaluation procedure.

It helps answer:

* Which search method finds the best CV accuracy?
* Which method completes more evaluations within the same time limit?
* Which method gives the best held-out test accuracy?
* How efficient is each method for tuning gradient boosting hyperparameters?

This version is useful as a controlled baseline for studying search quality, runtime, and generalization performance in hyperparameter optimization.
