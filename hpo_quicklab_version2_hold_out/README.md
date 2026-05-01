# HPO QuickLab Version 2: Hold-Out Experiment

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Status-MVP-orange)

## Short Introduction

This experiment compares several hyperparameter optimization methods for a machine learning classification task using a hold-out test set.

The experiment uses the UCI Adult income dataset and trains a `HistGradientBoostingClassifier`. The goal is to find strong hyperparameter configurations under a fixed time budget and compare the performance of different search methods.

## Problem Setup

The task is binary classification on the Adult income dataset.

The model predicts whether a person's income is:

- `>50K`
- `<=50K`

The dataset is loaded from OpenML. Categorical features are one-hot encoded, and numerical features are passed through directly. The data is then split into training and test sets.

The default split is:

- 75% training data
- 25% held-out test data

During hyperparameter optimization, the training set is used with cross-validation. The test set is not used during search. It is only used once at the end to evaluate the best configuration found by each method.

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

The experiment compares four hyperparameter optimization methods:

1. **Grid Search**

   A fixed grid of hyperparameter values is evaluated. This method is simple and systematic, but it can be inefficient when the search space is large.

2. **Random Search**

   Hyperparameter configurations are sampled randomly from the search space. This method can explore large spaces more efficiently than grid search.

3. **Bayesian Optimization**

   A Gaussian Process model is used to guide the search. The method uses expected improvement to choose promising candidate configurations.

4. **Genetic Algorithm**

   A population of configurations is evolved using selection, crossover, and mutation. Better configurations are more likely to influence future candidates.

Each method is run under the same time budget so that the comparison is fair.

## What Is Measured

The experiment measures both model quality and optimization efficiency.

During hyperparameter search, each candidate configuration is evaluated using cross-validation accuracy on the training set.

The main measurements are:

- Best cross-validation accuracy
- Number of evaluations completed
- Best hyperparameter configuration
- Runtime under the time budget
- Final test accuracy after retraining on the full training set

At the end of the experiment, the best configuration found by each method is retrained on the full training set and evaluated on the held-out test set.

The final output includes:

- Best CV score for grid search
- Best CV score for random search
- Best CV score for Bayesian optimization
- Best CV score for genetic algorithm
- Number of evaluations completed by each method
- Best hyperparameters found by each method
- Final test accuracy for each method

## How to Run the Code

This experiment is designed to run on an HPC cluster using SLURM.

### 1. Install Dependencies

The required Python packages are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
````

The required packages are:

```text
numpy
scikit-learn
pandas
```

### 2. Activate the Conda Environment

The SLURM script expects a conda environment named:

```bash
ml_exp
```

Activate it manually if running interactively:

```bash
conda activate ml_exp
```

### 3. Run with SLURM

Submit the batch script:

```bash
sbatch run_hpo.bash
```

You can also pass the experiment folder as the workspace path:

```bash
sbatch run_hpo.bash /path/to/hpo_quicklab_version2_hold_out
```

### 4. Command Used by the SLURM Script

The SLURM script runs:

```bash
python run.py \
  --budget 600 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42
```

This gives each search method a 600-second budget.

## Command-Line Options

You can also run the experiment directly with Python:

```bash
python run.py \
  --budget 600 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42 \
  --cv 3 \
  --test-size 0.25
```

Available options:

| Option          | Description                                         |
| --------------- | --------------------------------------------------- |
| `--budget`      | Time budget per algorithm in seconds                |
| `--print-every` | How often progress is printed                       |
| `--algos`       | Comma-separated list of algorithms to run           |
| `--seed`        | Random seed for reproducibility                     |
| `--cv`          | Number of cross-validation folds used during HPO    |
| `--test-size`   | Fraction of data reserved for the held-out test set |
| `--data-home`   | Optional OpenML cache directory                     |

Example: run only random search and Bayesian optimization:

```bash
python run.py \
  --budget 600 \
  --print-every 10 \
  --algos random,bayes \
  --seed 42
```

## Output

The SLURM job creates an output directory named using the job ID:

```bash
__experi_<job_id>_output/
```

The standard output and error files are written as:

```text
hpo_<job_id>.out
hpo_<job_id>.err
```

The output log reports progress during the search and prints a final summary.

Example final sections:

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

## Significance

This experiment shows how different hyperparameter optimization methods perform under the same computational budget.

It helps answer questions such as:

* Which search method finds the best validation accuracy?
* Which method performs the most evaluations within the time limit?
* Which method gives the best held-out test accuracy?
* How different search strategies behave on the same model and dataset?

This makes the experiment useful for studying the trade-off between search quality, runtime, and final model performance.

