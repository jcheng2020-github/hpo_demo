# HPO QuickLab Version 6: Sample Size and Hold-Out HPO Experiment

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Short Introduction

This experiment studies how dataset sample size affects model performance after hyperparameter optimization.

The experiment uses the UCI Adult income dataset and trains a `HistGradientBoostingClassifier`. Instead of running only one fixed train/test experiment, this version repeatedly subsamples the dataset at different sample sizes, runs HPO, evaluates the best model on a held-out test set, and summarizes the mean and variation of test accuracy.

The main question is:

**How does test accuracy after HPO change as the amount of training data increases?**

## Problem Setup

The task is binary classification on the Adult income dataset.

The model predicts whether income is:

- `>50K`
- `<=50K`

The dataset is loaded from OpenML. Categorical variables are one-hot encoded, numerical variables are passed through directly, and the final feature matrix is converted to a dense NumPy array.

Unlike the earlier fixed-size HPO experiments, this version first loads the full Adult dataset and then creates stratified subsamples of different sizes. Stratified subsampling keeps the class balance approximately consistent across sample sizes.

For each sample size and random seed:

1. A stratified subsample is selected.
2. The subsample is split into training and test sets.
3. HPO is run on the training set.
4. The best configuration is retrained on the training set.
5. The final model is evaluated on the held-out test set.

The default test split is:

```bash
--test-size 0.25
````

So the default setup uses:

* 75% training data
* 25% held-out test data

## Method Covered

This experiment tunes a `HistGradientBoostingClassifier`.

The searched hyperparameters include:

* `max_depth`
* `max_leaf_nodes`
* `learning_rate`
* `max_bins`
* `min_samples_leaf`
* `l2_regularization`
* `early_stopping`
* `validation_fraction`
* `n_estimators`

The configured search ranges include tree depth from 3 to 12, max leaf nodes from 15 to 255, learning rate from `1e-3` to about `3e-1`, max bins from 32 to 255, and number of estimators from 100 to 1000. 

## Hyperparameter Optimization Methods

This sample-size experiment supports:

### 1. Random Search

Random search samples hyperparameter configurations from the search space.

It is the default method for this experiment:

```bash
--algo random
```

### 2. Grid Search

Grid search evaluates configurations from a fixed coarse grid.

It can be selected with:

```bash
--algo grid
```

The script is designed specifically for comparing sample-size behavior using either random search or grid search. 

The shared `searchers.py` file also contains Bayesian optimization and genetic algorithm implementations, but the sample-size runner only exposes `random` and `grid` through the `--algo` argument.  

## What Is Measured

The experiment measures how held-out test accuracy changes as sample size increases.

For each sample size, the experiment runs multiple seeds and records:

* Best cross-validation accuracy during HPO
* Held-out test accuracy after HPO
* Runtime for each run
* Number of HPO evaluations
* Mean test accuracy across seeds
* Standard deviation of test accuracy
* Standard error of the mean

The HPO objective is cross-validation accuracy on the training set only. The objective uses `cross_val_score` with accuracy scoring and the configured number of CV folds. 

After all seeds are completed for a sample size, the script aggregates the test accuracy values using mean, standard deviation, and standard error. 

The default sample sizes are:

```text
500,1000,2000,5000,10000,20000,40000
```

The default seeds are:

```text
0,1,2,3,4
```

These defaults mean the experiment runs 7 sample sizes × 5 seeds = 35 HPO runs by default. 

## Output

The main script writes a CSV file containing the aggregate results.

Default output file:

```text
sample_size_hpo_results.csv
```

The CSV contains:

* `sample_size_total`
* `n_train`
* `n_test`
* `n_repeats`
* `test_acc_mean`
* `test_acc_std`
* `test_acc_sem`

The script also prints a final summary table to the console and writes the aggregate rows to the output CSV. 

A plotting script is also included. It plots sample size against mean test accuracy and uses an uncertainty band based on SEM. The plot is saved as `results.png`. 

## How to Run the Code

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install numpy scikit-learn pandas matplotlib
```

If your folder includes a `requirements.txt`, install from it instead:

```bash
pip install -r requirements.txt
```

### 2. Run the Sample Size HPO Experiment

Run the default random-search sample-size experiment:

```bash
python hpo_sample_size.py \
  --algo random \
  --budget 180 \
  --cv 3 \
  --test-size 0.25 \
  --sample-sizes 500,1000,2000,5000,10000,20000,40000 \
  --seeds 0,1,2,3,4 \
  --out-csv sample_size_hpo_results.csv
```

### 3. Run Grid Search Instead

```bash
python hpo_sample_size.py \
  --algo grid \
  --budget 180 \
  --cv 3 \
  --test-size 0.25 \
  --sample-sizes 500,1000,2000,5000,10000,20000,40000 \
  --seeds 0,1,2,3,4 \
  --out-csv sample_size_hpo_results_grid.csv
```

### 4. Run a Smaller Quick Test

For a short test run:

```bash
python hpo_sample_size.py \
  --algo random \
  --budget 60 \
  --sample-sizes 500,1000 \
  --seeds 0,1 \
  --out-csv quick_sample_size_results.csv
```

### 5. Use an OpenML Cache Directory

To avoid repeatedly downloading the Adult dataset:

```bash
python hpo_sample_size.py \
  --algo random \
  --budget 180 \
  --data-home ./openml_cache \
  --out-csv sample_size_hpo_results.csv
```

## Command-Line Options

| Option              | Description                                     |
| ------------------- | ----------------------------------------------- |
| `--algo`            | HPO method to run: `random` or `grid`           |
| `--budget`          | Time budget per HPO run, in seconds             |
| `--cv`              | Number of CV folds used inside HPO              |
| `--test-size`       | Fraction of each subsample reserved for testing |
| `--sample-sizes`    | Comma-separated list of total sample sizes      |
| `--seeds`           | Comma-separated list of random seeds            |
| `--data-home`       | Optional OpenML cache directory                 |
| `--countdown-every` | How often to print countdown progress           |
| `--out-csv`         | Output CSV filename                             |

## Plot the Results

After the CSV is generated, run:

```bash
python plot_results.py
```

This creates:

```text
results.png
```

The plot shows test accuracy after HPO as a function of sample size on a log-scaled x-axis.

## Expected Final Console Output

The script prints a final summary like:

```text
FINAL SUMMARY (TEST performance after HPO)

sample_size   n_train   n_test  repeats      mean       std       sem
---------------------------------------------------------------------
       500       ...      ...        5   ...
      1000       ...      ...        5   ...
      2000       ...      ...        5   ...
      5000       ...      ...        5   ...
     10000       ...      ...        5   ...
     20000       ...      ...        5   ...
     40000       ...      ...        5   ...
```

## Significance

This experiment helps show how much data is needed for HPO to produce stable and strong test performance.

It helps answer:

* Does test accuracy improve as sample size increases?
* At what sample size does performance begin to plateau?
* How much variation appears across random seeds?
* How stable is HPO when the dataset is small?
* Does the benefit of more data outweigh the added compute cost?

This version is useful for studying the relationship between sample size, hyperparameter optimization, and generalization performance.
