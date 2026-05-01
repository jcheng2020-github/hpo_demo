# HPO QuickLab Version 7: Repeated-CV HPO on Small Sample Size

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Short Introduction

This experiment studies how repeated k-fold cross-validation affects hyperparameter optimization when the dataset is very small.

The experiment uses the UCI Adult income dataset and trains a `HistGradientBoostingClassifier`. A small stratified subsample is selected from the full dataset, with the default sample size fixed at:

```bash
--n-total 50
````

The main question is:

**Does increasing the number of cross-validation repetitions improve the stability and final test performance of HPO when the dataset is tiny?**

## Problem Setup

The task is binary classification on the Adult income dataset.

The model predicts whether income is:

* `>50K`
* `<=50K`

The full Adult dataset is loaded from OpenML. Categorical features are one-hot encoded, numerical features are passed through, and the final feature matrix is stored as a dense NumPy array. 

For each outer seed, the experiment:

1. Loads the full Adult dataset.
2. Creates a stratified subsample of size `n_total`.
3. Splits that small subsample into train and test sets.
4. Runs HPO on the training set using repeated k-fold CV.
5. Retrains the best model on the small training set.
6. Evaluates the final model once on the held-out test set.

The default held-out split is:

```bash
--test-size 0.25
```

With the default `n_total=50`, this means the experiment uses approximately:

* 75% for training
* 25% for testing

The subsampling and train/test split are stratified to preserve the class balance as much as possible. 

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

## Hyperparameter Optimization Method

This experiment uses:

### Random Search

Random search samples hyperparameter configurations from the search space.

For each sampled configuration, the objective score is computed using repeated k-fold cross-validation on the training set.

Although `searchers.py` also contains grid search, Bayesian optimization, and genetic algorithm implementations, this repeated-CV experiment calls `random_search` in the main runner.  

## Repeated k-Fold CV Objective

The key feature of this experiment is the repeated-CV objective.

For each hyperparameter configuration:

1. Run k-fold CV.
2. Repeat the CV process `R` times.
3. Use a different split seed and model seed for each repetition.
4. Average accuracy across all folds and repetitions.
5. Use that average as the HPO objective score.

This reduces evaluation noise when `n` is very small, such as `n=50`. 

The default CV setting is:

```bash
--cv 3
```

The default repetition list is:

```bash
--repetitions 1,3,5,10
```

So, for example, with `--cv 3` and `--repetitions 10`, each hyperparameter configuration is evaluated over:

```text
3 folds × 10 repetitions = 30 validation scores
```

## What Is Measured

The experiment measures how repeated CV affects HPO performance and stability.

For each repetition count `R`, the experiment runs multiple outer seeds and records:

* Best repeated-CV accuracy found during HPO
* Number of HPO evaluations completed
* Held-out test accuracy after retraining
* Runtime per run
* Mean test accuracy across outer seeds
* Standard deviation of test accuracy
* Standard error of the mean

After finishing all seeds for a given `R`, the script aggregates the test accuracies using mean, standard deviation, and standard error. 

The default outer seeds are:

```bash
--seeds 0,1,2,3,4,5,6,7,8,9
```

The default output CSV is:

```bash
n50_repeatedcv_results.csv
```

## How to Run the Code

### 1. Install Dependencies

Install the required packages:

```bash
pip install numpy scikit-learn pandas matplotlib scipy
```

If your folder includes a `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

### 2. Run the Default Experiment

Run the repeated-CV experiment with the default settings:

```bash
python run_repeatedcv_n50.py \
  --budget 180 \
  --cv 3 \
  --test-size 0.25 \
  --n-total 50 \
  --repetitions 1,3,5,10 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --out-csv n50_repeatedcv_results.csv
```

### 3. Run a Faster Test

For a quick test run:

```bash
python run_repeatedcv_n50.py \
  --budget 60 \
  --n-total 50 \
  --repetitions 1,3 \
  --seeds 0,1 \
  --out-csv quick_n50_repeatedcv_results.csv
```

### 4. Run More CV Repetitions

To test larger repetition counts:

```bash
python run_repeatedcv_n50.py \
  --budget 180 \
  --cv 3 \
  --n-total 50 \
  --repetitions 1,5,10,50,100 \
  --seeds 0,1,2,3,4 \
  --out-csv n50_repeatedcv_largeR_results.csv
```

### 5. Use an OpenML Cache Directory

To avoid repeatedly downloading the Adult dataset:

```bash
python run_repeatedcv_n50.py \
  --budget 180 \
  --data-home ./openml_cache \
  --out-csv n50_repeatedcv_results.csv
```

## Command-Line Options

| Option              | Description                                       |
| ------------------- | ------------------------------------------------- |
| `--budget`          | Time budget per HPO run, in seconds               |
| `--cv`              | Number of folds for CV inside HPO                 |
| `--test-size`       | Fraction of the small sample reserved for testing |
| `--n-total`         | Total stratified subsample size                   |
| `--repetitions`     | Comma-separated list of CV repetition counts      |
| `--seeds`           | Comma-separated list of outer random seeds        |
| `--countdown-every` | How often to print countdown progress             |
| `--data-home`       | Optional OpenML cache directory                   |
| `--out-csv`         | Output CSV filename                               |

## Output

The experiment prints progress for each run, including:

* Current repetition count `R`
* Current outer seed
* Best CV score found by random search
* Number of evaluations
* Final test accuracy
* Runtime
* Estimated remaining time

At the end, it prints a final summary:

```text
FINAL SUMMARY (TEST performance after HPO)

   R      mean       std       sem  outer_seeds
------------------------------------------------
   1   ...
   3   ...
   5   ...
  10   ...
```

It also writes the aggregate results to a CSV file.

Default output:

```text
n50_repeatedcv_results.csv
```

The CSV contains:

* `n_total`
* `test_size`
* `cv_folds`
* `repetitions_R`
* `n_repeats_outer_seed`
* `test_acc_mean`
* `test_acc_std`
* `test_acc_sem`

## Plot the Results

After generating the CSV, run:

```bash
python plot_results.py
```

The plotting script saves:

```text
results.png
```

The plot shows mean test accuracy as a function of the repeated-CV repetition count `R`, using a log-scaled x-axis and an uncertainty band based on SEM. 

## Significance

This experiment is designed for the difficult small-data case where normal k-fold CV can be noisy.

It helps answer:

* Does repeated k-fold CV improve HPO reliability when `n=50`?
* Does increasing the repetition count improve final test accuracy?
* Does repeated CV reduce variation across random seeds?
* How much additional evaluation cost is introduced by repeated CV?
* At what repetition count does performance stop improving?

This version is useful for studying the trade-off between evaluation stability and compute cost in hyperparameter optimization with very small datasets.
