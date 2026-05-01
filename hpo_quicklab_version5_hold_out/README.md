# HPO QuickLab Version 5: Hold-Out Neural Network Experiment

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Short Introduction

This experiment compares hyperparameter optimization methods for PyTorch neural network models.

The experiment supports two image classification tasks:

- **MLP on Fashion-MNIST**
- **Small CNN on CIFAR-10**

The goal is to evaluate how different hyperparameter search methods perform under the same time budget.

Compared with earlier neural-network versions, this version uses a longer SLURM wall time and a larger search budget of 900 seconds per algorithm.

## Problem Setup

This is a supervised image classification experiment.

The code supports two datasets:

### Fashion-MNIST

Fashion-MNIST contains grayscale clothing images from 10 classes.

For this dataset, the experiment uses an MLP model.

### CIFAR-10

CIFAR-10 contains RGB natural images from 10 classes.

For this dataset, the experiment uses a small CNN model.

The dataset is split into:

- Training set
- Validation set
- Test set

The default validation fraction is:

```bash
--val-frac 0.15
````

During hyperparameter optimization, each candidate configuration is trained on the training split and evaluated on the validation split.

After the best configuration is selected, the model is retrained on the combined training and validation data and evaluated once on the test set.

## Models Used

### MLP for Fashion-MNIST

The MLP contains:

* Input flattening
* Fully connected layer 1
* ReLU activation
* Dropout
* Fully connected layer 2
* ReLU activation
* Dropout
* Output classification layer

The hidden layer sizes and dropout rate are tuned during HPO.

### Small CNN for CIFAR-10

The CNN contains:

* Four convolutional layers
* ReLU activations
* Max pooling
* Dropout
* Fully connected layer
* Output classification layer

The base number of convolutional channels is indirectly controlled through the `h1` hyperparameter.

## Methods Covered

This experiment compares four hyperparameter optimization methods.

### 1. Grid Search

Grid search evaluates configurations from a fixed grid.

It is easy to understand and reproducible, but it can be inefficient in high-dimensional search spaces.

### 2. Random Search

Random search samples configurations randomly from the search space.

It is often more flexible than grid search because it can explore a wider range of values under the same time budget.

### 3. Bayesian Optimization

Bayesian optimization uses a Gaussian Process surrogate model.

After initial random evaluations, it uses expected improvement to propose promising hyperparameter configurations.

### 4. Genetic Algorithm

The genetic algorithm evolves a population of candidate configurations.

It uses:

* Tournament selection
* Crossover
* Mutation
* Replacement of weaker candidates

This allows the search to gradually improve candidate configurations over time.

## Hyperparameters Tuned

The neural-network search space includes:

* `batch_size`
* `lr`
* `weight_decay`
* `optimizer`
* `momentum`
* `h1`
* `h2`
* `dropout`
* `max_epochs`
* `patience`

For Fashion-MNIST:

* `h1` controls hidden layer 1 size
* `h2` controls hidden layer 2 size
* `dropout` controls dropout after hidden layers

For CIFAR-10:

* `h1` indirectly controls the CNN base channel count
* `dropout` controls dropout in the CNN
* `h2` remains part of the shared search space

## What Is Measured

The main HPO objective is:

```text
Best validation accuracy
```

Each method reports:

* Best validation accuracy
* Number of evaluations completed
* Best hyperparameter configuration
* Runtime progress during the search

After the search finishes, the best configuration from each method is retrained on the combined training and validation set. The final model is then evaluated once on the test set.

The final output includes:

* Best validation accuracy for grid search
* Best validation accuracy for random search
* Best validation accuracy for Bayesian optimization
* Best validation accuracy for genetic algorithm
* Number of evaluations completed by each method
* Best hyperparameter configuration for each method
* Final test accuracy for each method

## How to Run the Code

This experiment is designed to run on an HPC cluster using SLURM.

There are two SLURM scripts:

```text
submit_hpo_torch_cpu.sh
submit_hpo_torch_gpu.sh
```

Use the CPU script for Fashion-MNIST and the GPU script for CIFAR-10.

## Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
torch
torchvision
scikit-learn
```

## Conda Environment

The SLURM scripts expect a conda environment named:

```bash
ml_exp_torch
```

Activate it manually if running outside SLURM:

```bash
conda activate ml_exp_torch
```

## Run Fashion-MNIST on CPU

Submit the CPU job:

```bash
sbatch submit_hpo_torch_cpu.sh
```

You can also pass the workspace path explicitly:

```bash
sbatch submit_hpo_torch_cpu.sh /path/to/hpo_quicklab_version5_hold_out
```

The CPU script runs:

```bash
python run.py \
  --dataset fashion_mnist \
  --data-dir "${DATADIR}" \
  --budget 900 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42 \
  --val-frac 0.15 \
  --max-batches-per-epoch 200
```

## Run CIFAR-10 on GPU

Submit the GPU job:

```bash
sbatch submit_hpo_torch_gpu.sh
```

You can also pass the workspace path explicitly:

```bash
sbatch submit_hpo_torch_gpu.sh /path/to/hpo_quicklab_version5_hold_out
```

The GPU script runs:

```bash
python run.py \
  --dataset cifar10 \
  --data-dir "${DATADIR}" \
  --budget 900 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42 \
  --val-frac 0.15 \
  --max-batches-per-epoch 150
```

## Run Directly with Python

You can also run the experiment without SLURM.

Example for Fashion-MNIST:

```bash
python run.py \
  --dataset fashion_mnist \
  --data-dir ./data \
  --budget 900 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42 \
  --val-frac 0.15 \
  --max-batches-per-epoch 200
```

Example for CIFAR-10:

```bash
python run.py \
  --dataset cifar10 \
  --data-dir ./data \
  --budget 900 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42 \
  --val-frac 0.15 \
  --max-batches-per-epoch 150
```

## Command-Line Options

| Option                    | Description                                     |
| ------------------------- | ----------------------------------------------- |
| `--dataset`               | Dataset to use: `fashion_mnist` or `cifar10`    |
| `--data-dir`              | Directory for downloaded datasets               |
| `--budget`                | Time budget per algorithm in seconds            |
| `--print-every`           | Print progress every N seconds                  |
| `--seed`                  | Random seed for reproducibility                 |
| `--algos`                 | Comma-separated list of algorithms to run       |
| `--val-frac`              | Fraction of training data used for validation   |
| `--max-batches-per-epoch` | Caps training batches per epoch to speed up HPO |

## Output

Each SLURM job creates an output directory named with the job ID:

```text
__experi_<job_id>_output/
```

For the CPU job, output files are written as:

```text
hpo_cpu_<job_id>.out
hpo_cpu_<job_id>.err
console.log
```

For the GPU job, output files are written as:

```text
hpo_gpu_<job_id>.out
hpo_gpu_<job_id>.err
console.log
```

The output contains progress logs for each search method and a final summary.

Example final output:

```text
=== SUMMARY (best VAL accuracy within budget) ===
    grid: best_val=...  evals=...  params={...}
  random: best_val=...  evals=...  params={...}
   bayes: best_val=...  evals=...  params={...}
 genetic: best_val=...  evals=...  params={...}

=== TEST SET PERFORMANCE (retrain on TRAIN+VAL, evaluate once) ===
    grid: test_acc=...
  random: test_acc=...
   bayes: test_acc=...
 genetic: test_acc=...
```

## Significance

This experiment compares HPO algorithms under equal time budgets on neural-network image classification tasks.

It helps answer:

* Which search method finds the best validation accuracy?
* Which method completes more evaluations within the same time budget?
* Which method gives the best final test accuracy?
* How do HPO methods behave differently for MLP and CNN models?
* How does a longer search budget affect performance compared with shorter experiments?

This version is useful for studying the relationship between search strategy, model type, compute budget, and final classification performance.
