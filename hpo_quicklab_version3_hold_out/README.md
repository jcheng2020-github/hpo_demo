# HPO QuickLab Version 3: Hold-Out Neural Network Experiment

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Short Introduction

This experiment compares hyperparameter optimization methods for training neural networks with PyTorch.

The experiment supports two image classification settings:

- **MLP on Fashion-MNIST**
- **Small CNN on CIFAR-10**

The goal is to compare how different hyperparameter search methods perform under the same time budget.

## Problem Setup

This is a supervised image classification experiment.

The code supports two datasets:

1. **Fashion-MNIST**

   Fashion-MNIST contains grayscale clothing images with 10 classes.  
   In this setup, the model uses an MLP.

2. **CIFAR-10**

   CIFAR-10 contains RGB natural images with 10 classes.  
   In this setup, the model uses a small convolutional neural network.

The dataset is split into:

- Training set
- Validation set
- Test set

The validation fraction is set by default to:

```bash
--val-frac 0.15
````

During hyperparameter optimization, each candidate model is trained on the training split and evaluated on the validation split. After the best configuration is selected, the model is retrained on the combined training and validation data and evaluated once on the test set.

## Models Used

### MLP for Fashion-MNIST

The MLP model contains:

* Input flattening layer
* Fully connected layer 1
* ReLU activation
* Dropout
* Fully connected layer 2
* ReLU activation
* Dropout
* Output classification layer

The hidden layer sizes and dropout rate are part of the hyperparameter search space.

### Small CNN for CIFAR-10

The CNN model contains:

* Four convolutional layers
* ReLU activations
* Max pooling
* Dropout
* Fully connected layer
* Output classification layer

The base number of convolutional channels is indirectly controlled using the `h1` hyperparameter.

## Methods Covered

This experiment compares four hyperparameter optimization methods:

### 1. Grid Search

Grid search evaluates configurations from a fixed predefined grid.

It is simple and systematic, but it can be inefficient when the search space has many dimensions.

### 2. Random Search

Random search samples hyperparameter configurations randomly from the search space.

It can explore a large search space more flexibly than grid search.

### 3. Bayesian Optimization

Bayesian optimization uses a Gaussian Process model to guide the search.

After initial random evaluations, it uses expected improvement to propose promising configurations.

### 4. Genetic Algorithm

The genetic algorithm keeps a population of candidate configurations.

It uses:

* Tournament selection
* Crossover
* Mutation
* Replacement of weak candidates

This allows the search to evolve better configurations over time.

## Hyperparameters Tuned

The experiment searches over neural-network training and architecture hyperparameters.

The search space includes:

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

For Fashion-MNIST, `h1` and `h2` control the hidden units in the MLP.

For CIFAR-10, `h1` is used to derive the base number of CNN channels.

## What Is Measured

The experiment measures model performance and search efficiency.

During hyperparameter optimization, the objective is:

```text
Best validation accuracy
```

Each search method reports:

* Best validation accuracy
* Number of evaluations completed
* Best hyperparameter configuration
* Runtime progress during the search

After HPO finishes, the best configuration from each method is retrained on the combined training and validation data. Then it is evaluated on the test set.

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

Install the required packages:

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
sbatch submit_hpo_torch_cpu.sh /path/to/hpo_quicklab_version3_hold_out
```

The CPU script runs:

```bash
python run.py \
  --dataset fashion_mnist \
  --data-dir "${DATADIR}" \
  --budget 600 \
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
sbatch submit_hpo_torch_gpu.sh /path/to/hpo_quicklab_version3_hold_out
```

The GPU script runs:

```bash
python run.py \
  --dataset cifar10 \
  --data-dir "${DATADIR}" \
  --budget 600 \
  --print-every 10 \
  --algos grid,random,bayes,genetic \
  --seed 42 \
  --val-frac 0.15 \
  --max-batches-per-epoch 150
```

## Run Directly with Python

You can also run the experiment directly without SLURM.

Example for Fashion-MNIST:

```bash
python run.py \
  --dataset fashion_mnist \
  --data-dir ./data \
  --budget 600 \
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
  --budget 600 \
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
| `--algos`                 | Comma-separated algorithms to run               |
| `--val-frac`              | Fraction of training data used for validation   |
| `--max-batches-per-epoch` | Caps training batches per epoch to speed up HPO |

## Output

Each SLURM job creates an output directory named with the job ID:

```text
__experi_<job_id>_output/
```

For the CPU job, SLURM writes:

```text
hpo_cpu_<job_id>.out
hpo_cpu_<job_id>.err
console.log
```

For the GPU job, SLURM writes:

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

This experiment shows how different HPO algorithms perform when tuning neural networks under the same time budget.

It helps compare:

* Search quality
* Number of evaluations completed
* Validation accuracy
* Final test accuracy
* CPU vs GPU experiment settings
* MLP performance on Fashion-MNIST
* CNN performance on CIFAR-10

The experiment is useful for studying the relationship between search strategy, compute budget, and neural-network performance.
