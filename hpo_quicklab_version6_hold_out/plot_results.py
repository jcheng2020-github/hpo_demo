import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Results from experiment
# -----------------------
sample_size = np.array([50, 500, 1000, 2000, 5000, 10000, 20000, 40000])
mean = np.array([0.69231, 0.82400, 0.85040, 0.86000, 0.86064, 0.86088, 0.86516, 0.86526])
std = np.array([0.09421, 0.01265, 0.01513, 0.00980, 0.00770, 0.00624, 0.00356, 0.00769])
sem = np.array([0.04213, 0.00566, 0.00676, 0.00438, 0.00344, 0.00279, 0.00159, 0.00344])

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8, 5))

# Continuous curve
plt.plot(sample_size, mean, marker='o', linewidth=2, label="Mean test accuracy")

# Uncertainty band (choose ONE: std or sem)
plt.fill_between(
    sample_size,
    mean - sem,
    mean + sem,
    alpha=0.25,
    label="± 1 SEM"
)

# Optional: log-scale x-axis (recommended)
plt.xscale("log")

plt.xlabel("Sample size (log scale)")
plt.ylabel("Test accuracy after HPO")
plt.title("Effect of Sample Size on Test Performance After HPO")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig('results.png')
