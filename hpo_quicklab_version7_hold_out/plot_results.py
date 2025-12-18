import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Data from your experiment
R = np.array([1,  5, 10, 50, 100], dtype=float)
mean = np.array([0.73077,  0.73846, 0.75385, 0.76923, 0.76923])
sem = np.array([0.02364, 0.02615, 0.01919, 0.0, 0.0])

# Smooth grid in log space
R_smooth = np.logspace(np.log10(R.min()), np.log10(R.max()), 400)

# Spline interpolation in log(R) space
logR = np.log10(R)
logR_smooth = np.log10(R_smooth)

mean_spline = make_interp_spline(logR, mean, k=3)
sem_spline = make_interp_spline(logR, sem, k=3)

mean_smooth = mean_spline(logR_smooth)
sem_smooth = np.maximum(sem_spline(logR_smooth), 0.0)  # guard against negatives

# Plot
plt.figure(figsize=(8, 5))

# Mean curve
plt.plot(
    R_smooth,
    mean_smooth,
    linewidth=3,
    label="Mean test accuracy"
)

# SEM band
plt.fill_between(
    R_smooth,
    mean_smooth - sem_smooth,
    mean_smooth + sem_smooth,
    alpha=0.25,
    label="±1 SEM"
)

# Original points
plt.scatter(R, mean, s=60, zorder=3)

# Styling
plt.xscale("log")
plt.xlabel("Repetition number (R) [log scale]")
plt.ylabel("Test accuracy")
plt.title("Effect of Repeated k-Fold CV on Test Performance (n = 50)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('results.png')
