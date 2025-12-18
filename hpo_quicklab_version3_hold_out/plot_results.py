import matplotlib.pyplot as plt
import numpy as np

algos = ["grid", "random", "bayes", "genetic"]

mlp = {
    "name": "MLP on Fashion-MNIST",
    "best_val": [0.86248, 0.89872, 0.89872, 0.89872],
    "test_acc": [0.85430, 0.88750, 0.88750, 0.88750],
    "evals":    [42,      26,      26,      36],
}
cnn = {
    "name": "Small CNN on CIFAR-10",
    "best_val": [0.45891, 0.71117, 0.76146, 0.74969],
    "test_acc": [0.49104, 0.73369, 0.77742, 0.81338],
    "evals":    [27,      21,      23,      36],
}

n = len(algos)
gap = 1.4
y_mlp = np.arange(n)
y_cnn = y_mlp + (n + gap)

offset = 0.28
acc_height = 0.48
eval_height = 0.24

y_mlp_val = y_mlp - offset
y_mlp_test = y_mlp + offset
y_cnn_val = y_cnn - offset
y_cnn_test = y_cnn + offset

fig = plt.figure(figsize=(13.5, 8.5))
ax = plt.gca()
ax2 = ax.twiny()

# Accuracy bars
b_mlp_val = ax.barh(
    y_mlp_val, mlp["best_val"], height=acc_height,
    edgecolor="black", linewidth=1.2, label="MLP best_val"
)
b_mlp_test = ax.barh(
    y_mlp_test, mlp["test_acc"], height=acc_height,
    edgecolor="black", linewidth=1.2, label="MLP test_acc"
)
b_cnn_val = ax.barh(
    y_cnn_val, cnn["best_val"], height=acc_height,
    edgecolor="black", linewidth=1.2, label="CNN best_val"
)
b_cnn_test = ax.barh(
    y_cnn_test, cnn["test_acc"], height=acc_height,
    edgecolor="black", linewidth=1.2, label="CNN test_acc"
)

# Eval bars (distinct neutral colors)
e_mlp = ax2.barh(
    y_mlp, mlp["evals"], height=eval_height,
    color="lightgray", edgecolor="black", linewidth=1.2, label="MLP evals"
)
e_cnn = ax2.barh(
    y_cnn, cnn["evals"], height=eval_height,
    color="dimgray", edgecolor="black", linewidth=1.2, label="CNN evals"
)

# Y ticks (ONLY algorithm names, no MLP/CNN text on bars)
yticks = np.concatenate([y_mlp, y_cnn])
ylabs = [a.upper() for a in algos] + [a.upper() for a in algos]
ax.set_yticks(yticks)
ax.set_yticklabels(ylabs, fontsize=11)

# Axis ranges
ax.set_xlim(0.40, 0.92)
ax2.set_xlim(15, 50)

ax.set_xlabel("Accuracy (best_val / test_acc)", fontsize=12)
ax2.set_xlabel("Number of evaluations within time budget", fontsize=12)

ax.set_title(
    "HPO Results with Dual X-Axes\n"
    "Horizontal bars; accuracy vs. efficiency under identical time budget",
    fontsize=13
)

ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.6)

# Value annotations ONLY (no text labels like MLP/CNN)
def label_acc(bars):
    for b in bars:
        w = b.get_width()
        y = b.get_y() + b.get_height()/2
        ax.text(w + 0.006, y, f"{w:.3f}", va="center", fontsize=9)

def label_evals(bars):
    for b in bars:
        w = b.get_width()
        y = b.get_y() + b.get_height()/2
        ax2.text(w + 0.8, y, f"{int(round(w))}", va="center", fontsize=9)

label_acc(b_mlp_val); label_acc(b_mlp_test)
label_acc(b_cnn_val); label_acc(b_cnn_test)
label_evals(e_mlp); label_evals(e_cnn)

# Legend kept intact
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=10, frameon=True)

ax.invert_yaxis()

plt.tight_layout()
out_path = "hpo_nn_results_no_text_on_bars.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
