import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np  # NEW
import matplotlib.ticker as ticker

rcParams['axes.linewidth'] = 1.2  # All subplot borders (left, bottom, top, right)
rcParams['lines.linewidth'] = 2.3

# --- Load Logit Lens data ---
logit_inter = pickle.load(open("cot_weights/arithmetic_inter_rank_results_16.pkl", "rb"))
logit_correct = pickle.load(open("cot_weights/arithmetic_correct_rank_results_16.pkl", "rb"))
logit_the = pickle.load(open("cot_weights/arithmetic_the_rank_results_16.pkl", "rb"))

def compute_average_ranks(data, L):
    results = [0 for _ in range(L)]
    for row in data:
        for i in range(L):
            results[i] += row[i]
    return [r / len(data) for r in results]

# NEW: std across rows for each position (ddof=1)
def compute_std_ranks(data, L):
    arr = np.array(data, dtype=float)  # shape: (n_rows, L)
    return np.std(arr, axis=0, ddof=1).tolist()

# NEW: convert relative error (std/mean) to bounds symmetric in log-space
def relative_to_log_symmetric_bounds(y, rel_err, eps=1e-12):
    y = np.asarray(y, dtype=float)
    rel = np.asarray(rel_err, dtype=float)
    rel = np.where(y > eps, rel, 0.0)          # safety for tiny y
    dz = 0.434 * rel                            # δz ≈ 0.434 * δy/y
    factor = 10.0 ** dz
    y_lower = y / factor                        # lower bound
    y_upper = y * factor                        # upper bound
    return y_lower, y_upper

def rel_err(std, mean, eps=1e-12):
    std = np.asarray(std, dtype=float); mean = np.asarray(mean, dtype=float)
    return np.divide(std, np.maximum(mean, eps))

# Logit lens: select three blocks
logit_len = 68
logit_inter_avg = compute_average_ranks(logit_inter, logit_len)
logit_correct_avg = compute_average_ranks(logit_correct, logit_len)
logit_the_avg = compute_average_ranks(logit_the, logit_len)

# NEW: std per position
logit_inter_std = compute_std_ranks(logit_inter, logit_len)
logit_correct_std = compute_std_ranks(logit_correct, logit_len)
logit_the_std = compute_std_ranks(logit_the, logit_len)

# R1-style (i where i%4==2, no offset)
logit_block1 = [logit_inter_avg[i] for i in range(64) if i % 4 == 2]
logit_block1_correct = [logit_correct_avg[i] for i in range(64) if i % 4 == 2]
logit_block1_the = [logit_the_avg[i] for i in range(64) if i % 4 == 2]
# NEW: std selections to match
logit_block1_std = [logit_inter_std[i] for i in range(64) if i % 4 == 2]
logit_block1_correct_std = [logit_correct_std[i] for i in range(64) if i % 4 == 2]
logit_block1_the_std = [logit_the_std[i] for i in range(64) if i % 4 == 2]

# R2-style (+1)
logit_block2 = [logit_inter_avg[i + 1] for i in range(64) if i % 4 == 2]
logit_block2_correct = [logit_correct_avg[i + 1] for i in range(64) if i % 4 == 2]
logit_block2_the = [logit_the_avg[i + 1] for i in range(64) if i % 4 == 2]
# NEW: std selections to match
logit_block2_std = [logit_inter_std[i + 1] for i in range(64) if i % 4 == 2]
logit_block2_correct_std = [logit_correct_std[i + 1] for i in range(64) if i % 4 == 2]
logit_block2_the_std = [logit_the_std[i + 1] for i in range(64) if i % 4 == 2]

# R4-style (+3)
logit_block4 = [logit_inter_avg[i + 3] for i in range(64) if i % 4 == 2]
logit_block4_correct = [logit_correct_avg[i + 3] for i in range(64) if i % 4 == 2]
logit_block4_the = [logit_the_avg[i + 3] for i in range(64) if i % 4 == 2]
# NEW: std selections to match
logit_block4_std = [logit_inter_std[i + 3] for i in range(64) if i % 4 == 2]
logit_block4_correct_std = [logit_correct_std[i + 3] for i in range(64) if i % 4 == 2]
logit_block4_the_std = [logit_the_std[i + 3] for i in range(64) if i % 4 == 2]

# --- Plotting side-by-side ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
x = np.arange(1, 17)

# --- R1 panel with shaded relative-SD bands ---
y = np.array(logit_block1); r = rel_err(logit_block1_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[0].plot(x, y, label="Intermediate Token")
axes[0].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(logit_block1_correct); r = rel_err(logit_block1_correct_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[0].plot(x, y, label="Final Token")
axes[0].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(logit_block1_the); r = rel_err(logit_block1_the_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[0].plot(x, y, label="Random Token: 'the'")
axes[0].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

axes[0].set_title("Logit Lens at $R_1$", fontsize=24)
axes[0].set_xlabel("Recurrent Steps", fontsize=21)
axes[0].set_ylabel("Rank", fontsize=21)
axes[0].set_yscale("log")
axes[0].tick_params(labelsize=20)

# --- R2 panel with shaded bands ---
y = np.array(logit_block2); r = rel_err(logit_block2_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[1].plot(x, y, label="Intermediate Token")
axes[1].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(logit_block2_correct); r = rel_err(logit_block2_correct_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[1].plot(x, y, label="Final Token")
axes[1].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(logit_block2_the); r = rel_err(logit_block2_the_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[1].plot(x, y, label="Random Token: 'the'")
axes[1].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

axes[1].set_title("Logit Lens at $R_2$", fontsize=24)
axes[1].set_xlabel("Recurrent Steps", fontsize=21)
axes[1].set_yscale("log")
axes[1].tick_params(labelsize=20)

# --- R4 panel with shaded bands ---
y = np.array(logit_block4); r = rel_err(logit_block4_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[2].plot(x, y, label="Intermediate Token")
axes[2].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(logit_block4_correct); r = rel_err(logit_block4_correct_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[2].plot(x, y, label="Final Token")
axes[2].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(logit_block4_the); r = rel_err(logit_block4_the_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[2].plot(x, y, label="Random Token: 'the'")
axes[2].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

axes[2].set_title("Logit Lens at $R_4$", fontsize=24)
axes[2].set_xlabel("Recurrent Steps", fontsize=21)
axes[2].set_yscale("log")
axes[2].tick_params(labelsize=20)

for ax in axes:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

# Shared Legend
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=21,
                    frameon=True, handlelength=1.0, borderpad=0.3, columnspacing=0.6)
legend.get_frame().set_linewidth(2.5)

plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave space for top legend
plt.savefig("graphs/appendix_logit.png")
