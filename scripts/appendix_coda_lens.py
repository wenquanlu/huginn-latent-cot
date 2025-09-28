import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np  # NEW
import matplotlib.ticker as ticker

rcParams['axes.linewidth'] = 1.2  # All subplot borders (left, bottom, top, right)
rcParams['lines.linewidth'] = 2.3

# --- Load Coda Lens data ---
coda_inter = pickle.load(open("cot_weights/coda_arithmetic_inter_rank_16.pkl", "rb"))
coda_correct = pickle.load(open("cot_weights/coda_arithmetic_correct_rank_16.pkl", "rb"))
coda_the = pickle.load(open("cot_weights/coda_arithmetic_the_rank_16.pkl", "rb"))

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

# Coda lens: select three blocks by modulo group (0,1,2)
coda_len = 64
coda_inter_avg = compute_average_ranks(coda_inter, coda_len)
coda_correct_avg = compute_average_ranks(coda_correct, coda_len)
coda_the_avg = compute_average_ranks(coda_the, coda_len)

# NEW: std per position
coda_inter_std = compute_std_ranks(coda_inter, coda_len)
coda_correct_std = compute_std_ranks(coda_correct, coda_len)
coda_the_std = compute_std_ranks(coda_the, coda_len)

# Block R1: i % 4 == 0
coda_block1 = [coda_inter_avg[i] for i in range(64) if i % 4 == 0]
coda_block1_correct = [coda_correct_avg[i] for i in range(64) if i % 4 == 0]
coda_block1_the = [coda_the_avg[i] for i in range(64) if i % 4 == 0]
# NEW: matching std selections
coda_block1_std = [coda_inter_std[i] for i in range(64) if i % 4 == 0]
coda_block1_correct_std = [coda_correct_std[i] for i in range(64) if i % 4 == 0]
coda_block1_the_std = [coda_the_std[i] for i in range(64) if i % 4 == 0]

# Block R2: i % 4 == 1
coda_block2 = [coda_inter_avg[i] for i in range(64) if i % 4 == 1]
coda_block2_correct = [coda_correct_avg[i] for i in range(64) if i % 4 == 1]
coda_block2_the = [coda_the_avg[i] for i in range(64) if i % 4 == 1]
# NEW: matching std selections
coda_block2_std = [coda_inter_std[i] for i in range(64) if i % 4 == 1]
coda_block2_correct_std = [coda_correct_std[i] for i in range(64) if i % 4 == 1]
coda_block2_the_std = [coda_the_std[i] for i in range(64) if i % 4 == 1]

# Block R3: i % 4 == 2
coda_block3 = [coda_inter_avg[i] for i in range(64) if i % 4 == 2]
coda_block3_correct = [coda_correct_avg[i] for i in range(64) if i % 4 == 2]
coda_block3_the = [coda_the_avg[i] for i in range(64) if i % 4 == 2]
# NEW: matching std selections
coda_block3_std = [coda_inter_std[i] for i in range(64) if i % 4 == 2]
coda_block3_correct_std = [coda_correct_std[i] for i in range(64) if i % 4 == 2]
coda_block3_the_std = [coda_the_std[i] for i in range(64) if i % 4 == 2]

# --- Plotting side-by-side ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
x = np.arange(1, 17)

# Panel R1 with shaded relative-SD bands
y = np.array(coda_block1); r = rel_err(coda_block1_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[0].plot(x, y, label="Intermediate Token")
axes[0].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(coda_block1_correct); r = rel_err(coda_block1_correct_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[0].plot(x, y, label="Final Token")
axes[0].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(coda_block1_the); r = rel_err(coda_block1_the_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[0].plot(x, y, label="Random Token: 'the'")
axes[0].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

axes[0].set_title("Coda Lens at $R_1$", fontsize=24)
axes[0].set_xlabel("Recurrent Steps", fontsize=21)
axes[0].set_ylabel("Rank", fontsize=21)
axes[0].set_yscale("log")
axes[0].tick_params(labelsize=20)

# Panel R2
y = np.array(coda_block2); r = rel_err(coda_block2_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[1].plot(x, y, label="Intermediate Token")
axes[1].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(coda_block2_correct); r = rel_err(coda_block2_correct_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[1].plot(x, y, label="Final Token")
axes[1].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(coda_block2_the); r = rel_err(coda_block2_the_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[1].plot(x, y, label="Random Token: 'the'")
axes[1].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

axes[1].set_title("Coda Lens at $R_2$", fontsize=24)
axes[1].set_xlabel("Recurrent Steps", fontsize=21)
axes[1].set_ylabel("Rank", fontsize=21)
axes[1].set_yscale("log")
axes[1].tick_params(labelsize=20)

# Panel R3
y = np.array(coda_block3); r = rel_err(coda_block3_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[2].plot(x, y, label="Intermediate Token")
axes[2].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(coda_block3_correct); r = rel_err(coda_block3_correct_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[2].plot(x, y, label="Final Token")
axes[2].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

y = np.array(coda_block3_the); r = rel_err(coda_block3_the_std, y); lo, hi = relative_to_log_symmetric_bounds(y, r)
line, = axes[2].plot(x, y, label="Random Token: 'the'")
axes[2].fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)

axes[2].set_title("Coda Lens at $R_3$", fontsize=24)
axes[2].set_xlabel("Recurrent Steps", fontsize=21)
axes[2].set_ylabel("Rank", fontsize=21)
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
plt.savefig("graphs/appendix_coda.png")
