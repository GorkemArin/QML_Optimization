import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# --------------------------------------------------------
# 1. Insert your data here
# --------------------------------------------------------
x = np.array([10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

y = np.array([316, 353, 310, 305, 356, 391, 384, 369, 544, 536, 789, 877])
y2 = np.array([157, 157, 147, 208, 189, 190, 223, 217, 207, 514, 475, 576])
y3 = np.array([92, 83, 96, 94, 107, 115, 114, 108, 118, 367, 338, 481])

y_err = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50])

window = 3  # odd number, adjust if needed
pad = window // 2

# y_padded = np.pad(y, pad_width=pad, mode='edge')
# y_err = np.array([
#     np.std(y_padded[i:i+window])
#     for i in range(len(y))
# ])

y1_s = savgol_filter(y, window_length=5, polyorder=3)
y2_s = savgol_filter(y2, window_length=5, polyorder=3)
y3_s = savgol_filter(y3, window_length=5, polyorder=3)
# --------------------------------------------------------
# 2. Create the figure
# --------------------------------------------------------
# --------------------------------------------------------
# Plot all three curves
# --------------------------------------------------------
plt.figure(figsize=(9,6))

# Curve 1
plt.plot(x, y1_s, linewidth=2, label="Epoch = 200")
plt.errorbar(
    x, y1_s,
    yerr=y_err,
    fmt='none',
    capsize=4,
    elinewidth=1,
    alpha=0.9
)


# Curve 2
plt.plot(x, y2_s, linewidth=2, label="Epoch = 100")
plt.errorbar(
    x, y2_s,
    yerr=y_err,
    fmt='none',
    capsize=4,
    elinewidth=1,
    alpha=0.9
)


# Curve 3
plt.plot(x, y3_s, linewidth=2, label="Epoch = 50")
plt.errorbar(
    x, y3_s,
    yerr=y_err,
    fmt='none',
    capsize=4,
    elinewidth=1,
    alpha=0.9
)


# --------------------------------------------------------
# Style
# --------------------------------------------------------
plt.scatter(x, y, s=25)
plt.scatter(x, y2, s=25)
plt.scatter(x, y3, s=25)

plt.xlabel("Dataset size")
plt.ylabel("Training time (ms)")
plt.title("Classical GD Training")
plt.grid(True, linestyle='--', alpha=0.6)

plt.legend()
plt.tight_layout()
plt.show()

