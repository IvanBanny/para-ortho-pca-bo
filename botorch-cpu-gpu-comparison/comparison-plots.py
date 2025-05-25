from matplotlib import pyplot as plt
import json
import numpy as np
import pandas as pd

dpi, figsize = 200, (12, 6)  # Increased height for better visualization
width, space = 0.25, 0.15
colors = np.array([
    ['#FF8080', '#FFA3A3'],
    ['#80B3FF', '#A3CBFF'],
    ['#A3D966', '#BFEA80'],
    ['#FF9966', '#FFBB99']
])

with open("stats.json", 'r') as f:
    stats = pd.DataFrame(json.load(f))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=dpi, figsize=figsize, layout="constrained")

# Preprocess the data
stats["dtype"] = stats["dtype"].replace("torch.float32", 'fp32')
stats["dtype"] = stats["dtype"].replace("torch.float64", 'fp64')

# Calculate the average 'best' per fid for normalization
fid_avg = stats.groupby("fid")["best"].mean().to_dict()
print(f"Average 'best' per fid: {fid_avg}")

# Normalize the 'best' values
stats["best_normalized"] = stats.apply(lambda row: row["best"] / fid_avg[row["fid"]], axis=1)
print("First few rows with normalized 'best':")
print(stats[["dim", "gpu", "method", "dtype", "fid", "best", "best_normalized"]].head())

# Group the data
stats_grouped = (stats.groupby(["dim", "gpu", "method", "dtype"])
         .agg({"gpr": "mean", "acqf": "mean", "best": "mean", "best_normalized": "mean"}).reset_index())

groups = [{"method": method, "dtype": dtype, "stats": stats_grouped[(stats_grouped["method"] == method) & (stats_grouped["dtype"] == dtype)]}
          for method in stats_grouped["method"].unique()
          for dtype in stats_grouped["dtype"].unique()]

# First subplot: Time for acqf
for i, group in enumerate(groups):
    acqf = ax1.bar(np.arange(len(group["stats"])) * (len(groups[0]["stats"]) * width + space) + i * width,
                   group["stats"]["acqf"],
                   width,
                   label=f"{group['method']} - {group['dtype']}",
                   color=colors[i][0])
    ax1.bar_label(acqf, labels=["acqf"] * len(groups[0]["stats"]), label_type="center", fontsize=6)

    gpr = ax1.bar(np.arange(len(group["stats"])) * (len(groups[0]["stats"]) * width + space) + i * width,
                  group["stats"]["gpr"],
                  width,
                  bottom=group["stats"]["acqf"],
                  label=f"{group['method']} - {group['dtype']}",
                  color=colors[i][1])
    ax1.bar_label(gpr, labels=["gpr"] * len(groups[0]["stats"]), label_type="center", fontsize=6)

# Second subplot: Original best values
for i, group in enumerate(groups):
    best = ax2.bar(np.arange(len(group["stats"])) * (len(groups[0]["stats"]) * width + space) + i * width,
                   group["stats"]["best"],
                   width,
                   label=f"{group['method']} - {group['dtype']}",
                   color=colors[i][0])

# Third subplot: Normalized best values
for i, group in enumerate(groups):
    best_norm = ax3.bar(np.arange(len(group["stats"])) * (len(groups[0]["stats"]) * width + space) + i * width,
                   group["stats"]["best_normalized"],
                   width,
                   label=f"{group['method']} - {group['dtype']}",
                   color=colors[i][0])

# Set labels and titles
ax1.set_ylabel("Time (s)")
ax1.set_title("Avg run time by dim and method")
# Create x-tick labels from the actual data points
tick_positions = np.arange(len(groups[0]["stats"])) * (len(groups[0]["stats"]) * width + space) + (len(groups) - 1) * width / 2
tick_labels = [f"{row['gpu']} - {row['dim']}" for _, row in groups[0]["stats"].iterrows()]
ax1.set_xticks(tick_positions, tick_labels)
ax1.legend(loc="upper right", ncols=2)

# Original best values
best_min, best_max = stats_grouped["best"].min(), stats_grouped["best"].max()
best_margin = (best_max - best_min) / 20  # Smaller margin for better visibility
ax2.set_ylabel("Best (raw values)")
ax2.set_title("Avg best by dim and method")
# Use the same tick positions and labels for the other subplots
ax2.set_xticks(tick_positions, tick_labels)
ax3.set_xticks(tick_positions, tick_labels)
# Use logarithmic scale if range is very large
if best_max / best_min > 100:
    ax2.set_yscale('log')
    ax2.set_title("Avg best by dim and method (log scale)")

# Normalized best values
norm_min, norm_max = stats_grouped["best_normalized"].min(), stats_grouped["best_normalized"].max()
norm_margin = (norm_max - norm_min) / 10
ax3.set_ylim(norm_min - norm_margin, norm_max + norm_margin)
ax3.set_ylabel("Best (normalized)")
ax3.set_title("Normalized best by dim and method")
# Rotate x-tick labels for better readability
for ax in [ax1, ax2, ax3]:
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ax3.legend(loc="upper left", ncols=2)

# Add a horizontal line at y=1.0 for the normalized plot to show the average
ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

plt.savefig("method-comparison-plots")
plt.show()
