from matplotlib import pyplot as plt
import json
import numpy as np
import pandas as pd

dpi, figsize = 200, (12, 3)
width, space = 0.25, 0.15
colors = np.array([
    ['#FF8080', '#FFA3A3'],
    ['#80B3FF', '#A3CBFF'],
    ['#A3D966', '#BFEA80'],
    ['#FF9966', '#FFBB99']
])

with open("stats.json", 'r') as f:
    stats = pd.DataFrame(json.load(f))

fig, (ax1, ax2) = plt.subplots(1, 2, dpi=dpi, figsize=figsize, layout="constrained")

stats["dtype"] = stats["dtype"].replace("torch.float32", 'fp32')
stats["dtype"] = stats["dtype"].replace("torch.float64", 'fp64')
stats = (stats.groupby(["dim", "gpu", "method", "dtype"])
         .agg({"gpr": "mean", "acqf": "mean", "best": "mean"}).reset_index())

print(stats)

groups = [{"method": method, "dtype": dtype, "stats": stats[(stats["method"] == method) & (stats["dtype"] == dtype)]}
          for method in stats["method"].unique()
          for dtype in stats["dtype"].unique()]

for i, group in enumerate(groups):
    acqf = ax1.bar(np.arange(len(group["stats"])) * (len(groups[0]["stats"]) * width + space) + i * width,
                   group["stats"]["acqf"],
                   width,
                   label=f"{group['method']} - {group['dtype']}",
                   color=colors[i][0])
    ax1.bar_label(acqf, labels=["acqf"]*len(groups), label_type="center", fontsize=6)

    gpr = ax1.bar(np.arange(len(group["stats"])) * (len(groups[0]["stats"]) * width + space) + i * width,
                  group["stats"]["gpr"],
                  width,
                  bottom=group["stats"]["acqf"],
                  label=f"{group['method']} - {group['dtype']}",
                  color=colors[i][1])
    ax1.bar_label(gpr, labels=["gpr"] * len(groups), label_type="center", fontsize=6)

    best = ax2.bar(np.arange(len(group["stats"])) * (len(groups[0]["stats"]) * width + space) + i * width,
                   group["stats"]["best"],
                   width,
                   label=f"{group['method']} - {group['dtype']}",
                   color=colors[i][0])

ax1.set_ylabel("Time (s)")
ax1.set_title("Avg run time by dim and method")
ax1.set_xticks(np.arange(len(groups)) * (len(groups[0]["stats"]) * width + space) + (len(groups[0]["stats"]) - 1) * width / 2,
               [f"{gpu} - {dim}" for dim in stats["dim"].unique() for gpu in stats["gpu"].unique()])
ax1.legend(loc="upper right", ncols=2)

best_min, best_max = stats["best"].min(), stats["best"].max()
best_margin = (best_max - best_min) / 2
ax2.set_ylim(best_min - best_margin, best_max + best_margin)
ax2.set_ylabel("Best")
ax2.set_title("Avg best by dim and method")
ax2.set_xticks(np.arange(len(groups)) * (len(groups[0]["stats"]) * width + space) + (len(groups[0]["stats"]) - 1) * width / 2,
               [f"{gpu} - {dim}" for dim in stats["dim"].unique() for gpu in stats["gpu"].unique()])
ax2.legend(loc="upper left", ncols=2)

plt.savefig("method-comparison-plots")
plt.show()
