import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── データ読み込み ────────────────────────────────────────────────
df = pd.read_csv("csvData/to_csv_out_June10_4.csv")

# ── カラーマップ ──────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    "nrmse",
    ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#f5a623"],
    N=256,
)

# ── スタイル設定 ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "monospace",
    "axes.facecolor":   "#0d0d0d",
    "figure.facecolor": "#0d0d0d",
    "text.color":       "#e0e0e0",
    "axes.labelcolor":  "#a0a0a0",
    "xtick.color":      "#606060",
    "ytick.color":      "#606060",
    "axes.edgecolor":   "#2a2a2a",
    "axes.linewidth":   0.5,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
})

VMIN, VMAX = 0, 1.2

# ── delay × k_scale のピボット ────────────────────────────────────
pivot = df.pivot(index="k_scale", columns="delay", values="NRMSE")
best  = df.loc[df["NRMSE"].idxmin()]

# ── 描画 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0d0d0d")

sns.heatmap(
    pivot,
    ax=ax,
    annot=False,
    cmap=cmap,
    vmin=VMIN,
    vmax=VMAX,
    linewidths=0,
    xticklabels="auto",
    yticklabels="auto",
    cbar_kws={"shrink": 0.8, "label": "NRMSE"},
)

ax.set_title(
    "NRMSE  Heatmap  —  k_scale  vs  delay",
    fontsize=14, fontweight="bold", color="#e0e0e0", pad=12,
)
ax.set_xlabel("delay (steps)", fontsize=10, labelpad=6)
ax.set_ylabel("k_scale",       fontsize=10, labelpad=6)
ax.tick_params(axis="both", length=2, pad=2)

# ベスト値を注記
ax.text(
    0.99, 0.97,
    f"best NRMSE {best['NRMSE']:.4f}  (delay={best['delay']:.0f}, k_scale={best['k_scale']})",
    transform=ax.transAxes,
    ha="right", va="top", fontsize=8, color="#f5a623",
    bbox=dict(boxstyle="round,pad=0.3", fc="#0d0d0d", ec="#f5a623", lw=0.7, alpha=0.9),
)

# カラーバーのスタイル
cbar = ax.collections[0].colorbar
cbar.set_label("NRMSE", fontsize=9, color="#a0a0a0")
cbar.ax.tick_params(labelsize=8, colors="#606060")
cbar.outline.set_edgecolor("#2a2a2a")

fig.text(
    0.5, 0.01,
    "Normalized Root Mean Square Error  ·  delay ∈ {1, …, 20}  ·  ω uniform (sd=0)",
    ha="center", fontsize=7, color="#606060",
)

plt.tight_layout()
plt.savefig(
    "nrmse_heatmaps_June9_3.png",
    dpi=180,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.show()
print("Saved → nrmse_heatmaps_June9_4.png")