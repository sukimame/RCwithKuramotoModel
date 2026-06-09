import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── データ読み込み ────────────────────────────────────────────────
df = pd.read_csv("to_csv_out_June7_2.csv")
Ls = [1, 2, 5, 10, 20, 50, 100, 200]

# ── カラーマップ（カスタム） ──────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    "nrmse",
    ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#f5a623"],
    N=256,
)

# ── スタイル設定 ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "monospace",
    "axes.facecolor":    "#0d0d0d",
    "figure.facecolor":  "#0d0d0d",
    "text.color":        "#e0e0e0",
    "axes.labelcolor":   "#a0a0a0",
    "xtick.color":       "#606060",
    "ytick.color":       "#606060",
    "axes.edgecolor":    "#2a2a2a",
    "axes.linewidth":    0.5,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
})

VMIN, VMAX = 0, 1.2

# ── レイアウト ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor="#0d0d0d")

# タイトル行 + ヒートマップ行 + カラーバー行
outer = gridspec.GridSpec(
    3, 1,
    figure=fig,
    height_ratios=[0.12, 1, 0.06],
    hspace=0.18,
)

# ヒートマップ用グリッド（2行×4列）
inner = gridspec.GridSpecFromSubplotSpec(
    2, 4,
    subplot_spec=outer[1],
    hspace=0.55,
    wspace=0.30,
)

axes = [[fig.add_subplot(inner[r, c]) for c in range(4)] for r in range(2)]

# ── ヒートマップ描画 ──────────────────────────────────────────────
for ax, L in zip([ax for row in axes for ax in row], Ls):
    tmp    = df[df["L"] == L]
    pivot  = tmp.pivot(index="lambda", columns="sd", values="NRMSE")
    best   = tmp.loc[tmp["NRMSE"].idxmin()]

    sns.heatmap(
        pivot,
        ax=ax,
        annot=False,
        cmap=cmap,
        vmin=VMIN,
        vmax=VMAX,
        cbar=False,
        linewidths=0,
        xticklabels="auto",
        yticklabels="auto",
    )

    # タイトル
    ax.set_title(
        f"L = {L}",
        fontsize=9,
        fontweight="bold",
        color="#c8c8c8",
        pad=4,
    )

    # ベスト値をヒートマップ右上に注記
    ax.text(
        0.98, 0.97,
        f"best {best['NRMSE']:.3f}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=6.5,
        color="#f5a623",
        bbox=dict(boxstyle="round,pad=0.25", fc="#0d0d0d", ec="#f5a623", lw=0.6, alpha=0.85),
    )

    ax.set_xlabel("sd",     fontsize=7, labelpad=2)
    ax.set_ylabel("lambda", fontsize=7, labelpad=2)
    ax.tick_params(axis="both", length=2, pad=2)

# ── タイトル ──────────────────────────────────────────────────────
title_ax = fig.add_subplot(outer[0])
title_ax.axis("off")
title_ax.text(
    0.5, 0.55,
    "NRMSE  Heatmaps",
    transform=title_ax.transAxes,
    ha="center", va="center",
    fontsize=18, fontweight="bold",
    color="#e0e0e0",
    fontfamily="monospace",
)
title_ax.text(
    0.5, 0.05,
    "Normalized Root Mean Square Error  ·  L ∈ {1, 2, 5, 10, 20, 50, 100, 200}",
    transform=title_ax.transAxes,
    ha="center", va="center",
    fontsize=8,
    color="#606060",
)

# ── 共通カラーバー ────────────────────────────────────────────────
cbar_ax = fig.add_subplot(outer[2])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("NRMSE", fontsize=8, color="#a0a0a0")
cbar.ax.tick_params(labelsize=7, colors="#606060", length=3)
cbar.outline.set_edgecolor("#2a2a2a")

# ── 保存 & 表示 ───────────────────────────────────────────────────
plt.savefig(
    "nrmse_heatmaps.png",
    dpi=180,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.show()
print("Saved → nrmse_heatmaps_June7_2.png")