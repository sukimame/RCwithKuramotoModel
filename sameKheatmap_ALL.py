import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

# ── データ読み込み ────────────────────────────────────────────────
df = pd.read_csv("to_csv_out_June7_2.csv")
Ls = [1, 2, 5, 10, 20, 50, 100, 200]

# ── カラーマップ ──────────────────────────────────────────────────
cmap_nrmse = LinearSegmentedColormap.from_list(
    "nrmse",
    ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#f5a623"],
    N=256,
)
cmap_r1   = "plasma"
cmap_wout = "coolwarm"  # 0をまたぐので発散型

# ── スタイル設定 ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "monospace",
    "axes.facecolor": "#0d0d0d",
    "figure.facecolor": "#0d0d0d",
    "text.color":     "#e0e0e0",
    "axes.labelcolor": "#a0a0a0",
    "xtick.color":    "#606060",
    "ytick.color":    "#606060",
    "axes.edgecolor": "#2a2a2a",
    "axes.linewidth": 0.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# ── ヘルパー ──────────────────────────────────────────────────────
def to_edges(v):
    v = np.asarray(v, dtype=float)
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])

def draw_heatmap(axes_row, col_key, cmap, norm, label):
    """1行分（8パネル）のヒートマップを描画"""
    for ax, L in zip(axes_row, Ls):
        tmp   = df[df["L"] == L]
        pivot = tmp.pivot(index="lambda", columns="sd", values=col_key)
        best  = tmp.loc[tmp[col_key].idxmin()] if col_key == "NRMSE" else None

        lambda_vals  = pivot.index.values.astype(float)
        lambda_edges = to_edges(lambda_vals)
        sd_vals      = np.arange(len(pivot.columns))
        sd_edges     = to_edges(sd_vals)

        ax.pcolormesh(
            sd_edges, lambda_edges, pivot.values,
            cmap=cmap, norm=norm, shading="flat",
        )

        ax.set_xticks(sd_vals)
        ax.set_xticklabels([str(v) for v in pivot.columns], rotation=90, fontsize=6)
        ax.set_yticks(lambda_vals)
        ax.set_yticklabels([str(v) for v in lambda_vals], fontsize=6)
        ax.set_xlabel("sd", fontsize=7, labelpad=2)
        ax.set_ylabel("λ",  fontsize=7, labelpad=2)
        ax.tick_params(axis="both", length=2, pad=2)

        ax.set_title(f"L = {L}", fontsize=9, fontweight="bold", color="#c8c8c8", pad=4)

        if best is not None:
            ax.text(
                0.98, 0.97, f"best {best['NRMSE']:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="#f5a623",
                bbox=dict(boxstyle="round,pad=0.25", fc="#0d0d0d", ec="#f5a623", lw=0.6, alpha=0.85),
            )

# ── レイアウト：3行（NRMSE / R1 / wout）────────────────────────────
fig = plt.figure(figsize=(20, 18), facecolor="#0d0d0d")

outer = gridspec.GridSpec(
    7, 1, figure=fig,
    height_ratios=[0.06, 1, 0.04, 1, 0.04, 1, 0.04],
    hspace=0.05,
)

def make_inner(subplot_spec):
    return gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=subplot_spec, hspace=0.6, wspace=0.3,
    )

inner_nrmse = make_inner(outer[1])
inner_r1    = make_inner(outer[3])
inner_wout  = make_inner(outer[5])

def get_axes(inner):
    return [[fig.add_subplot(inner[r, c]) for c in range(4)] for r in range(2)]

axes_nrmse = get_axes(inner_nrmse)
axes_r1    = get_axes(inner_r1)
axes_wout  = get_axes(inner_wout)

flat = lambda axes: [ax for row in axes for ax in row]

# ── 各行を描画 ────────────────────────────────────────────────────
draw_heatmap(flat(axes_nrmse), "NRMSE",
             cmap_nrmse, plt.Normalize(vmin=0, vmax=1.2),   "NRMSE")

draw_heatmap(flat(axes_r1),    "R1",
             cmap_r1,    plt.Normalize(vmin=0, vmax=1.0),   "R1")

wout_abs = df["median"].abs().max()
draw_heatmap(flat(axes_wout),  "median",
             cmap_wout,  TwoSlopeNorm(vcenter=0, vmin=-wout_abs, vmax=wout_abs), "wout median")

# ── 行ラベル ──────────────────────────────────────────────────────
for ax_row, label in zip([flat(axes_nrmse), flat(axes_r1), flat(axes_wout)],
                         ["NRMSE", "R1  (order parameter)", "wout  median"]):
    ax_row[0].set_ylabel(f"λ\n[{label}]", fontsize=7, labelpad=4)

# ── カラーバー ────────────────────────────────────────────────────
def add_cbar(subplot_spec, cmap, norm, label):
    cbar_ax = fig.add_subplot(subplot_spec)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(label, fontsize=8, color="#a0a0a0")
    cbar.ax.tick_params(labelsize=7, colors="#606060", length=3)
    cbar.outline.set_edgecolor("#2a2a2a")

add_cbar(outer[2], cmap_nrmse, plt.Normalize(0, 1.2), "NRMSE")
add_cbar(outer[4], cmap_r1,    plt.Normalize(0, 1.0), "R1")
add_cbar(outer[6], cmap_wout,
         TwoSlopeNorm(vcenter=0, vmin=-wout_abs, vmax=wout_abs), "wout median")

# ── タイトル ──────────────────────────────────────────────────────
title_ax = fig.add_subplot(outer[0])
title_ax.axis("off")
title_ax.text(
    0.5, 0.5, "NRMSE / R1 / wout  Heatmaps",
    transform=title_ax.transAxes, ha="center", va="center",
    fontsize=16, fontweight="bold", color="#e0e0e0", fontfamily="monospace",
)

# ── 保存 ──────────────────────────────────────────────────────────
plt.savefig("nrmse_heatmaps_all.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Saved → nrmse_heatmaps_all.png")