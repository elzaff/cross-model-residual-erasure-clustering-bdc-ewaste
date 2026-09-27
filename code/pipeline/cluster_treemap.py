"""Two separate figures of the final K=16 partition on the 3,792 BDC images (sans-serif, 15 cm wide):
fig_cluster_map.png     t-SNE map (results/modal/r8/bdc_tsne.csv) coloured by WEEE/B3 category, cluster ids at medians
fig_cluster_treemap.png two-level treemap: category strips (width = images in category) split into clusters (height = size)
Names and categories as in the interpretation table (assigned after gallery inspection); slice-and-dice layout is
enough for 6 strips and 16 boxes."""
from pathlib import Path
import textwrap
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]; FIG = ROOT / "paper_draft/fig"
a = pd.read_csv(ROOT / "results/modal/final_v4/assignments.csv", usecols=["set", "cluster"])
n = np.bincount(a.loc[a.set.eq("main"), "cluster"].to_numpy(), minlength=16)
ts = pd.read_csv(ROOT / "results/modal/r8/bdc_tsne.csv")
assert n.sum() == 3792 == len(ts) and np.array_equal(np.bincount(ts.cluster, minlength=16), n)

NAME = ["Aki", "Baterai laptop", "Turntable", "TV/monitor CRT", "Baterai ponsel/kamera", "Pemutar CD/radio",
        "Mesin cuci bukaan atas", "TV/monitor layar datar", "Mesin cuci front-load", "Laptop", "Keyboard", "Microwave",
        "Printer", "Mouse", "Ponsel", "PCB dan tumpukan campuran"]
CAT = ["B3", "B3", "5", "2", "B3", "5", "4", "2", "4", "2", "6", "5", "6", "6", "6", "K"]
COL = {"6": ("#6a1b9a", "WEEE 6: TI dan telekomunikasi kecil"), "2": ("#1565c0", "WEEE 2: layar dan monitor"),
       "K": ("#616161", "Komponen / campuran"), "5": ("#ef6c00", "WEEE 5: peralatan kecil"),
       "B3": ("#c62828", "Baterai (B3)"), "4": ("#2e7d32", "WEEE 4: peralatan besar")}
cats = sorted(COL, key=lambda k: -sum(n[c] for c in range(16) if CAT[c] == k))
LEGEND = [Rectangle((0, 0), 1, 1, color=COL[k][0], alpha=.9, label=f"{COL[k][1]} ({sum(n[c] for c in range(16) if CAT[c] == k)})")
          for k in cats]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
W = 15 / 2.54

# ---------------------------------------------------------------- map
fig, ax = plt.subplots(figsize=(W, W * 0.72))
ax.scatter(ts.x, ts.y, s=3, c=[COL[CAT[c]][0] for c in ts.cluster], alpha=.55, linewidths=0, rasterized=True)
for c in range(16):
    mx, my = ts.loc[ts.cluster == c, ["x", "y"]].median()
    ax.text(mx, my, f"C{c}", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#222222",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=COL[CAT[c]][0], lw=.9, alpha=.92))
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color("#cccccc")
ax.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7, frameon=False)
fig.savefig(FIG / "fig_cluster_map.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- treemap
fig, ax = plt.subplots(figsize=(W, W * 0.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
x = 0.0
for k in cats:
    members = sorted((c for c in range(16) if CAT[c] == k), key=lambda c: -n[c])
    w = sum(n[c] for c in members) / n.sum(); y = 1.0
    for c in members:
        h = n[c] / sum(n[m] for m in members); y -= h
        ax.add_patch(Rectangle((x, y), w, h, facecolor=COL[k][0], edgecolor="white", linewidth=1.5, alpha=.9))
        narrow = w < 0.1
        lines = textwrap.wrap(NAME[c].replace("/", "/ "), max(7, int(w * 80)), break_long_words=False)  # whole words
        lines = [s.replace("/ ", "/") for s in lines]
        ax.text(x + w / 2, y + h / 2 + 0.01, f"C{c}", ha="center", va="bottom", color="white",
                fontsize=7 if not narrow else 6.2, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 + 0.006, "\n".join(lines) + f"\n{n[c]} foto", ha="center", va="top",
                color="white", fontsize=6.2 if not narrow else 5.3, linespacing=1.1)
    x += w
ax.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=7, frameon=False)
fig.savefig(FIG / "fig_cluster_treemap.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("saved fig_cluster_map.png and fig_cluster_treemap.png")
