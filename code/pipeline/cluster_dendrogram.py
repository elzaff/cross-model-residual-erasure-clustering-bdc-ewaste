"""Dendrogram of the 16 final cluster centroids (average linkage, cosine distance) on the frozen CMRE/PCA-32 features.
Post-hoc reading aid only: the partition itself is the Spectral K=16 result. Leaves are coloured by the WEEE / B3
category assigned after gallery inspection (Tabel interpretasi). Writes paper_draft/fig/fig_dendrogram.png."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[2]
z = np.load(ROOT / "results/modal/z_final_v4.npy")
a = pd.read_csv(ROOT / "results/modal/final_v4/assignments.csv", usecols=["set", "cluster"])
main = a.set.eq("main").to_numpy(); lab = a.cluster.to_numpy()[main]; zm = z[main]
assert zm.shape == (3792, 32) and sorted(set(lab)) == list(range(16))
cent = np.vstack([zm[lab == c].mean(0) for c in range(16)])
n = np.bincount(lab, minlength=16)
L = linkage(pdist(cent, "cosine"), method="average")

NAME = ["Aki", "Baterai laptop", "Turntable", "TV/monitor CRT", "Baterai ponsel/kamera", "Pemutar CD/radio",
        "Mesin cuci bukaan atas", "TV/monitor layar datar", "Mesin cuci front-load", "Laptop", "Keyboard", "Microwave",
        "Printer", "Mouse", "Ponsel", "PCB dan tumpukan campuran"]
CAT = ["B3", "B3", "5", "2", "B3", "5", "4", "2", "4", "2", "6", "5", "6", "6", "6", "K"]
COL = {"B3": ("#c62828", "Baterai (B3)"), "2": ("#1565c0", "WEEE 2: layar dan monitor"),
       "4": ("#2e7d32", "WEEE 4: peralatan besar"), "5": ("#ef6c00", "WEEE 5: peralatan kecil"),
       "6": ("#6a1b9a", "WEEE 6: TI dan telekomunikasi kecil"), "K": ("#616161", "Komponen / campuran")}

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False})
W = 15 / 2.54
fig, ax = plt.subplots(figsize=(W, W * 0.62))
dendrogram(L, orientation="right", labels=[f"C{c} {NAME[c]} (n={n[c]})" for c in range(16)],
           color_threshold=0, above_threshold_color="#555555", leaf_font_size=8, ax=ax)
for t in ax.get_yticklabels():
    c = int(t.get_text().split()[0][1:]); t.set_color(COL[CAT[c]][0])
ax.set_xlabel("Jarak kosinus antarpusat kelompok (average linkage)")
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=v[0], label=v[1]) for v in COL.values()], loc="lower left", bbox_to_anchor=(0, 1.0),
          ncol=3, fontsize=7, frameon=False)
fig.savefig(ROOT / "paper_draft/fig/fig_dendrogram.png", dpi=300, bbox_inches="tight", facecolor="white")


def members(x):
    x = int(x); return [x] if x < 16 else members(L[x - 16, 0]) + members(L[x - 16, 1])


for k, (i, j, h, _) in enumerate(L[:6]):  # first merges, to check the text in the Penafsiran section
    print(f"merge {k + 1}: {[NAME[m] for m in members(i)]} + {[NAME[m] for m in members(j)]} at {h:.3f}")
