"""CPU-only reproduction of the final partition from the saved fused features (no GPU, no model download, < 1 min).
Inputs:  results/modal/z_final_v4.npy (CMRE fusion + PCA-32 for every row of items.csv, from pipeline/rq_final2.py)
         results/modal/items.csv, results/modal/final_v4/assignments.csv
Checks:  Spectral-kNN K=16 on D_main reproduces the submitted clusters; cross-source consistency (xsrc) on the five
         classes shared by Kaan and Bangladesh; kNN-OOD flag rates per source. Labels are read only for these checks.
Usage:   python code/reproduce.py"""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ewaste_cmre import cluster, KnnOOD

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "modal")
z = np.load(f"{R}/z_final_v4.npy"); it = pd.read_csv(f"{R}/items.csv", dtype={"id": str}, keep_default_na=False)
A = pd.read_csv(f"{R}/final_v4/assignments.csv", dtype={"id": str}); assert (A.id.values == it.id.values).all()
main = (it.set == "main").values; zm = z[main]

y_main = cluster(zm, k=16)
print(f"ARI vs submitted clusters: {adjusted_rand_score(A.cluster.values[main], y_main):.4f}")
C = np.stack([zm[y_main == c].mean(0) for c in range(16)]); y = ((z[:, None] - C[None]) ** 2).sum(-1).argmin(1); y[main] = y_main

lab, src = it.posthoc_label.values, it.source.values
per = {}
for c in ["battery", "keyboard", "mobile", "mouse", "pcb"]:
    k, b = (src == "kaan") & (lab == c), (src == "bangla") & (lab == c)
    per[c] = float((y[b] == np.bincount(y[k]).argmax()).mean())
print("xsrc per class:", {c: round(v, 3) for c, v in per.items()}, f"| mean {np.mean(list(per.values())):.3f}")

fl = np.zeros(len(z), bool); fl[~main] = KnnOOD(zm).is_unknown(z[~main]); bulb = lab == "light_bulb"  # Bangladesh lamps: not a BDC class
print("kNN-OOD flagged (false alarms on BDC classes):", {s: round(float(fl[(src == s) & ~bulb].mean()), 3) for s in ("kaan", "bangla", "karan")},
      "| Bangladesh lamps (true OOD):", round(float(fl[bulb].mean()), 3), "| GIZ field photos:", round(float(fl[src == "giz"].mean()), 3))
