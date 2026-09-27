"""Select one K=16 cluster for an unsupervised local split, without evaluation labels.

Input: frozen CMRE PCA-32 vectors and final K=16 assignments on D_main.
Rule fixed before execution: among parents with >=100 images and two children
of >=max(30, 10% of parent), maximize local silhouette * bootstrap ARI.
All parents use the same KMeans(k=2) fit and 100 bootstrap resamples.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


root = Path(__file__).resolve().parents[2]
out = root / "results/modal/auto_split"
out.mkdir(parents=True, exist_ok=True)
z = np.load(root / "results/modal/z_final_v4.npy")
base = pd.read_csv(root / "results/modal/final_v4/assignments.csv",
                   usecols=["set", "cluster"])
main = base["set"].eq("main").to_numpy()
labels = base.loc[main, "cluster"].to_numpy(dtype=int)
zm = z[main]
assert z.shape == (len(base), 32) and main.sum() == 3792
assert sorted(np.unique(labels).tolist()) == list(range(16))

rows = []
splits = {}
for parent in range(16):
    indices = np.flatnonzero(labels == parent)
    x = zm[indices]
    row = dict(parent=parent, n=len(x), eligible=False, silhouette=np.nan,
               bootstrap_ari=np.nan, score=np.nan, child_0=0, child_1=0)
    if len(x) >= 100:
        split = KMeans(2, random_state=0, n_init=20).fit_predict(x)
        sizes = np.bincount(split, minlength=2)
        row.update(child_0=int(sizes[0]), child_1=int(sizes[1]))
        if sizes.min() >= max(30, int(np.ceil(.1 * len(x)))):
            silhouette = silhouette_score(x, split)
            rng = np.random.default_rng(2026)
            stability = []
            for _ in range(100):
                sample = rng.choice(len(x), len(x), replace=True)
                fitted = KMeans(2, random_state=0, n_init=10).fit(x[sample])
                stability.append(adjusted_rand_score(split, fitted.predict(x)))
            row.update(eligible=True, silhouette=float(silhouette),
                       bootstrap_ari=float(np.mean(stability)),
                       score=float(silhouette * np.mean(stability)))
            splits[parent] = split
    rows.append(row)

table = pd.DataFrame(rows).sort_values("parent")
table.to_csv(out / "candidates.csv", index=False)
if not splits:
    raise RuntimeError("No eligible parent cluster under the frozen rule")
winner = table.loc[table.score.idxmax()]
parent = int(winner.parent)
selected = labels.copy()
child = splits[parent]
new_child = int(np.argmin(np.bincount(child, minlength=2)))
selected[np.flatnonzero(labels == parent)[child == new_child]] = 16
assert (selected != labels).sum() == (child == new_child).sum()
assert np.all(selected[labels != parent] == labels[labels != parent])
np.save(out / "main_labels.npy", selected)
print(table.round(4).to_string(index=False))
print(f"SELECTED parent={parent}, new_group_n={(selected == 16).sum()}, "
      f"silhouette={winner.silhouette:.4f}, bootstrap_ari={winner.bootstrap_ari:.4f}")
