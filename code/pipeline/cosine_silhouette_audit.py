"""Cosine silhouette diagnostics on the frozen, L2-normalized PCA-32 vectors."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances, silhouette_score


root = Path(__file__).resolve().parents[2]
out = root / "results/modal/cosine_silhouette"
out.mkdir(parents=True, exist_ok=True)
z = np.load(root / "results/modal/z_final_v4.npy")
a = pd.read_csv(root / "results/modal/final_v4/assignments.csv", usecols=["set", "cluster"])
main = a["set"].eq("main").to_numpy()
zm = z[main]
k16 = a.loc[main, "cluster"].to_numpy(dtype=int)
k17 = pd.read_csv(root / "results/modal/k17_selective_split/main_assignments.csv",
                  usecols=["k17_global"]).k17_global.to_numpy(dtype=int)
automatic = np.load(root / "results/modal/auto_split/main_labels.npy")
ward16 = AgglomerativeClustering(16, linkage="ward").fit_predict(zm)
assert zm.shape == (3792, 32)
assert np.allclose(np.linalg.norm(zm, axis=1), 1, atol=1e-5)
assert len(k16) == len(k17) == len(automatic) == len(zm)

d = pairwise_distances(zm, metric="cosine")
global_rows = []
for name, labels in (("K16", k16), ("K17_global", k17),
                     ("automatic_split_C4", automatic), ("Ward_K16", ward16)):
    global_rows.append(dict(method=name, K=int(np.unique(labels).size),
                            silhouette_cosine=float(silhouette_score(d, labels, metric="precomputed")),
                            silhouette_euclidean=float(silhouette_score(zm, labels))))
pd.DataFrame(global_rows).to_csv(out / "global.csv", index=False)

previous = pd.read_csv(root / "results/modal/auto_split/candidates.csv").set_index("parent")
local_rows = []
for parent in range(16):
    x = zm[k16 == parent]
    if len(x) < 100:
        continue
    split = KMeans(2, random_state=0, n_init=20).fit_predict(x)
    sizes = np.bincount(split, minlength=2)
    if sizes.min() < max(30, int(np.ceil(.1 * len(x)))):
        continue
    local_rows.append(dict(parent=parent, n=len(x),
                           silhouette_cosine=float(silhouette_score(x, split, metric="cosine")),
                           silhouette_euclidean=float(previous.loc[parent, "silhouette"]),
                           bootstrap_ari=float(previous.loc[parent, "bootstrap_ari"])))
local = pd.DataFrame(local_rows)
local["diagnostic_cosine_x_stability"] = local.silhouette_cosine * local.bootstrap_ari
local.to_csv(out / "local.csv", index=False)
print(pd.DataFrame(global_rows).round(4).to_string(index=False))
print(local.sort_values("diagnostic_cosine_x_stability", ascending=False).round(4).to_string(index=False))
