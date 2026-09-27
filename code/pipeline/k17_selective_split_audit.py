"""Exploratory C15-only split of the frozen K=16 final partition."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

from common import SHARED
import rq1_metrics as metrics


root = Path(__file__).resolve().parents[2]
out = root / "results/modal/k17_selective_split"
out.mkdir(parents=True, exist_ok=True)
z = np.load(root / "results/modal/z_final_v4.npy")
assign = pd.read_csv(root / "results/modal/final_v4/assignments.csv", keep_default_na=False)
main = assign["set"].eq("main").to_numpy()
base = assign.loc[main, "cluster"].to_numpy(dtype=int)
assert z.shape == (len(assign), 32) and main.sum() == 3792
assert (base == 15).sum() == 640

global17 = SpectralClustering(17, affinity="nearest_neighbors", n_neighbors=15,
                              random_state=0, assign_labels="cluster_qr").fit_predict(z[main])
assert sorted(np.unique(global17[base == 15]).tolist()) == [11, 16]

# Preserve every original C0-C14 assignment. C15 alone gains a second group.
restricted = base.copy()
restricted[(base == 15) & (global17 == 11)] = 16

local = base.copy()
local_split = KMeans(2, random_state=0, n_init=20).fit_predict(z[main][base == 15])
pcb_group = np.argmax([((local_split == c) & (global17[base == 15] == 16)).sum()
                       for c in (0, 1)])
local[np.flatnonzero(base == 15)[local_split != pcb_group]] = 16

lab = assign["posthoc_label"].to_numpy()
src = assign["source"].to_numpy()
style = assign["style"].to_numpy()
cat = main & (lab != "")
shared = (~main) & np.isin(lab, SHARED) & np.isin(src, ["kaan", "bangla", "karan"])
external = (~main) & (lab != "")
detail = []


def score(name, labels):
    centers = np.stack([z[main][labels == c].mean(axis=0) for c in np.unique(labels)])
    y = ((z[:, None] - centers[None]) ** 2).sum(axis=-1).argmin(axis=1)
    y[main] = labels
    s = metrics.score_labels(y, main, style, lab, src, cat, shared)
    for category in SHARED:
        catalog = shared & (src == "kaan") & (lab == category)
        phone = shared & (src == "bangla") & (lab == category)
        if catalog.any() and phone.any():
            majority = np.bincount(y[catalog]).argmax()
            detail.append(dict(scenario=name, category=category, catalog_n=int(catalog.sum()),
                               catalog_majority_cluster=int(majority),
                               catalog_majority_n=int((y[catalog] == majority).sum()),
                               phone_n=int(phone.sum()), phone_match=int((y[phone] == majority).sum()),
                               phone_match_rate=float((y[phone] == majority).mean())))
    split_sizes = (f"mixed={int((labels == 11).sum())}; PCB={int((labels == 16).sum())}"
                   if name == "K17_global" else
                   f"C15={int((labels == 15).sum())}; new={int((labels == 16).sum())}")
    s.update(scenario=name, moved_from_other=int(((base != 15) & (labels == 11)).sum())
             if name == "K17_global" else 0,
             changed_main=int((labels != base).sum()) if name != "K17_global" else None,
             ari_vs_k16=adjusted_rand_score(base, labels),
             purity_bdc=pd.crosstab(y[cat], lab[cat]).max(axis=1).sum() / cat.sum(),
             purity_external=pd.crosstab(y[external], lab[external]).max(axis=1).sum() / external.sum(),
             split_sizes=split_sizes)
    return s


rows = [score("K16", base), score("K17_global", global17),
        score("K17_C15_only_from_global", restricted), score("K17_C15_only_kmeans", local)]
pd.DataFrame(rows).set_index("scenario").to_csv(out / "comparison.csv")
pd.DataFrame(detail).to_csv(out / "xsrc_by_class.csv", index=False)
pd.DataFrame({"id": assign.loc[main, "id"].to_numpy(), "k16": base,
              "k17_global": global17, "k17_c15_only": restricted,
              "k17_c15_kmeans": local}).to_csv(out / "main_assignments.csv", index=False)
print(pd.DataFrame(rows).set_index("scenario").round(4).to_string())
