"""Fair, K-free and mapping-free comparison: cross-source top-1 nearest-neighbour label (PCA-32 space, cosine).
A-NN: gallery = BDC-train labeled (D_main, 10 classes); queries = each external set; PCA/CMRE fitted on D_main.
B-NN: non-BDC pool (rq_refit.pool_feats: CMRE/PCA fitted on the pool); leave-one-source-out (query source s,
      gallery = the other pool sources). Plus B clustering at fixed K=16 scored with ARI/AMI (no label mapping).
top1_micro = accuracy, top1_macro = balanced accuracy (mean per-class recall); 1000-sample bootstrap CI on macro.
Run: python code/pipeline/rq_fair.py"""
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, f1_score)
from sklearn.neighbors import NearestNeighbors
from rq_divide import ROOT, build, cmre, fuse  # must come first: sets EXP_OUT before common is imported
from common import reduce
from rq_labeled import CLS
from rq_metrics2 import partition
from rq_refit import pool_feats


def nn_scores(zg, tg, zq, tq, rng):
    p = tg[NearestNeighbors(n_neighbors=1, metric="cosine").fit(zg).kneighbors(zq, return_distance=False)[:, 0]]
    bs = [balanced_accuracy_score(tq[i], p[i]) for i in (rng.integers(0, len(tq), len(tq)) for _ in range(1000))]
    return dict(n=len(tq), top1_micro=accuracy_score(tq, p), top1_macro=balanced_accuracy_score(tq, p),
                macro_f1=f1_score(tq, p, labels=sorted(set(tq)), average="macro", zero_division=0),
                ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5))


def main():
    rows, rng = [], np.random.default_rng(0)
    S = build(); X, main, lab, src = S["X"], S["main"], S["lab"], S["src"]
    sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
    A = {"v4 (CMRE)": fuse(cmre(sig, dino, main)[0], cmre(pe, dino, main)[0], dino),
         "fusi mentah": fuse(sig, pe, dino), "DINOv3 saja": dino}
    g = main & np.isin(lab, CLS)
    for name, x in A.items():
        z = reduce(x[main], x, 32)
        for s in ("kaan", "bangla", "karan", "iliev", "shubha"):
            q = (src == s) & np.isin(lab, CLS)
            rows.append(dict(test="A-NN (galeri BDC)", method=name, set=s, **nn_scores(z[g], lab[g], z[q], lab[q], rng)))
            print(rows[-1], flush=True)
    P, t, s_, B = pool_feats()
    for name, x in B.items():
        name = {"v4 (CMRE -> Spectral)": "v4 (CMRE)", "fusi mentah -> Spectral": "fusi mentah"}.get(name, "DINOv3 saja")
        z = reduce(x[P], x, 32)[P]
        for s in ("kaan", "bangla", "karan", "iliev"):
            rows.append(dict(test="B-NN (leave-one-source-out)", method=name, set=s,
                             **nn_scores(z[s_ != s], t[s_ != s], z[s_ == s], t[s_ == s], rng)))
            print(rows[-1], flush=True)
        y = partition(x, P, 16)[0][P]
        rows.append(dict(test="B cluster K=16", method=name, set="pool", n=len(t),
                         ari=adjusted_rand_score(t, y), ami=adjusted_mutual_info_score(t, y)))
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "classic" / "fair_eval.csv", index=False)


if __name__ == "__main__":
    main()
