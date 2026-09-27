"""CTGL (Liu et al., TNNLS 2025; ctgl.py = port of the official MATLAB code) on BDC D_main, 3 views, K=16.
Hyper-parameters fixed a priori from the authors' Demo_CTGL_loop.m final call (alpha = lambda = 0.5, k = 15, pho = 1,
maxIter = 5); the paper tunes alpha/lambda with labels, which is not possible here. CTGL is transductive, so external,
Iliev and Shubha images get the nearest D_main centroid in the PCA-32 space of the fused views (same rule as every
other baseline). Views: raw [SigLIP2, PE-Core, DINOv3] vs CMRE [CMRE(SigLIP2), CMRE(PE-Core), DINOv3].
Run: python code/pipeline/rq_ctgl.py"""
import os, time, numpy as np, pandas as pd
from rq_divide import ROOT, build, cmre, fuse, score_all  # same stacked data, SHA check and 5-layer scoring
from common import reduce
from ctgl import CTGL

OUT = ROOT / "results" / "classic" / "ctgl_results.csv"
K, CFG = 16, dict(k=15, alpha=0.5, lam=0.5, pho=1, maxIter=5)


def main():
    S = build(); X, main = S["X"], S["main"]; sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
    views = {"mentah": [sig, pe, dino], "CMRE": [cmre(sig, dino, main)[0], cmre(pe, dino, main)[0], dino]}
    rows = []
    for tag in os.environ.get("CTGL_VIEWS", "CMRE,mentah").split(","):
        vv = views[tag]; t = time.time()
        lf = ROOT / "results" / "classic" / f"ctgl_labels_{tag}.npy"  # written by modal_ctgl.py
        ym = np.load(lf) if lf.exists() else CTGL([v[main].T.astype(float) for v in vv], K, **CFG)[0]
        z = reduce(fuse(*vv)[main], fuse(*vv), 32); zm = z[main]
        C = np.stack([zm[ym == c].mean(0) for c in np.unique(ym)])
        y = np.unique(ym)[((z[:, None] - C[None]) ** 2).sum(-1).argmin(1)]; y[main] = ym
        for sname, r in score_all(S, y, z).items():
            rows.append(dict(set=sname, method=f"{tag} -> CTGL", K=K, sec=round(time.time() - t), **r)); print(rows[-1], flush=True)
        pd.DataFrame(rows).round(4).to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
