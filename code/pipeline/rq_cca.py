"""Shared-subspace baseline (Chaudhuri et al., ICML 2009): regularized CCA between each
VLM and DINOv3-H+, cluster on canonical variates. Protocol fixed before running:
pre-PCA 256 per view, reg = 1e-3 * mean eigenvalue, k in {32, 64, 128} all reported,
variants "CCA only" and "CCA + DINO", scored with the v4 protocol (PCA-32, Spectral K=16).
"""
import csv

import numpy as np
from sklearn.decomposition import PCA

from compare_aimv2_fusions import ROOT, cmre, fuse, normalize, score, tta  # sets EXP_OUT
from common import load_items
import rq1_metrics as metrics

OUT = ROOT / "results" / "modal" / "cca_results.csv"


def whitener(x):
    c = x.T @ x / len(x)
    c += 1e-3 * np.trace(c) / len(c) * np.eye(len(c))
    w, v = np.linalg.eigh(c)
    return v @ np.diag(w ** -0.5) @ v.T


def cca_pair(a, b, main, k):
    pa, pb = (PCA(256, random_state=0).fit(x[main]) for x in (a, b))
    xa, xb = pa.transform(a), pb.transform(b)
    wa, wb = whitener(xa[main]), whitener(xb[main])
    ua, s, vbt = np.linalg.svd(wa @ (xa[main].T @ xb[main] / main.sum()) @ wb)
    za, zb = xa @ wa @ ua[:, :k], xb @ wb @ vbt.T[:, :k]
    return normalize((za + zb) / 2), s[:k]


def main():
    items = load_items()
    main_mask, style, labels, source, cat, shared = metrics.setup(items)
    sig, pe, dino = (tta(n) for n in ("siglip2", "pecoreg", "dinov3hp"))
    args = (main_mask, style, labels, source, cat, shared)
    rows = []
    for k in (32, 64, 128):
        cs, rho_s = cca_pair(sig, dino, main_mask, k)
        cp, rho_p = cca_pair(pe, dino, main_mask, k)
        print(f"k={k} canonical corr sig [{rho_s[0]:.3f}..{rho_s[-1]:.3f}] pe [{rho_p[0]:.3f}..{rho_p[-1]:.3f}]")
        for name, feats in ((f"CCA only k={k}", fuse(cs, cp)), (f"CCA + DINO k={k}", fuse(cs, cp, dino))):
            rows.append({"method": name, "k": k, **score(name, feats, *args)})
    for name, feats in (("raw fusion", fuse(sig, pe, dino)),
                        ("CMRE v4 (r=64)", fuse(cmre(sig, dino, main_mask)[0], cmre(pe, dino, main_mask)[0], dino))):
        rows.append({"method": name, "k": "", **score(name, feats, *args)})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
