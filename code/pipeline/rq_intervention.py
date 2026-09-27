"""Intervention test from existing embeddings: background removal (embed_fg.py patch masking, grey fill; SigLIP2 and
DINOv3 only) as a do(style) change that keeps the object. D_main rows, fits on D_main only.
(1) share of the SigLIP2 displacement ||Delta U'||^2/||Delta||^2 lying in the CMRE-removed subspace U (r = 64) for
    Delta_bg = z_fg - z (background removed) vs Delta_obj = z_j - z_i (same source, different class) vs chance 64/d;
(2) cluster flip rate when z is replaced by z_fg: 2-model variant raw [SigLIP2, DINOv3] vs CMRE [CMRE(SigLIP2), DINOv3],
    Spectral K=16 fitted on the original images, masked images -> nearest centroid.
Run: python code/pipeline/rq_intervention.py"""
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from compare_aimv2_fusions import ROOT, fuse, normalize, tta  # sets EXP_OUT
import rq1_metrics as metrics
from common import load_items, reduce
from rq_metrics2 import partition

FG = ROOT / "data" / "fg"


def main():
    items = load_items(); main, _, lab, src, _, _ = metrics.setup(items)
    fg = lambda k: normalize(normalize(np.load(FG / f"emb_{k}_fg.npy").astype(np.float32)) +
                             normalize(np.load(FG / f"emb_{k}_fg_flip.npy").astype(np.float32)))
    sig, dino, sig_f, dino_f = tta("siglip2"), tta("dinov3hp"), fg("siglip2"), fg("dinov3hp")
    res = sig[main] - Ridge(alpha=1.0).fit(dino[main], sig[main]).predict(dino[main])
    U = PCA(64, random_state=0).fit(res).components_.astype(np.float32)  # CMRE basis (r = 64, as in v4)
    share = lambda D: (((D @ U.T) ** 2).sum(1) / ((D ** 2).sum(1) + 1e-12))
    m = np.where(main)[0]; out = {}
    out["share_U_background_removed"] = float(np.median(share(sig_f[m] - sig[m])))
    rng = np.random.default_rng(0); ok = m[lab[m] != ""]
    i, j = rng.choice(ok, 20000), rng.choice(ok, 20000)
    keep = (lab[i] != lab[j]) & (src[i] == src[j])
    out["share_U_object_changed"] = float(np.median(share(sig[j[keep]] - sig[i[keep]])))
    out["share_U_chance"] = 64 / sig.shape[1]
    cm = lambda v: normalize(v - (v @ U.T) @ U)
    for tag, orig, masked in (("mentah", fuse(sig, dino), fuse(sig_f, dino_f)),
                              ("CMRE", fuse(cm(sig), dino), fuse(cm(sig_f), dino_f))):
        y, _ = partition(orig, main, 16)
        z = reduce(orig[main], np.vstack([orig, masked]), 32); zo, zm = z[:len(orig)], z[len(orig):]
        C = np.stack([zo[main][y[main] == c].mean(0) for c in range(16)])
        yf = ((zm[m][:, None] - C[None]) ** 2).sum(-1).argmin(1)
        out[f"flip_rate_{tag}"] = float((yf != y[m]).mean())
    for k, v in out.items(): print(f"{k:32s} {v:.3f}")
    pd.Series(out).round(4).rename("value").to_csv(ROOT / "results" / "classic" / "intervention_fg.csv", index_label="metric")


if __name__ == "__main__":
    main()
