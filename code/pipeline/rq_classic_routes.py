"""Classic routes to CMRE (2-model variant: SigLIP2 view corrected, DINOv3 raw; fused, PCA-32, Spectral K=16).
Every route removes r = 64 directions from SigLIP2 with the same projector x(I - U'U); only the estimate of U differs:
  ABTT                   : top PCs of SigLIP2 itself (Mu & Viswanath 2018); not a full SVA/RUV fit
  Proxy-batch scaling    : location/scale per pixel-style proxy group; not empirical-Bayes ComBat
  EPO-flip               : PCA of (z - z_flip) paired differences (Roger et al. 2003)
  EPO-latar              : PCA of (z_fg - z) background-removal pairs (data/fg, embed_fg.py)
  Label-conditioned PCA  : PCA of residual of SigLIP2 regressed on one-hot object labels;
                           diagnostic only, not a valid unlabeled method or performance upper bound
  CMRE                   : PCA of residual of SigLIP2 regressed on DINOv3 (label-free covariate substitute)
All bases are fitted on D_main; scored with rq_divide.score_all on BDC, Iliev, Shubha.
Run: python code/pipeline/rq_classic_routes.py"""
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from rq_divide import ROOT, build, fuse, score_all  # sets EXP_OUT first
from compare_aimv2_fusions import normalize
from common import emb
from rq_labeled import CLS
from rq_metrics2 import partition

R = 64


def proj(v, U): return normalize(v - (v @ U.T) @ U)
def pcs(D): return PCA(R, random_state=0).fit(D).components_.astype(np.float32)


def main():
    S = build(); X, main, lab = S["X"], S["main"], S["lab"]; sig, dino = X["siglip2"], X["dinov3hp"]
    N = len(emb("siglip2")); m = np.where(main)[0]; assert m.max() < N
    fg = lambda k: normalize(np.load(ROOT / "data" / "fg" / f"emb_{k}.npy").astype(np.float32))
    sig_fg = normalize(fg("siglip2_fg") + fg("siglip2_fg_flip"))
    lm = main & np.isin(lab, CLS); Y = pd.get_dummies(lab[lm]).values.astype(np.float32)
    style = S["style"]; mu, sd = sig[main].mean(0), sig[main].std(0) + 1e-6
    def batch_location_scale(v):
        y = v.copy()
        for g in np.unique(style[main]):
            f, a = main & (style == g), style == g
            y[a] = (v[a] - v[f].mean(0)) / (v[f].std(0) + 1e-6) * sd + mu
        return normalize(y)
    routes = {
        "1 mentah": lambda: sig,
        "2 ABTT tanpa kovariat": lambda: proj(sig - sig[main].mean(0), pcs(sig[main] - sig[main].mean(0))),
        "3 Location/scale proksi batch": lambda: batch_location_scale(sig),
        "4 EPO-flip (pasangan flip)": lambda: proj(sig, pcs(emb("siglip2")[m] - emb("siglip2_flip")[m])),
        "5 EPO-latar (pasangan tanpa latar)": lambda: proj(sig, pcs(sig_fg[m] - sig[m])),
        "6 Residual PCA berlabel (diagnostik)": lambda: proj(sig, pcs(sig[lm] - Ridge(alpha=1.0).fit(Y, sig[lm]).predict(Y))),
        "7 CMRE (DINOv3 sbg kovariat)": lambda: proj(sig, pcs(sig[main] - Ridge(alpha=1.0).fit(dino[main], sig[main]).predict(dino[main]))),
    }
    rows = []
    for name, f in routes.items():
        y, z = partition(fuse(f(), dino), main, 16)
        for sname, r in score_all(S, y, z).items():
            rows.append(dict(scenario=name, set=sname, **r)); print(rows[-1], flush=True)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "classic" / "classic_routes.csv", index=False)


if __name__ == "__main__":
    main()
