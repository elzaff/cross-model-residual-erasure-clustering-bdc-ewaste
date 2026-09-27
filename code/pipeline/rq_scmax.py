"""Label-free K via SCMax (Zhang et al., AAAI-26; official commit 89437a1, MIT) on debiased features (images only).
The SCMax loop mirrors the authors' demo.py with their default settings (AE 256-d, 200 MSE epochs, 50 back-feature epochs,
batch 256, lr 3e-4); only the input features and the seeds change. For each feature set and seed we record every
nearest-neighbour merge level (FINCH-like candidates) with its NNC score, the SCMax pick, and
  hybrid: Spectral-kNN (our pipeline) run with K = the SCMax pick.
External images are assigned to the nearest D_main centroid in the PCA-32 space used everywhere else.
Writes OUT/scmax_levels.csv and OUT/scmax_results.csv (checkpointed)."""
import os, random, numpy as np, pandas as pd, torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from common import *
import rq1_metrics as M
from scmax.auto_encoder import AutoEncoder
from scmax.neighbor_clustering import NeighborClustering
from scmax.feature_optimization import BackFeature

from scmax_loop import DEV, BS, LR, MSE_EP, BACK_EP, FDIM, nnc, scmax  # loop lives in scmax_loop.py
items = load_items(); main, style, lab, src, cat, shared = M.setup(items); G = len(np.unique(style[main]))
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def tta(a): return nz(emb(a) + emb(a + "_flip"))
def fuse(*a): return nz(np.hstack(a))
def purity(y, m): return float(pd.crosstab(y[m], lab[m]).max(axis=1).sum() / m.sum())
def full_metrics(y):
    return dict(**M.score_labels(y, main, style, lab, src, cat, shared), purity_bdc=purity(y, cat), purity_ext=purity(y, (~main) & (lab != "")))
def assign(z, ym):
    zm = z[main]; C = np.stack([zm[ym == c].mean(0) for c in np.unique(ym)])
    y = ((z[:, None] - C[None]) ** 2).sum(-1).argmin(1); y[np.where(main)[0]] = ym; return y
def cmre(v, ref):
    reg = Ridge(alpha=1.0).fit(ref[main], v[main]); res = v[main] - reg.predict(ref[main])
    p = PCA(64, random_state=0).fit(res); r = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.5) + 1)
    U = p.components_[:r].astype(np.float32); return nz(v - (v @ U.T) @ U)
def inlp_x(x, n_max=8):
    xf, s = x[main], style[main]; P = np.eye(x.shape[1], dtype=np.float32)
    for _ in range(n_max):
        W = LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T); P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s, cv=5, scoring="balanced_accuracy").mean() <= 1 / G: break
    return nz(x @ P)

sig, pe, dino = tta("siglip2"), tta("pecoreg"), tta("dinov3hp")
FEATS = {"CMRE(sig)+CMRE(pe)+dino": lambda: fuse(cmre(sig, dino), cmre(pe, dino), dino),
         "INLP x3": lambda: fuse(inlp_x(sig), inlp_x(pe), inlp_x(dino)),
         "fusi mentah": lambda: fuse(sig, pe, dino)}
SEEDS = [int(s) for s in os.environ.get("SCMAX_SEEDS", "3407,0,1").split(",")]  # extra seeds: SCMAX_SEEDS=2,3,...
FEATS = {k: f for k, f in FEATS.items() if k.startswith(os.environ.get("SCMAX_FEATS", ""))}

if __name__ == "__main__":  # guard so scmax() can be imported
    RUN_TAG = os.environ.get("SCMAX_RUN_TAG", "")
    LP, RP = f"{OUT}/scmax_levels{RUN_TAG}.csv", f"{OUT}/scmax_results{RUN_TAG}.csv"
    lev_rows = pd.read_csv(LP).to_dict("records") if os.path.exists(LP) else []
    res_rows = pd.read_csv(RP).to_dict("records") if os.path.exists(RP) else []
    done = {(r["feat"], r["seed"]) for r in res_rows}
    for name, fx in FEATS.items():
        x = fx(); z = reduce(x[main], x, 32); zm = z[main]
        for seed in SEEDS:
            if (name, seed) in done: continue
            levels = scmax(x[main], seed)
            for L in levels:
                lev_rows.append(dict(feat=name, seed=seed, K=L["K"], nnc=L["nnc"], sil=silhouette_score(zm, L["labels"]),
                                     **full_metrics(assign(z, L["labels"]))))
            best = max(levels, key=lambda L: L["nnc"])
            ys = SpectralClustering(best["K"], affinity="nearest_neighbors", n_neighbors=15, random_state=0,
                                    assign_labels="cluster_qr").fit_predict(zm)
            res_rows.append(dict(feat=name, seed=seed, K_levels=str([L["K"] for L in levels]), K_scmax=best["K"], nnc=best["nnc"],
                                 **{f"scmax_{k}": v for k, v in full_metrics(assign(z, best["labels"])).items()},
                                 **{f"hybrid_{k}": v for k, v in full_metrics(assign(z, ys)).items()}))
            pd.DataFrame(lev_rows).to_csv(LP, index=False); pd.DataFrame(res_rows).to_csv(RP, index=False)
            r = res_rows[-1]
            print("SCMAX", name, "seed", seed, "| levels", r["K_levels"], "| K*", r["K_scmax"], "nnc", round(r["nnc"], 3),
                  "| scmax xsrc", round(r["scmax_xsrc_agree"], 3), "ext_obj", round(r["scmax_ext_obj_nmi"], 3),
                  "| hybrid xsrc", round(r["hybrid_xsrc_agree"], 3), "ext_obj", round(r["hybrid_ext_obj_nmi"], 3), flush=True)
    print("SCMAX DONE")
