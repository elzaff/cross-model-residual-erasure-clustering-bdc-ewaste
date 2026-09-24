"""Shared stack for the round-8 analyses: D_main + external rows (items.csv) + Iliev + Shubha crops with the frozen v4
features (TTA-averaged, same loading as rq_iliev.py / rq_shubha.py), the v4 building blocks and per-class xsrc scoring.
Every fit (Ridge, PCA, clustering) uses D_main rows only; labels are read only for post-hoc scoring."""
import numpy as np, pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from common import *
import rq1_metrics as M

CLS10 = ["battery", "keyboard", "microwave", "mobile", "mouse", "pcb", "player", "printer", "television", "washing_machine"]
C4 = ["keyboard", "mobile", "mouse", "pcb"]
items = load_items(); main, style, lab, src, cat, shared = M.setup(items); N = len(items); G = len(np.unique(style[main]))

def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def _new(tag):
    T = pd.read_csv(f"{OUT}/{tag}/items.csv", keep_default_na=False)
    keep = (T.set != "control").values & (T.posthoc_label != "EXCLUDE").values
    return T[keep].reset_index(drop=True), keep
NI, KI = _new("iliev"); NS, KS = _new("shubha"); n_i, n_s = len(NI), len(NS)
mainX = np.r_[main, np.zeros(n_i + n_s, bool)]
bang = shared & (src == "bangla")
SETS = {"BDC": (np.where(bang)[0], lab[bang], SHARED), "Iliev": (N + np.arange(n_i), NI.posthoc_label.values, CLS10),
        "Shubha": (N + n_i + np.arange(n_s), NS.posthoc_label.values, C4)}
KA = {c: (src == "kaan") & (lab == c) for c in CLS10}

def feat(k):
    def new(t, kp): return nz(nz(np.load(f"{OUT}/{t}/emb_{k}.npy").astype(np.float32)) + nz(np.load(f"{OUT}/{t}/emb_{k}_flip.npy").astype(np.float32)))[kp]
    return np.vstack([nz(emb(k) + emb(k + "_flip")), new("iliev", KI), new("shubha", KS)])

def fuse(*a): return nz(np.hstack(a))
def cmre(v, ref, alpha=1.0, m=None, basis=False):  # identical to rq_final2.py at alpha=1 (PCA(64) cap -> r = 64)
    m = mainX if m is None else m
    res = v[m] - Ridge(alpha=alpha).fit(ref[m], v[m]).predict(ref[m]); p = PCA(64, random_state=0).fit(res)
    U = p.components_[:int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.5) + 1)].astype(np.float32)
    out = nz(v - (v @ U.T) @ U); return (out, U) if basis else out
def inlp_x(x, n_max=8):
    xf, s = x[mainX], style[main]; P = np.eye(x.shape[1], dtype=np.float32)
    for _ in range(n_max):
        W = LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T); P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s, cv=5, scoring="balanced_accuracy").mean() <= 1 / G: break
    return nz(x @ P)
def spectral(zm, K=16, seed=0):
    return SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15, random_state=seed, assign_labels="cluster_qr").fit_predict(zm)
def project(x, ym=None, m=None, K=16, seed=0):
    """PCA-32 fitted on D_main, Spectral-kNN on D_main (or the given ym), every other row to the nearest centroid."""
    m = mainX if m is None else m; z = reduce(x[m], x, 32); zm = z[m]
    if ym is None: ym = spectral(zm, K, seed)
    ks = np.unique(ym); C = np.stack([zm[ym == c].mean(0) for c in ks])
    y = ks[((z[:, None] - C[None]) ** 2).sum(-1).argmin(1)]; y[np.where(m)[0]] = ym
    return y, z

def per_class(y, name, sub=None):
    """Share of the second-source images of class c that land in the Kaan-majority cluster of c."""
    idx, labs, classes = SETS[name]
    if sub is not None: idx, labs = idx[sub], labs[sub]
    km = {c: np.bincount(y[:N][KA[c]]).argmax() for c in classes}
    return {c: float((y[idx][labs == c] == km[c]).mean()) for c in classes if (labs == c).any()}
def xsrc(y, name, sub=None): return float(np.mean(list(per_class(y, name, sub).values())))
def purity_bdc(y): return float(pd.crosstab(y[:N][cat], lab[cat]).max(axis=1).sum() / cat.sum())
def v4_main():
    A4 = pd.read_csv(f"{OUT}/final_v4/assignments.csv", dtype={"id": str}); v4 = dict(zip(A4.id, A4.cluster))
    return np.array([v4[r["id"]] for r, m in zip(items, main) if m])
def methods(sig, pe, dino):
    return {"usulan v4": lambda: fuse(cmre(sig, dino), cmre(pe, dino), dino), "fusi mentah": lambda: fuse(sig, pe, dino),
            "DINOv3-H+ saja": lambda: dino, "INLP x3": lambda: fuse(inlp_x(sig), inlp_x(pe), inlp_x(dino)),
            "ringan CMRE(pe)+dino": lambda: fuse(cmre(pe, dino), dino)}
