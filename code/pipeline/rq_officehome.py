"""Second domain: Office-Home, 20 electrical/electronic classes, Product (catalogue, white background) vs Real World
(camera photos) -- the same catalogue-vs-amateur split as Kaan vs Bangladesh, on a public benchmark. The unchanged
pipeline is rerun from scratch on these images (images only; class and domain labels used only for metrics):
  single models, raw fusion, INLP x3 (pixel style proxy), CMRE (r=64, DINOv3-H+ reference), light CMRE(PE)+DINOv3;
  Spectral-kNN with K = 20 (number of classes) averaged over 3 seeds, and SCMax (authors' defaults, 3 seeds) as the
  label-free K for raw / INLP / CMRE features ("CMRE before SCMax" test).
Metrics: xsrc (share of Real World images of class c in the Product majority cluster of c, mean over classes), domain
given object NMI (lower = better), object NMI, purity. Writes OUT/officehome_results.csv."""
import os, random, numpy as np, pandas as pd, torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, normalized_mutual_info_score as nmi
from common import OUT, reduce, style_groups, cond_nmi, ari
from scmax.auto_encoder import AutoEncoder
from scmax.neighbor_clustering import NeighborClustering
from scmax.feature_optimization import BackFeature
import warnings; warnings.filterwarnings("ignore")

D = f"{OUT}/officehome"; T = pd.read_csv(f"{D}/items.csv")
for k in ("width", "height", "border_white"): T[k] = T[k].astype(float)
dom, cls = T.domain.values, T.cls.values; style = style_groups(T.to_dict("records"))[0]; G = len(np.unique(style))
CL = sorted(set(cls)); K0 = len(CL)
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def tta(k): return nz(nz(np.load(f"{D}/emb_{k}.npy").astype(np.float32)) + nz(np.load(f"{D}/emb_{k}_flip.npy").astype(np.float32)))
def fuse(*a): return nz(np.hstack(a))
def cmre(v, ref):  # identical rule to the frozen pipeline (PCA(64) cap -> r = 64)
    res = v - Ridge(alpha=1.0).fit(ref, v).predict(ref); p = PCA(64, random_state=0).fit(res)
    U = p.components_[:int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.5) + 1)].astype(np.float32); return nz(v - (v @ U.T) @ U)
def inlp_x(x, n_max=8):
    P = np.eye(x.shape[1], dtype=np.float32)
    for _ in range(n_max):
        W = LogisticRegression(max_iter=300).fit(x @ P, style).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T); P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), x @ P, style, cv=5, scoring="balanced_accuracy").mean() <= 1 / G: break
    return nz(x @ P)
def metrics(y):
    xs = [(y[(cls == c) & (dom == "realworld")] == np.bincount(y[(cls == c) & (dom == "product")]).argmax()).mean() for c in CL]
    return dict(xsrc=float(np.mean(xs)), dom_given_obj=float(cond_nmi(y, dom, cls)), obj_nmi=float(nmi(cls, y)),
                purity=float(pd.crosstab(y, cls).max(axis=1).sum() / len(y)))
def spectral(z, K, s): return SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15, random_state=s, assign_labels="cluster_qr").fit_predict(z)
def evaluate(x, K):
    z = reduce(x, x, 32); ys = [spectral(z, K, s) for s in range(3)]; ms = [metrics(y) for y in ys]
    return {**{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}, "seed_ari": float(np.mean([ari(ys[0], o) for o in ys[1:]])),
            "sil": float(silhouette_score(z, ys[0])), "K": K}

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def nnc(a, b):
    Dd = max(a.max(), b.max()) + 1; Cm = np.zeros((Dd, Dd), np.int64); np.add.at(Cm, (a, b), 1)
    r, c = linear_sum_assignment(Cm.max() - Cm); return Cm[r, c].sum() / len(a)
def scmax_k(X, seed, BS=256, LR=3e-4, MSE_EP=200, BACK_EP=50, FDIM=256):  # same loop as rq_scmax.py
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    ae = AutoEncoder(1, [X.shape[1]], FDIM, DEV); opt = torch.optim.Adam(ae.parameters(), lr=LR); mse = torch.nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV); n = len(X)
    def epochs(E, labels=None, K=None):
        bf = BackFeature(BS, K, DEV) if labels is not None else None
        for _ in range(E):
            ae.train(); idx = np.random.permutation(n)
            for s in range(0, n, BS):
                b = idx[s:s + BS]; opt.zero_grad(); xr, zz = ae([Xt[b]])
                loss = mse(xr[0], Xt[b]) + (bf.forward_label(zz[0], labels[b]) if bf else 0); loss.backward(); opt.step()
    def emb_z():
        ae.eval()
        with torch.no_grad(): return ae.forward_all_z([Xt])
    epochs(MSE_EP); Z = emb_z(); cl = NeighborClustering(); labels = np.arange(n); Zp, lp, levels = None, None, []
    while True:
        K, labels = cl.step(Z, labels)
        if Zp is not None:
            Kp, lp = cl.step(Zp, lp)
            if K < 3 or Kp < 3: break
            levels.append((K, nnc(labels, lp)))
        lp = labels; epochs(BACK_EP, labels, K); Zp = emb_z()
    return max(levels, key=lambda L: L[1])[0], [L[0] for L in levels]

sig, pe, dino = tta("siglip2"), tta("pecoreg"), tta("dinov3hp")
FEATS = {"SigLIP2 mentah": lambda: sig, "PE mentah": lambda: pe, "DINOv3-H+ saja": lambda: dino,
         "fusi mentah": lambda: fuse(sig, pe, dino), "INLP x3": lambda: fuse(inlp_x(sig), inlp_x(pe), inlp_x(dino)),
         "usulan CMRE": lambda: fuse(cmre(sig, dino), cmre(pe, dino), dino), "ringan CMRE(pe)+dino": lambda: fuse(cmre(pe, dino), dino)}
PART = f"{OUT}/officehome_results.csv"; rows = []
print("OFFICEHOME n", len(T), {d: int((dom == d).sum()) for d in ("product", "realworld")}, "classes", K0, flush=True)
for name, f in FEATS.items():
    x = f(); r = dict(method=name, **evaluate(x, K0))
    if name in ("fusi mentah", "INLP x3", "usulan CMRE"):
        ks = [scmax_k(x, s) for s in (3407, 0, 1)]; Ks = int(np.median([k for k, _ in ks]))
        r.update(scmax_K=str([k for k, _ in ks]), scmax_levels=str([l for _, l in ks]), **{f"hyb_{k}": v for k, v in evaluate(x, Ks).items()})
    rows.append(r); pd.DataFrame(rows).to_csv(PART, index=False)
    print("OH", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "method"}, flush=True)
print("OFFICEHOME DONE", flush=True)
