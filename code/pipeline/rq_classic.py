"""Classic-method grid on the cached v4 embeddings (CPU only).
Stages: A debiasing (none / ABTT / ComBat-proxy / INLP-proxy / CMRE) x B fusion (concat / SNF / graph average)
x C clustering (K-Means / Ward / GMM / Spectral-kNN / Louvain), all at K=16. Every fit uses D_main rows only;
external rows are assigned to the nearest centroid in the concat PCA-32 space; labels are read only for scoring.
Run: EXP_OUT=data/embeddings_final_v4 python code/pipeline/rq_classic.py"""
import os, time, numpy as np, pandas as pd, networkx as nx, scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from common import *
import rq1_metrics as M

K, SEEDS = 16, (0, 1, 2)
items = load_items(); main, style, lab, src, cat, shared = M.setup(items); G = len(np.unique(style[main]))
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def tta(k): return nz(emb(k) + emb(k + "_flip"))
sig, pe, dino = tta("siglip2"), tta("pecoreg"), tta("dinov3hp")

# ---------- A: debiasing (fit on D_main, apply to all rows) ----------
def abtt(x, d):  # Mu & Viswanath 2018: remove the mean and the top-d principal directions of the model itself
    mu = x[main].mean(0); U = PCA(d, random_state=0).fit(x[main] - mu).components_
    xc = x - mu; return nz(xc - (xc @ U.T) @ U)
def combat(x):  # Johnson et al. 2007 location/scale per batch, batch = pixel style proxy (no EB shrinkage: groups n>=100)
    mu, sd, y = x[main].mean(0), x[main].std(0) + 1e-6, x.copy()
    for g in np.unique(style):
        f = main & (style == g); y[style == g] = (x[style == g] - x[f].mean(0)) / (x[f].std(0) + 1e-6) * sd + mu
    return nz(y)
def inlp(x, n_max=8):  # Ravfogel et al. 2020, stop when the proxy is at chance (same rule as stack.py)
    xf, s = x[main], style[main]; P = np.eye(x.shape[1], dtype=np.float32)
    for _ in range(n_max):
        Q, _ = np.linalg.qr(LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32).T)
        P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s, cv=5,
                           scoring="balanced_accuracy").mean() <= 1 / G: break
    return nz(x @ P)
def cmre(v, ref=None, r=64):  # residual of VLM given SSL (SVA/RUVr-style estimate) -> EPO projection
    ref = dino if ref is None else ref
    res = v[main] - Ridge(alpha=1.0).fit(ref[main], v[main]).predict(ref[main])
    U = PCA(r, random_state=0).fit(res).components_; return nz(v - (v @ U.T) @ U)

A = {"A0 mentah": lambda: [sig, pe, dino],
     "A1 ABTT d=D/100": lambda: [abtt(sig, 12), abtt(pe, 15), dino],
     "A1 ABTT d=64": lambda: [abtt(sig, 64), abtt(pe, 64), dino],
     "A2 ComBat-proksi": lambda: [combat(sig), combat(pe), dino],
     "A2 ComBat-proksi x3": lambda: [combat(sig), combat(pe), combat(dino)],
     "A3 INLP x3": lambda: [inlp(sig), inlp(pe), inlp(dino)],
     "A4 CMRE (v4)": lambda: [cmre(sig), cmre(pe), dino]}

# ---------- B: fusion -> affinity (for graph clusterers) ----------
def knn_sym(z, k=15):
    G_ = NearestNeighbors(n_neighbors=k).fit(z).kneighbors_graph(z, mode="connectivity")  # self included, as sklearn
    return 0.5 * (G_ + G_.T)
def snf(views, k=20, mu=0.5, t=20):  # Wang et al. 2014, Nature Methods
    Ps, Ss = [], []
    for z in views:
        D = cdist(z, z, "sqeuclidean"); np.fill_diagonal(D, 0)
        Ds = np.sort(D, 1)[:, 1:k + 1].mean(1); eps = (Ds[:, None] + Ds[None] + D) / 3
        W = np.exp(-D / (2 * (mu * eps) ** 2))
        P = W.copy(); np.fill_diagonal(P, 0); P = P / (2 * P.sum(1, keepdims=True)); np.fill_diagonal(P, 0.5)
        idx = np.argsort(-W, 1)[:, 1:k + 1]; S = np.zeros_like(W); r = np.arange(len(W))[:, None]
        S[r, idx] = W[r, idx]; S = sp.csr_matrix(S / S.sum(1, keepdims=True)); Ps.append(P); Ss.append(S)
    for _ in range(t):
        Ps = [S @ ((sum(Ps) - P) / (len(Ps) - 1)) @ S.T for P, S in zip(Ps, Ss)]
        Ps = [(P + P.T) / 2 for P in Ps]
    return sum(Ps) / len(Ps)

# ---------- C: clustering on D_main ----------
def louvain(Wm, seed):  # Blondel et al. 2008; resolution bisected until K communities
    if not sp.issparse(Wm):  # dense SNF affinity: keep top-15 per row (symmetrized), else networkx gets ~14M edges
        idx = np.argsort(-Wm, 1)[:, 1:16]; r = np.arange(len(Wm))[:, None]
        S = np.zeros_like(Wm); S[r, idx] = Wm[r, idx]; Wm = sp.csr_matrix(np.maximum(S, S.T))
    g = nx.from_scipy_sparse_array(sp.csr_matrix(Wm)); lo, hi = 0.05, 20.0
    for _ in range(40):
        res = (lo * hi) ** 0.5; com = nx.community.louvain_communities(g, resolution=res, seed=seed)
        if len(com) == K: break
        lo, hi = (res, hi) if len(com) < K else (lo, res)
    y = np.empty(Wm.shape[0], int)
    for i, c in enumerate(com): y[list(c)] = i
    return y
def cluster(name, zm, Wm, seed):
    if name == "K-Means": return kmeans(zm, K, seed).labels_
    if name == "Ward": return AgglomerativeClustering(K, linkage="ward").fit_predict(zm)
    if name == "GMM": return GaussianMixture(K, covariance_type="full", n_init=3, random_state=seed).fit(zm).predict(zm)
    if name == "Spectral": return SpectralClustering(K, affinity="precomputed", random_state=seed,
                                                     assign_labels="cluster_qr").fit_predict(Wm)
    if name == "Louvain": return louvain(Wm, seed)

def evaluate(y_main, z):
    zm = z[main]; ks = np.unique(y_main); C = np.stack([zm[y_main == c].mean(0) for c in ks])
    y = ks[((z[:, None] - C[None]) ** 2).sum(-1).argmin(1)]; y[main] = y_main
    s = M.score_labels(y, main, style, lab, src, cat, shared)
    pur = pd.crosstab(y[cat], lab[cat]).max(axis=1).sum() / cat.sum()
    return dict(xsrc=s["xsrc_agree"], src_obj=s["ext_src_given_obj"], ext_nmi=s["ext_obj_nmi"],
                bdc_nmi=s["obj_nmi_bdc"], purity=pur, style_nmi=s["style_nmi"],
                sil=silhouette_score(zm, y_main, metric="cosine"), n_clu=len(ks))

if __name__ == "__main__":
    rows, out = [], os.path.join(ROOT, "results", "classic"); os.makedirs(out, exist_ok=True)
    for a, fa in A.items():
        t0 = time.time(); views = fa(); x = nz(np.hstack(views)); z = reduce(x[main], x, 32); zm = z[main]
        vz = [reduce(v[main], v, 32)[main] for v in views]
        aff = {"concat": lambda: knn_sym(zm), "graf-rata": lambda: sum(knn_sym(v) for v in vz) / 3, "SNF": lambda: snf(vz)}
        for b, fb in aff.items():
            Wm = fb()
            for c in (["K-Means", "Ward", "GMM", "Spectral", "Louvain"] if b == "concat" else ["Spectral", "Louvain"]):
                ys = [cluster(c, zm, Wm, s) for s in (SEEDS if c != "Ward" else (0,))]
                ev = pd.DataFrame([evaluate(y, z) for y in ys]).mean().to_dict()
                ev["seed_ari"] = np.mean([ari(ys[0], y) for y in ys[1:]]) if len(ys) > 1 else 1.0
                rows.append(dict(debias=a, fusion=b, cluster=c, **ev)); print(rows[-1], flush=True)
        print(a, f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "grid_k16.csv"), index=False)
