"""Shared loaders, style proxies, debiasing and metrics for the analysis scripts."""
import csv, os, numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
OUT = os.environ.get("EXP_OUT", os.path.join(ROOT, "results", "pilot_local"))
SHARED = ["battery", "keyboard", "mobile", "mouse", "pcb"]  # classes present in >=2 external sources

def load_items():
    rows = list(csv.DictReader(open(os.path.join(OUT, "items.csv"), encoding="utf-8")))
    for r in rows:
        for k in ("width", "height", "border_white", "border_std", "colorfulness"): r[k] = float(r[k])
    return rows

def emb(name):
    x = np.load(os.path.join(OUT, f"emb_{name}.npy")).astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def style_groups(items, white_thr=0.5, res_bins=(200, 800)):
    """Acquisition-style proxy derived from pixels only: white/uniform background x resolution bin."""
    white = np.array([r["border_white"] >= white_thr for r in items])
    side = np.array([max(r["width"], r["height"]) for r in items])
    res = np.digitize(side, list(res_bins))  # 0: thumbnail, 1: web, 2: camera
    return white.astype(int) * 3 + res, white, res

def inlp(x_fit, s_fit, x_all, n_iter=8):
    """Iterative Null-space Projection (Ravfogel et al., 2020): remove directions linearly predictive of style."""
    P = np.eye(x_fit.shape[1], dtype=np.float32)
    for _ in range(n_iter):
        W = LogisticRegression(max_iter=300).fit(x_fit @ P, s_fit).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T)  # orthonormal basis of the style classifier directions
        P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
    y = x_all @ P
    return y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)

def reduce(x_fit, x_all, dim=64, seed=0):
    z = PCA(dim, random_state=seed).fit(x_fit).transform(x_all)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

def kmeans(z_fit, k, seed=0):
    return KMeans(k, n_init=10, random_state=seed).fit(z_fit)

def cond_nmi(y, a, b):
    """Weighted mean over values of b of NMI(y, a | b) -- e.g. cluster vs source within each object class."""
    tot, acc = 0, 0.0
    for v in np.unique(b):
        m = b == v
        if len(np.unique(a[m])) > 1:
            acc += m.sum() * nmi(a[m], y[m]); tot += m.sum()
    return acc / max(tot, 1)
