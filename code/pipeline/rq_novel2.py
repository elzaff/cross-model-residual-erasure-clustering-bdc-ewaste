"""Round 2 of method variants (images only, no labels, no text), K=16, compared on the same metrics as rq_novel.py.
Writes OUT/novel2_results.csv (checkpointed per method).
  G  CMRE generality: other VLM x SSL pairs (raw fusion / INLP both / CMRE)
  MR multi-reference CMRE (residual w.r.t. several SSL models) and multi-VLM CMRE fusion
  CL CMRE followed by LEACE
  HB hubness-reduced kNN graphs for Spectral (local scaling, mutual proximity)
  DC clustering heads on CMRE features: TURTLE (official loop), SCAN loss (frozen features), DMoN (GNN)
  FG foreground-masked embeddings (embed_fg.py) with / without CMRE
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.stats import norm as gauss
from scipy.spatial.distance import cdist
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
from common import *
import rq1_metrics as M
import warnings; warnings.filterwarnings("ignore")

DEV = "cuda" if torch.cuda.is_available() else "cpu"; K = 16
items = load_items(); main, style, lab, src, cat, shared = M.setup(items)
white = np.array([r["border_white"] >= 0.5 for r in items]); G = len(np.unique(style[main]))
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def tta(a): return nz(emb(a) + emb(a + "_flip"))
def probe(x, s=style): return cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), x[main], s[main], cv=5, scoring="balanced_accuracy").mean()
def purity(y, m): return float(pd.crosstab(y[m], lab[m]).max(axis=1).sum() / m.sum())
def xstyle_bdc(y):
    out = [(y[cat & (lab == c) & ~white] == np.bincount(y[cat & (lab == c) & white]).argmax()).mean()
           for c in np.unique(lab[cat]) if (cat & (lab == c) & white).sum() >= 15 and (cat & (lab == c) & ~white).sum() >= 15]
    return float(np.mean(out))
def full_metrics(y):
    return dict(**M.score_labels(y, main, style, lab, src, cat, shared), xstyle_bdc=xstyle_bdc(y),
                purity_bdc=purity(y, cat), purity_ext=purity(y, (~main) & (lab != "")))
def assign(z, ym):
    zm = z[main]; C = np.stack([zm[ym == c].mean(0) for c in np.unique(ym)])
    y = ((z[:, None] - C[None]) ** 2).sum(-1).argmin(1); y[np.where(main)[0]] = ym; return y
def spectral(z, seed=0, affinity=None):
    sc = SpectralClustering(K, affinity="precomputed" if affinity is not None else "nearest_neighbors", n_neighbors=15,
                            random_state=seed, assign_labels="cluster_qr")
    return sc.fit_predict(affinity if affinity is not None else z[main])
def evaluate(x, affinity_fn=None, extra=None):
    z = reduce(x[main], x, 32); aff = affinity_fn(z) if affinity_fn else None
    yms = [spectral(z, s, aff) for s in range(3)]
    return dict(seed_ari=float(np.mean([ari(yms[0], o) for o in yms[1:]])), sil=silhouette_score(z[main], yms[0]),
                probe_style=probe(x), **full_metrics(assign(z, yms[0])), **(extra or {}))
def from_labels(x, ys):
    """Metrics for a clustering head that predicts all items; ys = list of per-seed predictions."""
    z = reduce(x[main], x, 32)
    return dict(seed_ari=float(np.mean([ari(ys[0][main], o[main]) for o in ys[1:]])),
                sil=silhouette_score(z[main], ys[0][main]) if len(np.unique(ys[0][main])) > 1 else np.nan,
                n_clusters=int(len(np.unique(ys[0][main]))), **full_metrics(ys[0]))

def inlp_P(xf, s, n_max=8):
    P = np.eye(xf.shape[1], dtype=np.float32)
    for n in range(1, n_max + 1):
        W = LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T); P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s, cv=5, scoring="balanced_accuracy").mean() <= 1 / G: break
    return P
def inlp_x(x): return nz(x @ inlp_P(x[main], style[main]))
def leace(x, s=style):
    xm = x[main].astype(np.float64); mu = xm.mean(0); Xc = xm - mu
    Z = np.eye(G)[s[main]]; Z = Z - Z.mean(0)
    Sxx = Xc.T @ Xc / len(Xc) + 1e-4 * np.eye(Xc.shape[1]); Sxz = Xc.T @ Z / len(Xc)
    ev, V = np.linalg.eigh(Sxx); W = V @ np.diag(ev ** -0.5) @ V.T; Winv = V @ np.diag(ev ** 0.5) @ V.T
    Q, _ = np.linalg.qr(W @ Sxz); Pa = Q @ Q.T
    return nz(((x - mu) - (Winv @ Pa @ W @ (x - mu).T).T + mu).astype(np.float32))
def cmre(v, refs):
    """Cross-model residual erasure: erase the top principal directions (50%-variance rule) of the part of VLM
    embedding v that is not linearly predictable from the SSL reference embedding(s)."""
    R = np.hstack(refs); reg = Ridge(alpha=1.0).fit(R[main], v[main]); res = v[main] - reg.predict(R[main])
    p = PCA(64, random_state=0).fit(res); r = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.5) + 1)
    U = p.components_[:r].astype(np.float32); return nz(v - (v @ U.T) @ U), r
def fuse(*a): return nz(np.hstack(a))

PART = f"{OUT}/novel2_results.csv"
rows = pd.read_csv(PART).to_dict("records") if os.path.exists(PART) else []
done = {r["method"] for r in rows}
def run(name, fn):
    if name in done: return
    r = fn(); r["method"] = name; rows.append(r); pd.DataFrame(rows).to_csv(PART, index=False)
    print("NOVEL2", name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

sig, dino = tta("siglip2"), tta("dinov3hp")
sig_c, r0 = cmre(sig, [dino]); base = fuse(sig_c, dino)
run("REF CMRE(sig|dinov3hp)+dino", lambda: evaluate(base, extra=dict(r=r0)))

# ---- G: generality over VLM x SSL pairs
PAIRS = [("siglip2", "dinov2l_cls"), ("siglip2", "dinov3_7b"), ("pecoreg", "dinov3hp"), ("clip", "dinov3hp"),
         ("clipcnxxxl", "dinov3hp"), ("aimv2", "dinov3hp")]
for v, s in PAIRS:
    if not os.path.exists(f"{OUT}/emb_{v}.npy") or not os.path.exists(f"{OUT}/emb_{s}.npy"): print("skip pair", v, s); continue
    xv, xs = tta(v), tta(s)
    run(f"G {v}+{s} mentah", lambda: evaluate(fuse(xv, xs)))
    run(f"G {v}+{s} INLP", lambda: evaluate(fuse(inlp_x(xv), inlp_x(xs))))
    run(f"G {v}+{s} CMRE", lambda: (lambda c: evaluate(fuse(c[0], xs), extra=dict(r=c[1])))(cmre(xv, [xs])))

# ---- MR: multi-reference / multi-VLM
refs = [dino, tta("dinov2l_cls"), tta("dinov3_7b")]
sig_mr, r_mr = cmre(sig, refs)
run("MR CMRE(sig|3 SSL)+dino", lambda: evaluate(fuse(sig_mr, dino), extra=dict(r=r_mr)))
pe_c, r_pe = cmre(tta("pecoreg"), [dino])
run("MR CMRE(sig)+CMRE(pe)+dino", lambda: evaluate(fuse(sig_c, pe_c, dino), extra=dict(r=r_pe)))

# ---- CL: CMRE then LEACE
run("CL LEACE(CMRE(sig))+dino", lambda: evaluate(fuse(leace(sig_c), dino)))
run("CL LEACE(CMRE(sig))+LEACE(dino)", lambda: evaluate(fuse(leace(sig_c), leace(dino))))

# ---- HB: hubness-reduced graphs on the CMRE fusion
def local_scaling(z, k=15, k_sigma=7):
    zm = z[main]; D = cdist(zm, zm); np.fill_diagonal(D, np.inf); idx = np.argsort(D, 1)[:, :k]
    sig_ = np.sort(D, 1)[:, k_sigma - 1]; rr = np.repeat(np.arange(len(zm)), k); cc = idx.ravel()
    A = csr_matrix((np.exp(-D[rr, cc] ** 2 / (sig_[rr] * sig_[cc])), (rr, cc)), shape=D.shape); return 0.5 * (A + A.T)
def mutual_proximity(z, k=15):
    zm = z[main]; D = cdist(zm, zm); np.fill_diagonal(D, np.nan)
    mu, sd = np.nanmean(D, 1), np.nanstd(D, 1)
    S = gauss.sf((D - mu[:, None]) / sd[:, None]) * gauss.sf((D - mu[None]) / sd[None]); np.fill_diagonal(S, 0)
    idx = np.argsort(-S, 1)[:, :k]; rr = np.repeat(np.arange(len(zm)), k); cc = idx.ravel()
    A = csr_matrix((S[rr, cc], (rr, cc)), shape=S.shape); return 0.5 * (A + A.T)
run("HB local scaling (CMRE fusi)", lambda: evaluate(base, affinity_fn=local_scaling))
run("HB mutual proximity (CMRE fusi)", lambda: evaluate(base, affinity_fn=mutual_proximity))

# ---- DC: clustering heads on CMRE features
def turtle(Zs, seed, T=6000, M_in=10, gamma=10., lr=1e-3):
    """TURTLE (Gadetsky et al., 2024) official alternating loop, fitted on D_main, predicts all items."""
    torch.manual_seed(seed)
    Ztr = [torch.tensor(z[main], device=DEV) for z in Zs]; dims = [z.shape[1] for z in Zs]
    enc = [nn.utils.weight_norm(nn.Linear(d, K)).to(DEV) for d in dims]
    opt = torch.optim.Adam(sum([list(e.parameters()) for e in enc], []), lr=lr)
    def encode(Z): per = [F.softmax(e(z), 1) for e, z in zip(enc, Z)]; return torch.stack(per).mean(0), per
    for _ in range(T):
        labels, per = encode(Ztr)
        W = [nn.Linear(d, K).to(DEV) for d in dims]; iopt = torch.optim.Adam(sum([list(w.parameters()) for w in W], []), lr=lr)
        for _ in range(M_in):
            iopt.zero_grad(); sum(F.cross_entropy(w(z), labels.detach()) for w, z in zip(W, Ztr)).backward(); iopt.step()
        opt.zero_grad()
        err = sum(F.cross_entropy(w(z).detach(), labels) for w, z in zip(W, Ztr))
        (err - gamma * sum(torch.special.entr(p.mean(0)).sum() for p in per)).backward(); opt.step()
    with torch.no_grad(): return encode([torch.tensor(z, device=DEV) for z in Zs])[0].argmax(1).cpu().numpy()
def scan(x, seed, k=20, epochs=300, w_ent=5.0):
    """SCAN clustering loss (Van Gansbeke et al., 2020) with a linear head on frozen features (no self-labelling)."""
    torch.manual_seed(seed); xm = x[main]
    nb = kneighbors_graph(xm, k, include_self=False).tocoo()
    X = torch.tensor(xm, device=DEV)
    a, b = torch.tensor(nb.row.astype(np.int64), device=DEV), torch.tensor(nb.col.astype(np.int64), device=DEV)
    head = nn.Linear(x.shape[1], K).to(DEV); opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        p = F.softmax(head(X), 1); opt.zero_grad()
        cons = -torch.log((p[a] * p[b]).sum(1) + 1e-8).mean()
        (cons - w_ent * torch.special.entr(p.mean(0)).sum()).backward(); opt.step()
    with torch.no_grad(): return head(torch.tensor(x, device=DEV)).argmax(1).cpu().numpy()
def sp_t(S):
    S = S.tocoo(); return torch.sparse_coo_tensor(np.vstack([S.row, S.col]).astype(np.int64), S.data.astype(np.float32), S.shape).to(DEV)
def dmon(x, seed, k=15, hidden=64, epochs=1000, collapse=1.0):
    """DMoN (Tsitsulin et al., JMLR 2023): 1-layer GCN on the kNN graph, spectral-modularity loss + collapse regulariser."""
    torch.manual_seed(seed); z = reduce(x[main], x, 32)
    A = kneighbors_graph(z[main], k, include_self=False); A = csr_matrix(((A + A.T) > 0).astype(np.float32))
    d = np.asarray(A.sum(1)).ravel(); m2 = float(d.sum()); n = A.shape[0]
    Ah = A + csr_matrix(np.eye(n, dtype=np.float32)); dh = np.asarray(Ah.sum(1)).ravel() ** -0.5
    Ah = csr_matrix(Ah.multiply(dh[:, None]).multiply(dh[None]))
    At, Aht = sp_t(A), sp_t(Ah)
    X = torch.tensor(z[main], device=DEV).float(); dt = torch.tensor(d, device=DEV).float()[:, None]
    W1, W2 = nn.Linear(32, hidden).to(DEV), nn.Linear(hidden, K).to(DEV)
    opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=1e-3)
    def fwd(): return F.softmax(W2(F.selu(torch.sparse.mm(Aht, W1(X)))), 1)
    for _ in range(epochs):
        C = fwd(); opt.zero_grad()
        mod = (torch.trace(C.T @ torch.sparse.mm(At, C)) - ((dt * C).sum(0) ** 2).sum() / m2) / m2
        col = torch.linalg.norm(C.sum(0)) / n * K ** 0.5 - 1
        (-mod + collapse * col).backward(); opt.step()
    with torch.no_grad(): ym = fwd().argmax(1).cpu().numpy()
    _, ym = np.unique(ym, return_inverse=True); return assign(z, ym)
run("DC TURTLE (CMRE sig, dino)", lambda: from_labels(base, [turtle([sig_c, dino], s) for s in range(3)]))
run("DC SCAN head (CMRE fusi)", lambda: from_labels(base, [scan(base, s) for s in range(3)]))
run("DC DMoN GNN (CMRE fusi)", lambda: from_labels(base, [dmon(base, s) for s in range(3)]))

# ---- FG: foreground-masked embeddings
if os.path.exists(f"{OUT}/emb_siglip2_fg.npy"):
    sig_f, dino_f = tta("siglip2_fg"), tta("dinov3hp_fg")
    run("FG mentah", lambda: evaluate(fuse(sig_f, dino_f)))
    run("FG CMRE", lambda: (lambda c: evaluate(fuse(c[0], dino_f), extra=dict(r=c[1])))(cmre(sig_f, [dino_f])))
    run("FG INLP", lambda: evaluate(fuse(inlp_x(sig_f), inlp_x(dino_f))))
    run("FG CMRE + asli CMRE", lambda: (lambda c: evaluate(fuse(c[0], dino_f, sig_c, dino), extra=dict(r=c[1])))(cmre(sig_f, [dino_f])))
print("NOVEL2 DONE")
