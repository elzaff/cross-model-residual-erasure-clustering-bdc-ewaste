"""Novel debiasing / clustering variants, all at K=16 on SigLIP2-So400m (VLM) + DINOv3-H+ (SSL), compared with the
current final (INLP-heuristic both -> fusion -> Spectral-kNN). Writes OUT/novel_results.csv (checkpointed per method).
  M1 CMRE   cross-model residual erasure: the part of the VLM embedding not linearly predictable from the SSL embedding
            is assumed to carry source; its top-r principal directions are erased (no style proxy at all)
  M2 CCE    cluster-conditional erasure: INLP fitted on cluster-centred features, so only within-cluster style
            variation is removed and between-object separation is kept; re-cluster, 2 rounds
  M3 TURTLE-SI  official TURTLE loop + lambda * MI(soft cluster labels ; style group) penalty
  M4 XSG    cross-style kNN graph: Spectral clustering whose same-style edges are down-weighted by alpha
  M5 LEACE  closed-form least-squares concept erasure (Belrose et al., 2023) instead of INLP
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.sparse import coo_matrix
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
def evaluate(name, x, affinity_fn=None, extra=None):
    z = reduce(x[main], x, 32); aff = affinity_fn(z) if affinity_fn else None
    yms = [spectral(z, s, aff) for s in range(3)]
    y = assign(z, yms[0])
    return dict(method=name, seed_ari=float(np.mean([ari(yms[0], o) for o in yms[1:]])), sil=silhouette_score(z[main], yms[0]),
                probe_style=probe(x), **full_metrics(y), **(extra or {}))

def inlp_P(xf, s, n_max=8):
    """INLP projection matrix, iterating until the style probe is at chance."""
    P = np.eye(xf.shape[1], dtype=np.float32)
    for n in range(1, n_max + 1):
        W = LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T); P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s, cv=5, scoring="balanced_accuracy").mean() <= 1 / G: break
    return P
def leace(x, s):
    """LEACE (Belrose et al., 2023): x' = x - W^+ P_{W S_xz} W (x - mu), W = S_xx^{-1/2}; fitted on D_main."""
    xm = x[main].astype(np.float64); mu = xm.mean(0); Xc = xm - mu
    Z = np.eye(G)[s[main]]; Z = Z - Z.mean(0)
    Sxx = Xc.T @ Xc / len(Xc) + 1e-4 * np.eye(Xc.shape[1]); Sxz = Xc.T @ Z / len(Xc)
    ev, V = np.linalg.eigh(Sxx); W = V @ np.diag(ev ** -0.5) @ V.T; Winv = V @ np.diag(ev ** 0.5) @ V.T
    Q, _ = np.linalg.qr(W @ Sxz); Pa = Q @ Q.T
    return nz(((x - mu) - (Winv @ Pa @ W @ (x - mu).T).T + mu).astype(np.float32))
def fuse(a, b): return nz(np.hstack([a, b]))

sig, dino = tta("siglip2"), tta("dinov3hp")
PART = f"{OUT}/novel_results.csv"
rows = pd.read_csv(PART).to_dict("records") if os.path.exists(PART) else []
done = {r["method"] for r in rows}
def run(name, fn):
    if name in done: return
    r = fn(); r["method"] = name; rows.append(r); pd.DataFrame(rows).to_csv(PART, index=False)
    print("NOVEL", name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

# reference: current final (INLP heuristic on both) and raw fusion
sig_i, dino_i = nz(sig @ inlp_P(sig[main], style[main])), nz(dino @ inlp_P(dino[main], style[main]))
run("REF final: INLP(sig)+INLP(dino)", lambda: evaluate("", fuse(sig_i, dino_i)))
run("REF fusi mentah", lambda: evaluate("", fuse(sig, dino)))

# ---- M1 cross-model residual erasure (no style proxy)
reg = Ridge(alpha=1.0).fit(dino[main], sig[main]); res = sig[main] - reg.predict(dino[main])
pca_r = PCA(64, random_state=0).fit(res); U = pca_r.components_.astype(np.float32)
cum = np.cumsum(pca_r.explained_variance_ratio_); r_rule = int(np.searchsorted(cum, 0.5) + 1)
pc1 = res @ U[0]
print("CMRE: residual/VLM variance =", round(float(res.var(0).sum() / sig[main].var(0).sum()), 3),
      "| NMI(PC1 quintiles, style) =", round(nmi(style[main], np.digitize(pc1, np.quantile(pc1, [.2, .4, .6, .8]))), 3),
      "| r(50% rule) =", r_rule, flush=True)
for r in sorted({2, 4, 8, 16, 32, r_rule}):
    Ur = U[:r]; sig_c = nz(sig - (sig @ Ur.T) @ Ur)
    tag = f"M1 CMRE r={r}" + (" (aturan 50%)" if r == r_rule else "")
    run(tag, lambda: evaluate("", fuse(sig_c, dino), extra=dict(r=r)))
sig_cmre = nz(sig - (sig @ U[:r_rule].T) @ U[:r_rule])

# ---- M2 cluster-conditional erasure (applied cumulatively to the current representation)
def cce(xs, xd, rounds=2):
    for _ in range(rounds):
        z = reduce(fuse(xs, xd)[main], fuse(xs, xd), 32); ym = spectral(z)
        def cond(x):
            mu = np.stack([x[main][ym == c].mean(0) for c in range(K)])
            return nz(x @ inlp_P(x[main] - mu[ym], style[main]))
        xs, xd = cond(xs), cond(xd)
    return fuse(xs, xd)
run("M2 CCE (2 ronde)", lambda: evaluate("", cce(sig_i, dino_i)))

# ---- M4 cross-style kNN graph (same-style edges down-weighted)
def xs_affinity(alpha):
    def f(z):
        A = kneighbors_graph(z[main], 15, mode="connectivity", include_self=False).tocoo(); s = style[main]
        B = coo_matrix((np.where(s[A.row] == s[A.col], alpha, 1.0), (A.row, A.col)), shape=A.shape).tocsr()
        return 0.5 * (B + B.T)
    return f
for alpha in (0.3, 0.6):
    run(f"M4 XSG alpha={alpha} (fusi final)", lambda: evaluate("", fuse(sig_i, dino_i), affinity_fn=xs_affinity(alpha)))
    run(f"M4 XSG alpha={alpha} (fusi mentah)", lambda: evaluate("", fuse(sig, dino), affinity_fn=xs_affinity(alpha)))

# ---- M5 LEACE instead of INLP
run("M5 LEACE(sig)+LEACE(dino)", lambda: evaluate("", fuse(leace(sig, style), leace(dino, style))))
run("M5 LEACE(sig)+dino mentah", lambda: evaluate("", fuse(leace(sig, style), dino)))

# ---- combinations of the new pieces
run("M1+M4 CMRE(aturan)+XSG0.3", lambda: evaluate("", fuse(sig_cmre, dino), affinity_fn=xs_affinity(0.3)))
run("M1+M2 CMRE(aturan) lalu CCE", lambda: evaluate("", cce(sig_cmre, dino)))

# ---- M3 style-invariant TURTLE (official loop + MI penalty)
def turtle_si(Zs, seed, lam, T=6000, M_in=10, gamma=10., lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    Ztr = [torch.tensor(z[main], device=DEV) for z in Zs]; dims = [z.shape[1] for z in Zs]
    S = torch.tensor(np.eye(G)[style[main]], dtype=torch.float32, device=DEV)
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
        Pkg = labels.T @ S / len(S); Pk, Pg = Pkg.sum(1, keepdim=True), Pkg.sum(0, keepdim=True)
        mi = (Pkg * (torch.log(Pkg + 1e-10) - torch.log(Pk @ Pg + 1e-10))).sum()
        (err - gamma * sum(torch.special.entr(p.mean(0)).sum() for p in per) + lam * mi).backward(); opt.step()
    with torch.no_grad(): return encode([torch.tensor(z, device=DEV) for z in Zs])[0].argmax(1).cpu().numpy()
for lam in (10.0, 30.0):
    def fn(lam=lam):
        ys = [turtle_si([sig, dino], s, lam) for s in range(3)]
        return dict(seed_ari=float(np.mean([ari(ys[0][main], o[main]) for o in ys[1:]])), **full_metrics(ys[0]))
    run(f"M3 TURTLE-SI lambda={lam:g} (mentah)", fn)
print("NOVEL DONE")
