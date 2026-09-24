"""Finalisation round for the proxy-free pipeline CMRE(SigLIP2) + CMRE(PE-Core) + DINOv3-H+ (images only, no labels).
Writes OUT/final2_results.csv (checkpointed per row), OUT/z_final_v3.npy and OUT/final_config_v3.json (read by final_max.py
with FINAL_TAG=_v3).
  DC  data control: MetaCLIP (language) vs Web-DINO (no language), both trained on MetaCLIP web data -- Table-2 protocol
      (K-Means K=14, PCA-32) raw / INLP, plus CMRE with the same-data SSL reference (Spectral K=16)
  FB  fair baselines with the same three backbones (INLP x3, LEACE x3, INLP on VLMs only)
  K   label-free K choice for the new pipeline: highest silhouette among K with seed and bootstrap stability >= 0.95
  KG  kNN-graph sensitivity (n_neighbors) and other algorithms (K-Means, Ward, GMM) on the same features
"""
import os, json, numpy as np, pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from common import *
import rq1_metrics as M
import warnings; warnings.filterwarnings("ignore")

K0 = 16
items = load_items(); main, style, lab, src, cat, shared = M.setup(items)
G = len(np.unique(style[main]))
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def has(a): return os.path.exists(f"{OUT}/emb_{a}.npy") and os.path.exists(f"{OUT}/emb_{a}_flip.npy")
def tta(a): return nz(emb(a) + emb(a + "_flip"))
def probe(x): return cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), x[main], style[main], cv=5, scoring="balanced_accuracy").mean()
def purity(y, m): return float(pd.crosstab(y[m], lab[m]).max(axis=1).sum() / m.sum())
def full_metrics(y):
    return dict(**M.score_labels(y, main, style, lab, src, cat, shared), purity_bdc=purity(y, cat), purity_ext=purity(y, (~main) & (lab != "")))
def assign(z, ym):
    zm = z[main]; C = np.stack([zm[ym == c].mean(0) for c in np.unique(ym)])
    y = ((z[:, None] - C[None]) ** 2).sum(-1).argmin(1); y[np.where(main)[0]] = ym; return y
def spectral(zm, K=K0, seed=0, nn=15):
    return SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=nn, random_state=seed, assign_labels="cluster_qr").fit_predict(zm)
def evaluate(x, K=K0, nn=15, extra=None):
    z = reduce(x[main], x, 32); yms = [spectral(z[main], K, s, nn) for s in range(3)]
    return dict(seed_ari=float(np.mean([ari(yms[0], o) for o in yms[1:]])), sil=silhouette_score(z[main], yms[0]),
                probe_style=probe(x), **full_metrics(assign(z, yms[0])), **(extra or {}))
def labels_eval(x, fit):
    """Other algorithms on the same PCA-32 features: fit(zm, seed) -> labels of D_main."""
    z = reduce(x[main], x, 32); yms = [fit(z[main], s) for s in range(3)]
    return dict(seed_ari=float(np.mean([ari(yms[0], o) for o in yms[1:]])), sil=silhouette_score(z[main], yms[0]), **full_metrics(assign(z, yms[0])))
def inlp_x(x, n_max=8):
    xf, s = x[main], style[main]; P = np.eye(x.shape[1], dtype=np.float32)
    for _ in range(n_max):
        W = LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T); P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s, cv=5, scoring="balanced_accuracy").mean() <= 1 / G: break
    return nz(x @ P)
def leace(x):
    xm = x[main].astype(np.float64); mu = xm.mean(0); Xc = xm - mu
    Z = np.eye(G)[style[main]]; Z = Z - Z.mean(0)
    Sxx = Xc.T @ Xc / len(Xc) + 1e-4 * np.eye(Xc.shape[1]); Sxz = Xc.T @ Z / len(Xc)
    ev, V = np.linalg.eigh(Sxx); W = V @ np.diag(ev ** -0.5) @ V.T; Winv = V @ np.diag(ev ** 0.5) @ V.T
    Q, _ = np.linalg.qr(W @ Sxz); Pa = Q @ Q.T
    return nz(((x - mu) - (Winv @ Pa @ W @ (x - mu).T).T + mu).astype(np.float32))
def cmre(v, ref):
    reg = Ridge(alpha=1.0).fit(ref[main], v[main]); res = v[main] - reg.predict(ref[main])
    p = PCA(64, random_state=0).fit(res); r = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.5) + 1)
    U = p.components_[:r].astype(np.float32); return nz(v - (v @ U.T) @ U), r
def fuse(*a): return nz(np.hstack(a))

PART = f"{OUT}/final2_results.csv"
rows = pd.read_csv(PART).to_dict("records") if os.path.exists(PART) else []
done = {r["method"] for r in rows}
def run(name, fn):
    if name in done: return
    r = fn(); r["method"] = name; rows.append(r); pd.DataFrame(rows).to_csv(PART, index=False)
    print("FINAL2", name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

# ---- DC: data-controlled language vs no-language pair (Table-2 protocol for single models)
def table2(x): return dict(probe_style=probe(x), **M.score(x, None, 14, 32, main, style, lab, src, cat, shared))
SINGLE = [("dinov2l_cls", "DINOv2-L"), ("clip", "CLIP-L/14"), ("siglip2", "SigLIP2-So400m"),
          ("webdino300m", "Web-DINO 300M"), ("webdino1b", "Web-DINO 1B"), ("metaclipl", "MetaCLIP-L/14"), ("metacliph", "MetaCLIP-H/14")]
for key, name in SINGLE:
    if not has(key): print("skip", key, flush=True); continue
    x = tta(key)
    run(f"DC {name} mentah (T2)", lambda: table2(x))
    run(f"DC {name} INLP (T2)", lambda: table2(inlp_x(x)))
for vk, sk, name in [("metaclipl", "webdino300m", "MetaCLIP-L | Web-DINO 300M"), ("metacliph", "webdino1b", "MetaCLIP-H | Web-DINO 1B")]:
    if not (has(vk) and has(sk)): continue
    xv, xs = tta(vk), tta(sk)
    run(f"DC {name} fusi mentah", lambda: evaluate(fuse(xv, xs)))
    run(f"DC {name} INLP", lambda: evaluate(fuse(inlp_x(xv), inlp_x(xs))))
    run(f"DC {name} CMRE", lambda: (lambda c: evaluate(fuse(c[0], xs), extra=dict(r=c[1])))(cmre(xv, xs)))

# ---- FB: fair three-backbone baselines
sig, pe, dino = tta("siglip2"), tta("pecoreg"), tta("dinov3hp")
(sig_c, r_s), (pe_c, r_p) = cmre(sig, dino), cmre(pe, dino)
best = fuse(sig_c, pe_c, dino)
run("FB CMRE(sig)+CMRE(pe)+dino [usulan]", lambda: evaluate(best, extra=dict(r=r_s, r_pe=r_p)))
run("FB INLP(sig)+INLP(pe)+INLP(dino)", lambda: evaluate(fuse(inlp_x(sig), inlp_x(pe), inlp_x(dino))))
run("FB INLP(sig)+INLP(pe)+dino", lambda: evaluate(fuse(inlp_x(sig), inlp_x(pe), dino)))
run("FB LEACE(sig)+LEACE(pe)+LEACE(dino)", lambda: evaluate(fuse(leace(sig), leace(pe), leace(dino))))
run("FB fusi mentah sig+pe+dino", lambda: evaluate(fuse(sig, pe, dino)))

# ---- K: label-free K for the proposed pipeline
z = reduce(best[main], best, 32); zm = z[main]; rng = np.random.default_rng(0)
def boot_stab(K, B=5):
    y0 = spectral(zm, K); out = []
    for _ in range(B):
        i = np.sort(rng.choice(len(zm), int(0.8 * len(zm)), replace=False)); out.append(ari(y0[i], spectral(zm[i], K)))
    return float(np.mean(out))
for K in (12, 14, 16, 18, 20):
    run(f"K={K}", lambda: evaluate(best, K=K, extra=dict(K=K, boot_stab=boot_stab(K))))
kr = pd.DataFrame([r for r in rows if str(r["method"]).startswith("K=")])
ok = kr[(kr.seed_ari >= 0.95) & (kr.boot_stab >= 0.95)]
K_best = int((ok if len(ok) else kr).sort_values("sil", ascending=False).iloc[0]["K"])
print("K rule (sil max | seed & boot stab >= .95) ->", K_best, flush=True)
K_best = int(os.environ.get("K_OVERRIDE", K_best))  # final: median SCMax K over 3 seeds (rq_scmax.py)
print("K used ->", K_best, flush=True)

# ---- KG: graph sensitivity and other algorithms on the same features
for nn in (5, 10, 15, 20, 30):
    run(f"KG spectral n_neighbors={nn} K={K_best}", lambda: evaluate(best, K=K_best, nn=nn, extra=dict(nn=nn)))
run(f"KG K-Means K={K_best}", lambda: labels_eval(best, lambda zz, s: kmeans(zz, K_best, s).labels_))
run(f"KG Agglomerative-Ward K={K_best}", lambda: labels_eval(best, lambda zz, s: AgglomerativeClustering(K_best).fit_predict(zz)))
run(f"KG GMM (diag) K={K_best}", lambda: labels_eval(best, lambda zz, s: GaussianMixture(K_best, covariance_type="diag", random_state=s).fit(zz).predict(zz)))

np.save(f"{OUT}/z_final_v3.npy", z)
json.dump(dict(rep="CMRE(SigLIP2-So400m|DINOv3-H+) + CMRE(PE-Core-G|DINOv3-H+) + DINOv3-H+", K=K_best, method="Spectral-kNN",
               r_siglip2=r_s, r_pecore=r_p), open(f"{OUT}/final_config_v3.json", "w"), indent=1)
print("FINAL2 DONE", flush=True)
