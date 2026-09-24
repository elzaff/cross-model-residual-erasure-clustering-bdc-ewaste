"""Methodology fixes for the frozen v3 pipeline (images only; labels are used only for post-hoc metrics and probes).
Writes OUT/fix_results.csv (checkpointed per row). Clustering everywhere: PCA-32 + Spectral-kNN, metrics averaged over 3 seeds.
  BASE  simple baselines vs the proposal at K=13 and K=16 (single DINOv3 models, lighter fusions)
  PROBE does CMRE remove source information? On the external shared classes: source probe within class (Kaan vs Bangladesh,
        lower = less source information) and cross-source object transfer (fit on Kaan, test on Bangladesh, higher = better),
        in full and PCA-32 space; plus the removed subspace itself and a random subspace of the same size as control
  R     CMRE rule sensitivity. Note: in v3 the "50% variance" rule never triggered (64 PCs explain < 50% of the residual),
        so r = 64 (the PCA cap) everywhere; here r is swept explicitly and thresholds use the uncapped spectrum
  NN    kNN-graph sensitivity with per-class cross-source agreement (why n_neighbors=20 drops)
  KB    seed and bootstrap stability over K
  DIAG  label-free statistics per VLM|SSL pair (ridge R2, style-proxy probe) + single-model agreement, joined later with
        novel2/final2 results to answer "when does CMRE work?"
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, balanced_accuracy_score
from common import *
import rq1_metrics as M
import warnings; warnings.filterwarnings("ignore")

items = load_items(); main, style, lab, src, cat, shared = M.setup(items)
G = len(np.unique(style[main]))
EXT = shared & np.isin(src, ["kaan", "bangla"])
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def has(a): return os.path.exists(f"{OUT}/emb_{a}.npy") and os.path.exists(f"{OUT}/emb_{a}_flip.npy")
def tta(a): return nz(emb(a) + emb(a + "_flip"))
def fuse(*a): return nz(np.hstack(a))
def LR(): return LogisticRegression(max_iter=1000, class_weight="balanced")
def purity(y, m): return float(pd.crosstab(y[m], lab[m]).max(axis=1).sum() / m.sum())
def assign(z, ym):
    zm = z[main]; C = np.stack([zm[ym == c].mean(0) for c in np.unique(ym)])
    y = ((z[:, None] - C[None]) ** 2).sum(-1).argmin(1); y[np.where(main)[0]] = ym; return y
def spectral(zm, K, seed=0, nn=15):
    return SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=nn, random_state=seed, assign_labels="cluster_qr").fit_predict(zm)
def xsrc_cls(y):
    out = {}
    for c in SHARED:
        kc, bc = shared & (src == "kaan") & (lab == c), shared & (src == "bangla") & (lab == c)
        if kc.sum() and bc.sum(): out[f"xsrc_{c}"] = float((y[bc] == np.bincount(y[kc]).argmax()).mean())
    return out
def evaluate(x, K=13, nn=15, extra=None):
    z = reduce(x[main], x, 32); yms = [spectral(z[main], K, s, nn) for s in range(3)]
    ms = []
    for ym in yms:
        y = assign(z, ym)
        ms.append(dict(**M.score_labels(y, main, style, lab, src, cat, shared), purity_bdc=purity(y, cat),
                       purity_ext=purity(y, (~main) & (lab != "")), **xsrc_cls(y)))
    r = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    r.update(xsrc_std=float(np.std([m["xsrc_agree"] for m in ms])), seed_ari=float(np.mean([ari(yms[0], o) for o in yms[1:]])),
             sil=float(silhouette_score(z[main], yms[0])), K=K, nn=nn)
    return {**r, **(extra or {})}
def style_probe(x):  # label-free pixel proxy (white border x resolution), chance = 1/G
    return float(cross_val_score(LR(), x[main], style[main], cv=5, scoring="balanced_accuracy").mean())
def probes(x, pca=False):
    if pca: x = reduce(x[main], x, 32)
    sp = [cross_val_score(LR(), x[m], src[m], cv=5, scoring="balanced_accuracy").mean()
          for m in (EXT & (lab == c) for c in SHARED) if len(np.unique(src[m])) == 2]
    tr, te = EXT & (src == "kaan"), EXT & (src == "bangla")
    return dict(src_probe=float(np.mean(sp)), obj_transfer=float(balanced_accuracy_score(lab[te], LR().fit(x[tr], lab[tr]).predict(x[te]))))
def inlp_x(x, n_max=8):
    xf, s = x[main], style[main]; P = np.eye(x.shape[1], dtype=np.float32)
    for _ in range(n_max):
        W = LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T); P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s, cv=5, scoring="balanced_accuracy").mean() <= 1 / G: break
    return nz(x @ P)
def cmre(v, ref, r=64, thr=None):
    """Remove from v the top-r principal directions of its residual after ridge-predicting v from ref (r=64 reproduces v3)."""
    reg = Ridge(alpha=1.0).fit(ref[main], v[main]); res = v[main] - reg.predict(ref[main])
    p = PCA(random_state=0).fit(res)
    if thr is not None: r = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), thr) + 1)
    U = p.components_[:r].astype(np.float32)
    r2 = 1 - (res ** 2).sum() / ((v[main] - v[main].mean(0)) ** 2).sum()
    return nz(v - (v @ U.T) @ U), dict(r=r, ridge_r2=float(r2)), U

PART = f"{OUT}/fix_results.csv"
rows = pd.read_csv(PART).to_dict("records") if os.path.exists(PART) else []
done = {r["method"] for r in rows}
def run(name, fn):
    if name in done: return
    r = fn(); r["method"] = name; rows.append(r); pd.DataFrame(rows).to_csv(PART, index=False)
    print("FIX", name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

sig, pe, dino = tta("siglip2"), tta("pecoreg"), tta("dinov3hp")
sig_c, s_info, U_s = cmre(sig, dino); pe_c, p_info, U_p = cmre(pe, dino)
best = fuse(sig_c, pe_c, dino)

# ---- BASE: is the 3-model pipeline needed?
BASE = {"usulan CMRE(sig)+CMRE(pe)+dinoH+": lambda: best,
        "DINOv3-H+ saja": lambda: dino,
        "DINOv3-B saja": lambda: tta("dinov3_cls"),
        "DINOv3-7B saja": lambda: tta("dinov3_7b"),
        "ringan CMRE(pe)+dinoH+": lambda: fuse(pe_c, dino),
        "ringan CMRE(sig)+dinoH+": lambda: fuse(sig_c, dino),
        "PE+dinoH+ mentah": lambda: fuse(pe, dino)}
for K in (13, 16):
    for name, f in BASE.items(): run(f"BASE {name} K={K}", lambda: evaluate(f(), K=K))

# ---- PROBE: what does CMRE remove?
rng = np.random.default_rng(0)
def rand_sub(v, r): Q, _ = np.linalg.qr(rng.standard_normal((v.shape[1], r)).astype(np.float32)); return nz(v @ Q)
PROBE = {"SigLIP2 mentah": lambda: sig, "SigLIP2 CMRE": lambda: sig_c, "SigLIP2 subruang dibuang": lambda: nz(sig @ U_s.T),
         "SigLIP2 subruang acak (kontrol)": lambda: rand_sub(sig, U_s.shape[0]),
         "PE mentah": lambda: pe, "PE CMRE": lambda: pe_c, "PE subruang dibuang": lambda: nz(pe @ U_p.T),
         "PE subruang acak (kontrol)": lambda: rand_sub(pe, U_p.shape[0]), "DINOv3-H+": lambda: dino,
         "fusi mentah": lambda: fuse(sig, pe, dino), "fusi INLPx3": lambda: fuse(inlp_x(sig), inlp_x(pe), inlp_x(dino)),
         "fusi usulan": lambda: best}
for name, f in PROBE.items():
    run(f"PROBE {name}", lambda: (lambda x: dict(style_probe=style_probe(x), **probes(x),
                                                  **{f"pca32_{k}": v for k, v in probes(x, pca=True).items()}))(f()))

# ---- R: CMRE rule sensitivity (same rule for SigLIP2 and PE)
for r in (4, 8, 16, 32, 64, 128, 256):
    run(f"R r={r}", lambda: evaluate(fuse(cmre(sig, dino, r=r)[0], cmre(pe, dino, r=r)[0], dino), extra=dict(r=r)))
for thr in (0.3, 0.5, 0.7):
    run(f"R thr={thr}", lambda: (lambda a, b: evaluate(fuse(a[0], b[0], dino), extra=dict(r=a[1]["r"], r_pe=b[1]["r"], thr=thr)))(
        cmre(sig, dino, thr=thr), cmre(pe, dino, thr=thr)))

# ---- NN: graph sensitivity, per-class agreement
for nn in (5, 10, 15, 20, 25, 30):
    run(f"NN n_neighbors={nn} K=13", lambda: evaluate(best, K=13, nn=nn))

# ---- KB: stability over K
zm = reduce(best[main], best, 32)[main]
def boot_stab(K, B=10):
    y0, g, out = spectral(zm, K), np.random.default_rng(K), []
    for _ in range(B):
        i = np.sort(g.choice(len(zm), int(0.8 * len(zm)), replace=False)); out.append(ari(y0[i], spectral(zm[i], K)))
    return float(np.mean(out)), float(np.std(out))
for K in range(10, 21):
    run(f"KB K={K}", lambda: (lambda b: dict(K=K, boot_stab=b[0], boot_std=b[1],
                                             seed_ari=float(np.mean([ari(spectral(zm, K, 0), spectral(zm, K, s)) for s in (1, 2)]))))(boot_stab(K)))

# ---- DIAG: when does CMRE work?
PAIRS = [("siglip2", "dinov3hp"), ("siglip2", "dinov2l_cls"), ("siglip2", "dinov3_7b"), ("pecoreg", "dinov3hp"), ("clip", "dinov3hp"),
         ("clipcnxxxl", "dinov3hp"), ("aimv2", "dinov3hp"), ("metaclipl", "webdino300m"), ("metacliph", "webdino1b")]
for vk, sk in PAIRS:
    if not (has(vk) and has(sk)): print("skip pair", vk, sk, flush=True); continue
    def diag(vk=vk, sk=sk):
        xv, xs = tta(vk), tta(sk); xc, info, _ = cmre(xv, xs)
        sv, ss, sc = probes(xv), probes(xs), probes(xc)
        ev, es = evaluate(xv, K=16), evaluate(xs, K=16)
        return dict(vlm=vk, ssl=sk, **info, style_vlm=style_probe(xv), style_ssl=style_probe(xs), style_cmre=style_probe(xc),
                    src_vlm=sv["src_probe"], src_ssl=ss["src_probe"], src_cmre=sc["src_probe"],
                    xsrc_vlm=ev["xsrc_agree"], xsrc_ssl=es["xsrc_agree"])
    run(f"DIAG {vk}|{sk}", diag)
# ---- NBR: unsupervised view of source bias -- a supervised probe finds the source in every representation (~0.99), so
#      measure what clustering actually sees: for each external shared-class image, among its k nearest external
#      shared-class neighbours (PCA-32 space), the share from the same source (homophily; chance = source share)
#      and the share with the same object (higher = better)
from sklearn.neighbors import NearestNeighbors
def nbr(x, k=10):
    z = reduce(x[main], x, 32)[EXT]; s, o = src[EXT], lab[EXT]
    idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z, return_distance=False)[:, 1:]
    b = s == "bangla"
    return dict(nbr_same_src=float((s[idx] == s[:, None]).mean()), nbr_same_obj=float((o[idx] == o[:, None]).mean()),
                nbr_bangla_same_src=float((s[idx][b] == "bangla").mean()), nbr_bangla_same_obj=float((o[idx][b] == o[b][:, None]).mean()),
                src_chance=float(np.mean([(s == v).mean() ** 2 for v in np.unique(s)]) * len(np.unique(s))))
for name, f in {**PROBE, "PE+dinoH+ mentah": lambda: fuse(pe, dino), "ringan CMRE(pe)+dinoH+": lambda: fuse(pe_c, dino)}.items():
    if "subruang" in name: continue
    run(f"NBR {name}", lambda: nbr(f()))
# ---- LEN: is a cross-source "miss" a wrong object or a sub-type split? A Bangladesh image counts as consistent if its
#      cluster's majority BDC filename label equals its class (lenient), next to the strict Kaan-majority-cluster rule
def lenient(y):
    maj = {c: pd.Series(lab[cat][y[cat] == c]).mode()[0] for c in np.unique(y[cat])}
    out = {}
    for c in SHARED:
        b = shared & (src == "bangla") & (lab == c)
        if b.sum(): out[f"len_{c}"] = float(np.mean([maj.get(v) == c for v in y[b]]))
    out["xsrc_lenient"] = float(np.mean(list(out.values()))); return out
for nn in (15, 20, 30):
    def len_row(nn=nn):
        z = reduce(best[main], best, 32); y = assign(z, spectral(z[main], 13, 0, nn))
        return dict(nn=nn, **lenient(y), **xsrc_cls(y))
    run(f"LEN n_neighbors={nn} K=13", len_row)
print("FIX DONE", flush=True)
