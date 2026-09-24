"""Round 8b.
  #1 r8b_reference_labelfree.csv  can the CMRE reference be chosen WITHOUT labels? For every reference with D_main+ext
     embeddings: label-free criteria on D_main only (cross-validated Ridge R^2 of VLM from reference, silhouette, seed ARI
     over 4 Spectral seeds, mean bootstrap Jaccard of clusters, 20 resamples) next to the post-hoc xsrc (BDC).
     The original selection rule (final_max.py) is: highest stability, silhouette breaks ties.
  #2 r8b_fast.csv  fast variants without PE-Core-G (the 3.7 img/s bottleneck): CMRE(SigLIP2)+DINOv3, raw SigLIP2+DINOv3,
     INLP(SigLIP2)+INLP(DINOv3), each vs v4 with paired bootstrap on BDC/Iliev/Shubha + kNN-OOD on Iliev / Bangladesh / GIZ.
Writes OUT/r8/."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import silhouette_score, roc_auc_score, r2_score
from sklearn.neighbors import NearestNeighbors
from stack import *
import warnings; warnings.filterwarnings("ignore")

R8 = f"{OUT}/r8"; os.makedirs(R8, exist_ok=True); g = np.random.default_rng(0)
sig, pe, dino = feat("siglip2"), feat("pecoreg"), feat("dinov3hp"); ym4 = v4_main()

def labelfree(x, m):
    z = reduce(x[m], x, 32); zm = z[m]; ys = [spectral(zm, 16, s) for s in range(4)]
    jac = []
    for _ in range(20):  # Hennig (2007) bootstrap Jaccard: best-match Jaccard of each reference cluster, averaged
        i = np.unique(g.integers(0, len(zm), len(zm))); yb = spectral(zm[i], 16, 0); y0 = ys[0][i]
        jac.append(np.mean([max(((y0 == c) & (yb == d)).sum() / ((y0 == c) | (yb == d)).sum() for d in range(16)) for c in range(16)]))
    return dict(sil=float(silhouette_score(zm, ys[0])), seed_ari=float(np.mean([ari(ys[0], y) for y in ys[1:]])), jaccard=float(np.mean(jac))), z

# ---------------- #1 label-free reference choice ----------------
REFS = {"dinov3hp": "DINOv3-H+ (v4)", "dinov2l_cls": "DINOv2-L", "dinov3l_cls": "DINOv3-L", "dinov3_cls": "DINOv3-B", "dinov3cnxl": "DINOv3-ConvNeXt-L",
        "dinov3_7b": "DINOv3-7B", "webdino300m": "Web-DINO-300M", "webdino1b": "Web-DINO-1B", "cnxv2h": "ConvNeXt V2-H (supervisi)"}
s_, p_, rows = sig[:N], pe[:N], []
for k, nm in REFS.items():
    R = nz(emb(k) + emb(k + "_flip")) if os.path.exists(f"{OUT}/emb_{k}_flip.npy") else emb(k)
    if len(R) != N: continue
    r2 = np.mean([r2_score(v[main], cross_val_predict(Ridge(alpha=1.0), R[main], v[main], cv=5)) for v in (s_, p_)])
    x = fuse(cmre(s_, R, m=main), cmre(p_, R, m=main), R); lf, z = labelfree(x, main)
    y = project(x, m=main)[0]
    rows.append(dict(ref=nm, r2_ridge=float(r2), **lf, xsrc_BDC=xsrc(y, "BDC"), purity_bdc=purity_bdc(y)))
    print("#1", rows[-1], flush=True); pd.DataFrame(rows).to_csv(f"{R8}/r8b_reference_labelfree.csv", index=False)
T = pd.DataFrame(rows)
for c in ("r2_ridge", "sil", "seed_ari", "jaccard"): print(f"rank of v4 by {c}: {int(T[c].rank(ascending=False)[T.ref == 'DINOv3-H+ (v4)'].iloc[0])}/{len(T)}",
                                                          f"| spearman with xsrc {T[c].corr(T.xsrc_BDC, method='spearman'):.2f}", flush=True)

# ---------------- #2 fast variants without PE-Core-G ----------------
ind_cls = set(NI.cls.values[NI.posthoc_label.values != ""]) | {"Laptop"}; ind_i = np.isin(NI.cls.values, list(ind_cls))
bang_bulb, giz = (src == "bangla") & (lab == "light_bulb"), src == "giz"
V = {"usulan v4": lambda: fuse(cmre(sig, dino), cmre(pe, dino), dino),
     "cepat CMRE(SigLIP2)+DINOv3": lambda: fuse(cmre(sig, dino), dino),
     "cepat fusi mentah SigLIP2+DINOv3": lambda: fuse(sig, dino),
     "cepat INLP(SigLIP2)+INLP(DINOv3)": lambda: fuse(inlp_x(sig), inlp_x(dino))}
Y, out = {}, []
for name, f in V.items():
    x = f(); y, z = project(x, ym4 if name == "usulan v4" else None); Y[name] = y
    zm = z[mainX]; nnm = NearestNeighbors(n_neighbors=11).fit(zm); thr = np.percentile(nnm.kneighbors(zm)[0][:, 10], 95)
    s = np.zeros(len(z)); s[~mainX] = nnm.kneighbors(z[~mainX], 10)[0][:, 9]; fl = s > thr; I = SETS["Iliev"][0]
    lf = labelfree(x, mainX)[0]
    out.append(dict(variant=name, **{f"xsrc_{d}": xsrc(y, d) for d in SETS}, purity_bdc=purity_bdc(y), **lf,
                    iliev_auroc=float(roc_auc_score(~ind_i, s[I])), flag_bangla_in=float(fl[:N][bang & ~bang_bulb[:N]].mean()),
                    flag_bangla_bulb=float(fl[:N][bang_bulb].mean()), flag_giz=float(fl[:N][giz].mean())))
    print("#2", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[-1].items()}, flush=True)
for d, (idx, _, _) in SETS.items():
    boots = [g.integers(0, len(idx), len(idx)) for _ in range(2000)]; b4 = np.array([xsrc(Y["usulan v4"], d, b) for b in boots])
    for r in out[1:]:
        dl = np.array([xsrc(Y[r["variant"]], d, b) for b in boots]) - b4
        r[f"delta_{d}"] = r[f"xsrc_{d}"] - out[0][f"xsrc_{d}"]; r[f"lo_{d}"], r[f"hi_{d}"] = np.percentile(dl, [2.5, 97.5])
pd.DataFrame(out).to_csv(f"{R8}/r8b_fast.csv", index=False); print(pd.DataFrame(out).round(3).T.to_string(), "\nR8B DONE", flush=True)
