"""Assets for the paper (images only; labels are used only to colour plots and compute metrics). Writes OUT/paper/:
  src_*.jpg        6 example images per source (same object, mouse, where the source has it)
  bias_raw.jpg     Bangladesh phone photos that raw fusion puts into one cluster regardless of object (+ composition csv)
  bias_cmre.jpg    the v4 mouse cluster: mice from BDC, Kaan, Bangladesh together
  tsne.csv         t-SNE of the Kaan + Bangladesh shared-class images, raw fusion vs CMRE fusion (PCA-32)
  algo16.csv       K=16 on the v4 features: Spectral n_neighbors sweep, K-Means, Ward, GMM + internal indices
  ksweep.csv       Spectral K = 8..24 on the v4 features
  ood_knn.csv      kNN-OOD (k=10, PCA-32) flag rate per source and per Iliev / Shubha class"""
import os, numpy as np, pandas as pd
from PIL import Image
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from common import *
import rq1_metrics as M
import warnings; warnings.filterwarnings("ignore")

P = f"{OUT}/paper"; os.makedirs(P, exist_ok=True)
items = load_items(); main, style, lab, src, cat, shared = M.setup(items); N = len(items)
path = np.array([r["path"] for r in items]); rng = np.random.default_rng(0)
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def sheet(paths, fn, cols=8, s=200):
    rows = (len(paths) + cols - 1) // cols; canvas = Image.new("RGB", (cols * s, rows * s), "white")
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB"); im.thumbnail((s - 8, s - 8))
        canvas.paste(im, ((i % cols) * s + (s - im.width) // 2, (i // cols) * s + (s - im.height) // 2))
    canvas.save(fn, quality=90)
def pick(mask, n): idx = np.where(mask)[0]; return idx[rng.permutation(len(idx))[:n]]

# ---- example sheets per source (same object where available)
new = {t: pd.read_csv(f"{OUT}/{t}/items.csv", keep_default_na=False) for t in ("iliev", "shubha", "officehome")}
def newpick(t, col, val, n=6): d = new[t][new[t][col] == val]; return list(d.path.values[rng.permutation(len(d))[:n]])
S = {"bdc": list(path[pick(main & (lab == "mouse"), 6)]), "kaan": list(path[pick((src == "kaan") & (lab == "mouse"), 6)]),
     "bangla": list(path[pick((src == "bangla") & (lab == "mouse"), 6)]), "giz": list(path[pick(src == "giz", 6)]),
     "iliev": newpick("iliev", "cls", "Computer-Mouse"), "shubha": newpick("shubha", "cls", "Mouse"),
     "oh_product": list(new["officehome"].query("domain == 'product' and cls == 'Mouse'").path.values[:6]),
     "oh_realworld": list(new["officehome"].query("domain == 'realworld' and cls == 'Mouse'").path.values[:6])}
for k, v in S.items(): sheet(v, f"{P}/src_{k}.jpg", cols=6)

# ---- features
def tta(a): return nz(emb(a) + emb(a + "_flip"))
sig, pe, dino = tta("siglip2"), tta("pecoreg"), tta("dinov3hp")
raw = nz(np.hstack([sig, pe, dino])); z = np.load(f"{OUT}/z_final_v4.npy"); zr = reduce(raw[main], raw, 32)
A4 = pd.read_csv(f"{OUT}/final_v4/assignments.csv", dtype={"id": str}); v4 = dict(zip(A4.id, A4.cluster))
ym = np.array([v4[r["id"]] for r, m in zip(items, main) if m])
def spectral(zm, K, s=0, nn=15): return SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=nn, random_state=s, assign_labels="cluster_qr").fit_predict(zm)
def assign(zz, yy):
    zm = zz[main]; C = np.stack([zm[yy == c].mean(0) for c in np.unique(yy)])
    y = ((zz[:, None] - C[None]) ** 2).sum(-1).argmin(1); y[np.where(main)[0]] = yy; return y
def purity(y, m): return float(pd.crosstab(y[m], lab[m]).max(axis=1).sum() / m.sum())
def metrics(zz, y):
    return dict(**M.score_labels(y, main, style, lab, src, cat, shared), purity_bdc=purity(y, cat), purity_ext=purity(y, (~main) & (lab != "")),
                sil=float(silhouette_score(zz[main], y[main])), db=float(davies_bouldin_score(zz[main], y[main])), ch=float(calinski_harabasz_score(zz[main], y[main])))

# ---- raw fusion: where do the Bangladesh phone photos go?
yr = assign(zr, spectral(zr[main], 16)); b = (src == "bangla") & shared
top = np.bincount(yr[b]).argmax(); mb = b & (yr == top)
pd.Series(lab[mb]).value_counts().rename_axis("objek").reset_index(name="n").to_csv(f"{P}/bias_raw_composition.csv", index=False)
sheet(list(path[pick(mb, 16)]), f"{P}/bias_raw.jpg")
y4 = assign(z, ym); mm = y4 == 13
sheet(list(path[pick(mm & main & (lab == "mouse"), 6)]) + list(path[pick(mm & (src == "kaan") & (lab == "mouse"), 5)]) +
      list(path[pick(mm & (src == "bangla") & (lab == "mouse"), 5)]), f"{P}/bias_cmre.jpg")
print("PAPER bias: raw cluster", top, "holds", round(mb.sum() / b.sum(), 3), "of Bangladesh shared images", flush=True)

# ---- t-SNE (external shared classes, Kaan + Bangladesh)
e = shared & np.isin(src, ["kaan", "bangla"])
T = {k: TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(zz[e]) for k, zz in (("raw", zr), ("cmre", z))}
pd.DataFrame(dict(x_raw=T["raw"][:, 0], y_raw=T["raw"][:, 1], x_cmre=T["cmre"][:, 0], y_cmre=T["cmre"][:, 1], label=lab[e], source=src[e])).to_csv(f"{P}/tsne.csv", index=False)

# ---- K=16 comparisons on the v4 features and K sweep
rows = [dict(method="Spectral-kNN nn=15 (v4)", **metrics(z, y4), ari_v4=1.0)]
for nn in (5, 10, 20, 30):
    y = assign(z, spectral(z[main], 16, 0, nn)); rows.append(dict(method=f"Spectral-kNN nn={nn}", **metrics(z, y), ari_v4=float(ari(y[main], ym))))
for name, f in (("K-Means", lambda zm: kmeans(zm, 16, 0).labels_), ("Agglomerative-Ward", lambda zm: AgglomerativeClustering(16).fit_predict(zm)),
                ("GMM (diag)", lambda zm: GaussianMixture(16, covariance_type="diag", random_state=0).fit(zm).predict(zm))):
    y = assign(z, f(z[main])); rows.append(dict(method=name, **metrics(z, y), ari_v4=float(ari(y[main], ym))))
pd.DataFrame(rows).to_csv(f"{P}/algo16.csv", index=False); print(pd.DataFrame(rows).round(3).to_string(), flush=True)
pd.DataFrame([dict(K=K, **metrics(z, assign(z, spectral(z[main], K)))) for K in range(8, 25)]).to_csv(f"{P}/ksweep.csv", index=False)

# ---- kNN-OOD rates per source and per new-set class (same stacking as rq_fix2.py)
def stack(t):
    d = new[t]; k = (d.set != "control").values
    X = {b: nz(np.load(f"{OUT}/{t}/emb_{b}.npy").astype(np.float32)) + nz(np.load(f"{OUT}/{t}/emb_{b}_flip.npy").astype(np.float32)) for b in ("siglip2", "pecoreg", "dinov3hp")}
    return {b: nz(v[k]) for b, v in X.items()}, d[k]
out = []
for t in ("iliev", "shubha"):
    Xn, d = stack(t); n = len(d); mX = np.r_[main, np.zeros(n, bool)]
    s2, p2, d2 = (np.vstack([a, Xn[b]]) for a, b in ((sig, "siglip2"), (pe, "pecoreg"), (dino, "dinov3hp")))
    def cm(v, ref):
        res = v[mX] - Ridge(alpha=1.0).fit(ref[mX], v[mX]).predict(ref[mX]); p = PCA(64, random_state=0).fit(res)
        U = p.components_[:int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.5) + 1)].astype(np.float32); return nz(v - (v @ U.T) @ U)
    x = nz(np.hstack([cm(s2, d2), cm(p2, d2), d2])); zz = reduce(x[mX], x, 32)
    nnm = NearestNeighbors(n_neighbors=11).fit(zz[mX]); sm = nnm.kneighbors(zz[mX])[0][:, 10]; thr = np.percentile(sm, 95)
    so = nnm.kneighbors(zz[N:], 10)[0][:, 9]  # new-set images only
    if t == "iliev":
        for s in ("kaan", "bangla", "karan", "giz"):
            ss = nnm.kneighbors(zz[:N][src == s], 10)[0][:, 9]; out.append(dict(set="sumber", name=s, n=int((src == s).sum()), flag=float((ss > thr).mean())))
    for c in sorted(d.cls.unique()):
        k = (d.cls == c).values; out.append(dict(set=t, name=c, n=int(k.sum()), flag=float((so[k] > thr).mean()), bdc_class=d.posthoc_label.values[k][0]))
pd.DataFrame(out).to_csv(f"{P}/ood_knn.csv", index=False); print("PAPER DONE", flush=True)
