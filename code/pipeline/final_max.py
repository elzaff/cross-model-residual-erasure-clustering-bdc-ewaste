"""Final partition on the label-free pick (SigLIP2-So400m+INLP (+) DINOv3-H+ +INLP, PCA-32).
Method rule, applied uniformly to K and method choice: highest stability, silhouette breaks ties -> Spectral-kNN.
Outputs (OUT/final/): assignments.csv, cluster_profile.csv, bootstrap_ci.csv, galleries."""
import json, os, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from PIL import Image
from common import *
import rq1_metrics as M

TAG = os.environ.get("FINAL_TAG", "")  # "_v3" = proxy-free CMRE pipeline; "" = paper v2
F = os.path.join(OUT, "final" + TAG); os.makedirs(f"{F}/gallery", exist_ok=True)
items = load_items(); main, style, lab, src, cat, shared = M.setup(items)
cfg = json.load(open(f"{OUT}/final_config{TAG}.json")); K = cfg["K"]
z = np.load(f"{OUT}/z_final{TAG}.npy"); zm = z[main]
spec = lambda s: SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15, random_state=s, assign_labels="cluster_qr").fit_predict(zm)
ym = spec(0)
C = np.stack([zm[ym == c].mean(0) for c in range(K)])
d_all = ((z[:, None] - C[None]) ** 2).sum(-1); y = d_all.argmin(1); d = np.sqrt(d_all.min(1))
y[np.where(main)[0]] = ym  # main keeps its spectral labels; external images go to the nearest centroid
def align(ref, other):
    Cm = np.zeros((K, K)); np.add.at(Cm, (ref, other), 1); r, c = linear_sum_assignment(-Cm); mp = dict(zip(c, r)); return np.array([mp[o] for o in other])
votes = (align(ym, kmeans(zm, K, 0).labels_) == ym).astype(int) + (align(ym, AgglomerativeClustering(K).fit_predict(zm)) == ym)
cons = np.full(len(z), -1); cons[main] = votes  # 2 = spectral, k-means and ward all agree
thr = np.percentile(d[main], 95)
A = pd.DataFrame(dict(id=[r["id"] for r in items], set=[r["set"] for r in items], source=src, path=[r["path"] for r in items],
                      posthoc_label=lab, style=style, cluster=y, dist=d, consensus=cons, ood=d > thr,
                      white=[r["border_white"] >= .5 for r in items], side=[max(r["width"], r["height"]) for r in items]))
A.to_csv(f"{F}/assignments.csv", index=False)
m = A[A.set == "main"]; g = A[A.source == "giz"]
ext_lab = (~main) & (lab != "")
def pur(yy, ll, mk): return pd.crosstab(yy[mk], ll[mk]).max(axis=1).sum() / mk.sum()
# ---- bootstrap CIs (resample images with replacement) for the headline metrics of the final partition
rng = np.random.default_rng(0); B = []
for _ in range(1000):
    i = rng.choice(len(A), len(A), replace=True); yy, ll, ss, mm = y[i], lab[i], src[i], main[i]
    ca, sh = mm & (ll != ""), (~mm) & np.isin(ll, SHARED) & np.isin(ss, ["kaan", "bangla", "karan"])
    B.append(dict(purity_bdc=pur(yy, ll, ca), purity_ext=pur(yy, ll, (~mm) & (ll != "")),
                  xsrc=M.xsrc_agree(yy, ll, ss, sh), ext_obj_nmi=nmi(ll[sh], yy[sh])))
Bd = pd.DataFrame(B); pt = M.score_labels(y, main, style, lab, src, cat, shared)
point = dict(purity_bdc=pur(y, lab, cat), purity_ext=pur(y, lab, ext_lab), xsrc=pt["xsrc_agree"], ext_obj_nmi=pt["ext_obj_nmi"])
ci = pd.DataFrame({k: dict(point=float(v), lo=Bd[k].quantile(.025), hi=Bd[k].quantile(.975)) for k, v in point.items()}).T
ci.to_csv(f"{F}/bootstrap_ci.csv"); print(ci.round(3).to_string())
print("seed ARI (4 extra seeds):", round(np.mean([ari(ym, spec(s)) for s in range(1, 5)]), 3))
print("consensus 3/3, 2/3, <2:", [round(float((m.consensus == v).mean()), 3) for v in (2, 1, 0)])
print("OOD: main", round(m.ood.mean(), 3), "| giz", round(g.ood.mean(), 3), "|", {s: round(A[A.source == s].ood.mean(), 3) for s in ("kaan", "bangla", "karan")})
prof = []
for c in range(K):
    s = m[m.cluster == c]; e = A[(A.set == "ext") & (A.cluster == c)]; sl = s.posthoc_label.fillna("")
    prof.append(dict(cluster=c, n=len(s), test_n=int((s.source == "bdc_test").sum()), agree3=round((s.consensus == 2).mean(), 2),
                     white_bg=round(s.white.mean(), 2), thumb=round((s.side <= 200).mean(), 2), unlabeled=round((sl == "").mean(), 2),
                     top_bdc=s[sl != ""].posthoc_label.value_counts(normalize=True).round(2).head(3).to_dict(),
                     ext=e[e.posthoc_label.fillna("") != ""].posthoc_label.value_counts().head(3).to_dict(),
                     giz_n=int((e.source == "giz").sum()), giz_ood=int(((e.source == "giz") & e.ood).sum())))
P = pd.DataFrame(prof); P.to_csv(f"{F}/cluster_profile.csv", index=False); print(P.to_string(index=False))
def sheet(paths, fn, cols=8, T=112):
    S = Image.new("RGB", (cols * T, ((len(paths) + cols - 1) // cols) * T), "white")
    for i, p in enumerate(paths):
        im = Image.open(p); im.draft("RGB", (224, 224)); im = im.convert("RGB"); im.thumbnail((T - 4, T - 4)); S.paste(im, ((i % cols) * T + 2, (i // cols) * T + 2))
    S.save(fn, quality=80)
for c in range(K):
    s = m[m.cluster == c].sort_values("dist"); r = s.sample(min(8, len(s)), random_state=0)
    sheet(list(s.path[:16]) + list(r.path), f"{F}/gallery/c{c:02d}.jpg")  # 16 most central + 8 random
sheet(list(m[m.consensus == 0].sample(min(40, int((m.consensus == 0).sum())), random_state=0).path), f"{F}/gallery/ambiguous.jpg")
sheet(list(g[g.ood].sample(min(24, int(g.ood.sum())), random_state=0).path), f"{F}/gallery/giz_ood.jpg")
print("FINAL DONE")
