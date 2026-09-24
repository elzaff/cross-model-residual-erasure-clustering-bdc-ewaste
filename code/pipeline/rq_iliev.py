"""Held-out validation on Iliev crops (protocol locked in RECAP 3.11 before download; images only, labels post hoc).
The frozen v4 pipeline is refit exactly as before on D_main (CMRE r=64, PCA-32, K=16 clusters from final_v4) and the
Iliev crops are only projected to the nearest D_main centroid. Baselines use the same protocol with Spectral K=16.
  control  20 D_main images re-embedded: cosine with the stored embeddings must be ~1 (same preprocessing)
  (a) xsrc10: share of Iliev class-c crops in the Kaan majority cluster of c (10 BDC classes), + lenient (cluster's BDC
      majority label == c)   (b) purity on mapped classes   (c) subtypes CRT-TV->C3, Flat-Panel-TV->C7, Laptop->C9
  (d) OOD: AUROC of the centroid distance, in-distribution (mapped + Laptop) vs all other classes; per-class OOD rate at
      the final_max threshold (95th percentile of D_main distances)   (e) EU-6 WEEE category per cluster (author mapping
      of the 77 classes to Directive 2012/19/EU Annex III/IV; batteries and PCBs kept apart)   (f) baselines
Bootstrap 95% CIs resample Iliev crops. Writes OUT/iliev_results.csv, iliev_perclass.csv, iliev_cluster_eu6.csv."""
import os, numpy as np, pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from common import *
import rq1_metrics as M
import warnings; warnings.filterwarnings("ignore")

EU6 = {1: "Refrigerator Freezer Air-Conditioner Dehumidifier Cooled-Dispenser Cooling-Display",
       2: "CRT-Monitor CRT-TV Flat-Panel-Monitor Flat-Panel-TV Laptop Tablet",
       3: "Compact-Fluorescent-Lamps Straight-Tube-Fluorescent-Lamp LED-Bulb",
       4: "Washing-Machine Dishwasher Tumble-Dryer Stove Oven Range-Hood Photovoltaic-Panel Server Boiler Street-Lamp Electric-Bicycle Rotary-Mower",
       6: "Smartphone Bar-Phone Computer-Keyboard Computer-Mouse Printer Router Network-Switch HDD SSD USB-Flash-Drive Telephone-Set Calculator Smart-Watch Desktop-PC"}
EU6 = {c: str(k) for k, v in EU6.items() for c in v.split()} | {"Battery": "baterai", "PCB": "komponen"}  # all others: 5 (small equipment)
SUB = {"CRT-TV": 3, "Flat-Panel-TV": 7, "Laptop": 9}
CLS10 = ["battery", "keyboard", "microwave", "mobile", "mouse", "pcb", "player", "printer", "television", "washing_machine"]

items = load_items(); main, style, lab, src, cat, shared = M.setup(items); G = len(np.unique(style[main])); N = len(items)
NI = pd.read_csv(f"{OUT}/iliev/items.csv", keep_default_na=False); ctrl = (NI.set == "control").values; il = ~ctrl
cls_i, lab_i = NI.cls.values[il], NI.posthoc_label.values[il]; n_i = int(il.sum())
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def new(k): return nz(np.load(f"{OUT}/iliev/emb_{k}.npy").astype(np.float32))
X, rows = {}, []
for k in ("siglip2", "pecoreg", "dinov3hp"):
    old, nw = nz(emb(k) + emb(k + "_flip")), nz(new(k) + new(k + "_flip"))
    cs = (old[NI.cls.values[ctrl].astype(int)] * nw[ctrl]).sum(1)
    rows.append(dict(method=f"control {k}", ctrl_cos_min=float(cs.min()), ctrl_cos_mean=float(cs.mean())))
    print("CONTROL", k, "cos min", round(cs.min(), 4), "mean", round(cs.mean(), 4), "OK" if cs.min() > 0.98 else "!! MISMATCH", flush=True)
    X[k] = np.vstack([old, nw[il]])
mainX = np.r_[main, np.zeros(n_i, bool)]; I = np.arange(N, N + n_i)

def fuse(*a): return nz(np.hstack(a))
def cmre(v, ref):  # identical to rq_final2.py (PCA(64) cap -> r = 64)
    reg = Ridge(alpha=1.0).fit(ref[mainX], v[mainX]); res = v[mainX] - reg.predict(ref[mainX])
    p = PCA(64, random_state=0).fit(res); r = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.5) + 1)
    U = p.components_[:r].astype(np.float32); return nz(v - (v @ U.T) @ U)
def inlp_x(x, n_max=8):
    xf, s = x[mainX], style[main]; P = np.eye(x.shape[1], dtype=np.float32)
    for _ in range(n_max):
        W = LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T); P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s, cv=5, scoring="balanced_accuracy").mean() <= 1 / G: break
    return nz(x @ P)
sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
METHODS = {"usulan v4": lambda: fuse(cmre(sig, dino), cmre(pe, dino), dino), "fusi mentah": lambda: fuse(sig, pe, dino),
           "DINOv3-H+ saja": lambda: dino, "INLP x3": lambda: fuse(inlp_x(sig), inlp_x(pe), inlp_x(dino)),
           "ringan CMRE(pe)+dino": lambda: fuse(cmre(pe, dino), dino)}
A4 = pd.read_csv(f"{OUT}/final_v4/assignments.csv", dtype={"id": str}); v4 = dict(zip(A4.id, A4.cluster))
kaan = {c: (src == "kaan") & (lab == c) for c in CLS10}

def scores(y, d, thr, idx):
    yi, li, ci, di = y[I][idx], lab_i[idx], cls_i[idx], d[I][idx]
    km = {c: np.bincount(y[:N][kaan[c]]).argmax() for c in CLS10}
    bm = {k: pd.Series(lab[cat][y[:N][cat] == k]).mode()[0] for k in np.unique(y[:N][cat])}
    per = {c: (yi[li == c] == km[c]).mean() for c in CLS10 if (li == c).any()}
    le = {c: np.mean([bm.get(v) == c for v in yi[li == c]]) for c in CLS10 if (li == c).any()}
    m = li != ""; ind = np.isin(ci, list(set(ci[m])) + ["Laptop"])
    pur = pd.crosstab(yi[m], li[m]).max(axis=1).sum() / m.sum()
    return dict(xsrc10=float(np.mean(list(per.values()))), xsrc10_lenient=float(np.mean(list(le.values()))), purity=float(pur),
                ood_auroc=float(roc_auc_score(~ind, di)), ood_rate_in=float((di[ind] > thr).mean()), ood_rate_out=float((di[~ind] > thr).mean()),
                **{f"xsrc_{c}": float(v) for c, v in per.items()})

per_rows, eu_rows = [], []
for name, f in METHODS.items():
    x = f(); z = reduce(x[mainX], x, 32); zm = z[mainX]
    if name == "usulan v4":
        z4 = np.load(f"{OUT}/z_final_v4.npy"); dz = float(np.abs(z[:N] - z4).max())
        print("SANITY z vs z_final_v4 max|diff|", round(dz, 5), "OK" if dz < 1e-3 else "!! MISMATCH", flush=True)
        ym = np.array([v4[r["id"]] for r, m in zip(items, main) if m])
    else:
        ym = SpectralClustering(16, affinity="nearest_neighbors", n_neighbors=15, random_state=0, assign_labels="cluster_qr").fit_predict(zm)
    C = np.stack([zm[ym == c].mean(0) for c in range(16)])
    D2 = ((z[:, None] - C[None]) ** 2).sum(-1); y = D2.argmin(1); d = np.sqrt(D2.min(1)); y[np.where(mainX)[0]] = ym
    thr = np.percentile(d[mainX], 95); allidx = np.arange(n_i)
    r = dict(method=name, **scores(y, d, thr, allidx))
    if name == "usulan v4":
        yi = y[I]; r.update({f"sub_{k}": float((yi[cls_i == k] == v).mean()) for k, v in SUB.items() if (cls_i == k).any()})
        g = np.random.default_rng(0); B = [scores(y, d, thr, g.integers(0, n_i, n_i)) for _ in range(1000)]
        for k in ("xsrc10", "xsrc10_lenient", "purity", "ood_auroc"):
            r[f"{k}_lo"], r[f"{k}_hi"] = np.percentile([b[k] for b in B], [2.5, 97.5])
        ood = d[I] > thr
        for c in sorted(set(cls_i)):
            mk = cls_i == c; top = np.bincount(yi[mk], minlength=16)
            per_rows.append(dict(cls=c, mapped=lab_i[mk][0], eu6=EU6.get(c, "5"), n=int(mk.sum()), top_cluster=int(top.argmax()),
                                 top_share=float(top.max() / mk.sum()), ood_rate=float(ood[mk].mean())))
        for k in range(16):
            mk = (yi == k) & ~ood; e = pd.Series([EU6.get(c, "5") for c in cls_i[mk]]).value_counts()
            eu_rows.append(dict(cluster=k, n_iliev_in=int(mk.sum()), eu6_top=e.index[0] if len(e) else "", eu6_share=float(e.iloc[0] / e.sum()) if len(e) else np.nan,
                                classes=", ".join(f"{a}:{b}" for a, b in pd.Series(cls_i[mk]).value_counts().head(4).items())))
    rows.append(r); print("ILIEV", name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/iliev_results.csv", index=False)
pd.DataFrame(per_rows).to_csv(f"{OUT}/iliev_perclass.csv", index=False); pd.DataFrame(eu_rows).to_csv(f"{OUT}/iliev_cluster_eu6.csv", index=False)
print(pd.DataFrame(eu_rows).to_string(), flush=True); print("ILIEV DONE", flush=True)
