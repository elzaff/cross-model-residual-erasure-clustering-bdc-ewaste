"""Fresh test set for the round-7 choices (protocol locked in RECAP 3.11 before download): Roboflow Shubha crops,
projected onto the frozen v4 partition. OOD score = kNN k=10 in PCA-32 (chosen on Iliev) vs the v4 centroid distance;
threshold = 95th percentile on D_main. Mouse/Keyboard/Mobile/PCB are in-distribution, Remote/Dryer/Headphone/Modem/
Pendrive out-of-distribution, Computer/Electronics excluded. Variants compared on xsrc4. Writes OUT/shubha_results.csv."""
import numpy as np, pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from common import *
import rq1_metrics as M
import warnings; warnings.filterwarnings("ignore")

C4 = ["keyboard", "mobile", "mouse", "pcb"]; OODC = ["Remote", "Dryer", "Headphone", "Modem", "Pendrive"]
items = load_items(); main, style, lab, src, cat, shared = M.setup(items); G = len(np.unique(style[main])); N = len(items)
NS = pd.read_csv(f"{OUT}/shubha/items.csv", keep_default_na=False); ctrl = (NS.set == "control").values
keep = ~ctrl & (NS.posthoc_label != "EXCLUDE").values; cls_s, lab_s = NS.cls.values[keep], NS.posthoc_label.values[keep]
n_s = int(keep.sum()); I = np.arange(N, N + n_s); ind, out = np.isin(lab_s, C4), np.isin(cls_s, OODC)
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def new(k): return nz(np.load(f"{OUT}/shubha/emb_{k}.npy").astype(np.float32))
X, rows = {}, []
for k in ("siglip2", "pecoreg", "dinov3hp"):
    old, nw = nz(emb(k) + emb(k + "_flip")), nz(new(k) + new(k + "_flip")); cs = (old[NS.cls.values[ctrl].astype(int)] * nw[ctrl]).sum(1)
    print("CONTROL", k, round(cs.min(), 4), "OK" if cs.min() > 0.98 else "!! MISMATCH", flush=True); X[k] = np.vstack([old, nw[keep]])
mainX = np.r_[main, np.zeros(n_s, bool)]
print("SHUBHA n", n_s, pd.Series(cls_s).value_counts().to_dict(), flush=True)
def fuse(*a): return nz(np.hstack(a))
def cmre(v, ref):
    res = v[mainX] - Ridge(alpha=1.0).fit(ref[mainX], v[mainX]).predict(ref[mainX]); p = PCA(64, random_state=0).fit(res)
    U = p.components_[:int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.5) + 1)].astype(np.float32); return nz(v - (v @ U.T) @ U)
def inlp_x(x, n_max=8):
    xf, s = x[mainX], style[main]; P = np.eye(x.shape[1], dtype=np.float32)
    for _ in range(n_max):
        W = LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T); P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s, cv=5, scoring="balanced_accuracy").mean() <= 1 / G: break
    return nz(x @ P)
def knn(zz, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(zz[mainX]); s = np.empty(len(zz))
    s[mainX] = nn.kneighbors(zz[mainX])[0][:, k]; s[~mainX] = nn.kneighbors(zz[~mainX], k)[0][:, k - 1]; return s
def scores(y, idx):
    yi, li = y[I][idx], lab_s[idx]; km = {c: np.bincount(y[:N][(src == "kaan") & (lab == c)]).argmax() for c in C4}
    bm = {k: pd.Series(lab[cat][y[:N][cat] == k]).mode()[0] for k in np.unique(y[:N][cat])}
    per = {c: (yi[li == c] == km[c]).mean() for c in C4 if (li == c).any()}; m = np.isin(li, C4)
    return dict(xsrc4=float(np.mean(list(per.values()))), xsrc4_lenient=float(np.mean([np.mean([bm.get(v) == c for v in yi[li == c]]) for c in per])),
                purity=float(pd.crosstab(yi[m], li[m]).max(axis=1).sum() / m.sum()), **{f"xsrc_{c}": float(v) for c, v in per.items()})

sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
A4 = pd.read_csv(f"{OUT}/final_v4/assignments.csv", dtype={"id": str}); v4 = dict(zip(A4.id, A4.cluster))
ym4 = np.array([v4[r["id"]] for r, m in zip(items, main) if m])
V = {"usulan v4": lambda: fuse(cmre(sig, dino), cmre(pe, dino), dino), "CMRE lalu INLP": lambda: fuse(inlp_x(cmre(sig, dino)), inlp_x(cmre(pe, dino)), inlp_x(dino)),
     "INLP x3": lambda: fuse(inlp_x(sig), inlp_x(pe), inlp_x(dino)), "fusi mentah": lambda: fuse(sig, pe, dino), "DINOv3-H+ saja": lambda: dino}
for name, f in V.items():
    x = f(); z = reduce(x[mainX], x, 32); zm = z[mainX]
    ym = ym4 if name == "usulan v4" else SpectralClustering(16, affinity="nearest_neighbors", n_neighbors=15, random_state=0, assign_labels="cluster_qr").fit_predict(zm)
    C = np.stack([zm[ym == c].mean(0) for c in range(16)]); D2 = ((z[:, None] - C[None]) ** 2).sum(-1)
    y = D2.argmin(1); y[np.where(mainX)[0]] = ym; r = dict(method=name, **scores(y, np.arange(n_s)))
    for sn, s in (("knn", knn(z)), ("centroid", np.sqrt(D2.min(1)))):
        fl = s > np.percentile(s[mainX], 95); sel = ind | out
        r.update({f"{sn}_auroc": float(roc_auc_score(out[sel], s[I][sel])), f"{sn}_flag_in": float(fl[I][ind].mean()), f"{sn}_flag_out": float(fl[I][out].mean())})
    if name == "usulan v4":
        g = np.random.default_rng(0); B = [scores(y, g.integers(0, n_s, n_s)) for _ in range(1000)]
        for k in ("xsrc4", "xsrc4_lenient", "purity"): r[f"{k}_lo"], r[f"{k}_hi"] = np.percentile([b[k] for b in B], [2.5, 97.5])
        sk = knn(z); A = []
        for _ in range(1000):
            i = g.integers(0, n_s, n_s); o, ii = out[i], (ind | out)[i]
            if o[ii].any() and (~o[ii]).any(): A.append(roc_auc_score(o[ii], sk[I][i][ii]))
        r["knn_auroc_lo"], r["knn_auroc_hi"] = np.percentile(A, [2.5, 97.5])
        r.update({f"ood_{c}": float((sk[I][cls_s == c] > np.percentile(sk[mainX], 95)).mean()) for c in OODC if (cls_s == c).any()})
    rows.append(r); pd.DataFrame(rows).to_csv(f"{OUT}/shubha_results.csv", index=False)
    print("SHUBHA", name, {k: round(float(v), 3) for k, v in r.items() if k != "method"}, flush=True)
print("SHUBHA DONE", flush=True)
