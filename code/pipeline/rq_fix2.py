"""Round 7 fixes. Iliev was evaluated once under the locked protocol (rq_iliev.py); from here on it is a declared
DEVELOPMENT set, so anything chosen here needs a fresh test set before it can be claimed as held-out.
  OOD    centroid distance (v4) vs per-cluster normalised distance vs kNN distance (Sun et al. 2022) in several spaces:
         Iliev AUROC (BDC classes + Laptop vs the rest), false-alarm rate on in-distribution Kaan / Bangladesh / Karan,
         GIZ rate, and how many out-of-BDC Iliev crops the C0 "attractor" absorbs unflagged
  JAC    per-cluster stability: mean best Jaccard over 20 bootstrap (80%) Spectral runs (Hennig 2007)
  MAP    Iliev mapping sensitivity (post hoc): without Music-Player (portable players, not BDC turntables/CD), and TV
         counted correct in the CRT/flat subtype cluster
  COMBO  CMRE followed by INLP, on BDC (5-class xsrc, 3 seeds) and Iliev (10-class xsrc)
Writes OUT/fix2_results.csv and OUT/fix2_jaccard.csv."""
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

CLS10 = ["battery", "keyboard", "microwave", "mobile", "mouse", "pcb", "player", "printer", "television", "washing_machine"]
items = load_items(); main, style, lab, src, cat, shared = M.setup(items); G = len(np.unique(style[main])); N = len(items)
NI = pd.read_csv(f"{OUT}/iliev/items.csv", keep_default_na=False); il = (NI.set != "control").values
cls_i, lab_i = NI.cls.values[il], NI.posthoc_label.values[il]; n_i = int(il.sum()); I = np.arange(N, N + n_i)
ind = np.isin(cls_i, list(set(cls_i[lab_i != ""])) + ["Laptop"])
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def new(k): return nz(np.load(f"{OUT}/iliev/emb_{k}.npy").astype(np.float32))
X = {k: np.vstack([nz(emb(k) + emb(k + "_flip")), nz(new(k) + new(k + "_flip"))[il]]) for k in ("siglip2", "pecoreg", "dinov3hp")}
mainX = np.r_[main, np.zeros(n_i, bool)]
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
def spectral(zm, K, s=0): return SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15, random_state=s, assign_labels="cluster_qr").fit_predict(zm)
def project(x, ym=None):
    z = reduce(x[mainX], x, 32); zm = z[mainX]; ym = spectral(zm, 16) if ym is None else ym
    C = np.stack([zm[ym == c].mean(0) for c in range(16)]); D2 = ((z[:, None] - C[None]) ** 2).sum(-1)
    y = D2.argmin(1); y[np.where(mainX)[0]] = ym; return z, y, np.sqrt(D2.min(1))
def xsrc_iliev(y, drop=(), tv_sub=False):
    km = {c: np.bincount(y[:N][(src == "kaan") & (lab == c)]).argmax() for c in CLS10}; yi = y[I]; out = []
    for c in CLS10:
        m = (lab_i == c) & ~np.isin(cls_i, drop)
        if not m.any(): continue
        ok = yi[m] == km[c]
        if tv_sub and c == "television": ok |= ((cls_i[m] == "CRT-TV") & (yi[m] == 3)) | ((cls_i[m] == "Flat-Panel-TV") & (yi[m] == 7))
        out.append(ok.mean())
    return float(np.mean(out))

sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
best, raw = fuse(cmre(sig, dino), cmre(pe, dino), dino), fuse(sig, pe, dino)
A4 = pd.read_csv(f"{OUT}/final_v4/assignments.csv", dtype={"id": str}); v4 = dict(zip(A4.id, A4.cluster))
ym4 = np.array([v4[r["id"]] for r, m in zip(items, main) if m])
z, y, d = project(best, ym4); zr, _, _ = project(raw); rows = []
ext_in = {"kaan": src == "kaan", "bangla": (src == "bangla") & (lab != "light_bulb"), "karan": src == "karan", "giz": src == "giz"}
def ood_row(name, s):
    thr = np.percentile(s[mainX], 95); fl = s > thr; out = ~ind
    r = dict(method=f"OOD {name}", auroc=float(roc_auc_score(out, s[I])), iliev_in_flag=float(fl[I][ind].mean()), iliev_out_flag=float(fl[I][out].mean()),
             c0_absorbed=float(((y[I] == 0) & ~fl[I] & out).sum() / out.sum()), **{f"flag_{k}": float(fl[:N][m].mean()) for k, m in ext_in.items()})
    rows.append(r); print("FIX2", r, flush=True)
def knn(zz, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(zz[mainX]); s = np.empty(len(zz))
    s[mainX] = nn.kneighbors(zz[mainX])[0][:, k]; s[~mainX] = nn.kneighbors(zz[~mainX], k)[0][:, k - 1]; return s
ood_row("centroid (v4)", d)
q = np.array([np.percentile(d[mainX][ym4 == c], 95) for c in range(16)]); ood_row("per-cluster normalised", d / q[y])
for k in (1, 5, 10, 20): ood_row(f"kNN k={k} usulan PCA-32", knn(z, k))
ood_row("kNN k=10 usulan penuh", knn(best, 10)); ood_row("kNN k=10 fusi mentah PCA-32", knn(zr, 10))

# ---- MAP: Iliev mapping sensitivity (post hoc)
rows.append(dict(method="MAP locked protocol", xsrc10=xsrc_iliev(y))); rows.append(dict(method="MAP tanpa Music-Player", xsrc10=xsrc_iliev(y, ("Music-Player",))))
rows.append(dict(method="MAP tanpa Music-Player + subtipe TV", xsrc10=xsrc_iliev(y, ("Music-Player",), True))); print("FIX2", rows[-3:], flush=True)

# ---- COMBO: CMRE then INLP
for name, f in {"usulan (CMRE)": lambda: best, "CMRE lalu INLP": lambda: fuse(inlp_x(cmre(sig, dino)), inlp_x(cmre(pe, dino)), inlp_x(dino)),
                "INLP x3": lambda: fuse(inlp_x(sig), inlp_x(pe), inlp_x(dino))}.items():
    x = f(); zz = reduce(x[mainX], x, 32); bd = []
    for s in range(3):
        yy = spectral(zz[mainX], 16, s); C = np.stack([zz[mainX][yy == c].mean(0) for c in range(16)])
        ya = ((zz[:, None] - C[None]) ** 2).sum(-1).argmin(1); ya[np.where(mainX)[0]] = yy; bd.append((M.xsrc_agree(ya[:N], lab, src, shared), xsrc_iliev(ya)))
    r = dict(method=f"COMBO {name}", xsrc_bdc=float(np.mean([b[0] for b in bd])), xsrc10=float(np.mean([b[1] for b in bd]))); rows.append(r); print("FIX2", r, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/fix2_results.csv", index=False)

# ---- JAC: per-cluster bootstrap stability of the v4 partition
zm = z[mainX]; g = np.random.default_rng(0); J = np.zeros((20, 16))
for b in range(20):
    i = np.sort(g.choice(len(zm), int(0.8 * len(zm)), replace=False)); yb, y0 = spectral(zm[i], 16, b), ym4[i]
    for c in range(16):
        A = y0 == c; J[b, c] = max(((A & (yb == k)).sum() / (A | (yb == k)).sum()) for k in range(16)) if A.any() else np.nan
jac = pd.DataFrame(dict(cluster=range(16), n=np.bincount(ym4, minlength=16), jaccard_mean=J.mean(0), jaccard_min=J.min(0)))
jac.to_csv(f"{OUT}/fix2_jaccard.csv", index=False); print(jac.round(3).to_string(), flush=True); print("FIX2 DONE", flush=True)
