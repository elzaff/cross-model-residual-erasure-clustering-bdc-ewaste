"""Maximal-analysis pass (Kaggle). Writes CSVs to OUT:
  max_reps.csv       every backbone +/- INLP-to-chance, K=14, 5 seeds, all metrics incl. BDC cross-style (10 classes)
  max_fusion.csv     debiased VLM (+) SSL pairs; label-free pick = best silhouette among stability >= 0.95
  max_methods.csv    clustering methods on the picked representation (KMeans/Ward/GMM/Spectral/TURTLE)
  max_proxy.csv      robustness of conclusions to alternative pixel-style proxies
  final_config.json  picked representation, method and K (consumed by final.py)
"""
import json, os, numpy as np, pandas as pd, torch
from itertools import product
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from common import *
import rq1_metrics as M

items = load_items(); main, style, lab, src, cat, shared = M.setup(items)
white = np.array([r["border_white"] >= 0.5 for r in items])
rng = np.random.default_rng(0)
def nz(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def tta(a): return nz(emb(a) + emb(a + "_flip")) if os.path.exists(f"{OUT}/emb_{a}_flip.npy") else emb(a)
def probe(x, s): return cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), x[main], s[main], cv=5, scoring="balanced_accuracy").mean()
def to_chance(x, s):
    ch = 1 / len(np.unique(s[main]))
    for n in range(1, 9):
        xi = inlp(x[main], s[main], x, n)
        if probe(xi, s) <= ch: return xi, n
    return xi, 8

def xstyle_bdc(y):
    """Independent of external sets: within each BDC filename class, share of natural-background photos landing in
    the majority cluster of that class's white-background photos (classes with >=15 per group)."""
    out = []
    for c in np.unique(lab[cat]):
        w, n = cat & (lab == c) & white, cat & (lab == c) & ~white
        if w.sum() >= 15 and n.sum() >= 15: out.append((y[n] == np.bincount(y[w]).argmax()).mean())
    return float(np.mean(out)), len(out)

def purity(y, m):
    ct = pd.crosstab(y[m], lab[m]); return float(ct.max(1).sum() / m.sum())

def full_metrics(y):
    xs, nc = xstyle_bdc(y)
    return dict(**M.score_labels(y, main, style, lab, src, cat, shared), xstyle_bdc=xs, xstyle_classes=nc,
                purity_bdc=purity(y, cat), purity_ext=purity(y, (~main) & (lab != "")))

def evaluate(name, x, k=14, dim=32, seeds=5, extra=None):
    z = reduce(x[main], x, dim); fits = [kmeans(z[main], k, s) for s in range(seeds)]
    return [dict(rep=name, seed=s, seed_ari=np.mean([ari(f.labels_, g.labels_) for g in fits if g is not f]),
                 sil=silhouette_score(z[main], f.labels_), **full_metrics(f.predict(z)), **(extra or {})) for s, f in enumerate(fits)]

# ---------- 1. representation bank ----------
hand = np.load(f"{OUT}/emb_hand.npy").astype(np.float32)
bank = {"handcrafted": (hand - hand[main].mean(0)) / (hand[main].std(0) + 1e-6)}
names = {"dinov2l_cls": "DINOv2-L", "dinov3_cls": "DINOv3-B", "dinov3l_cls": "DINOv3-L", "dinov3_obj": "DINOv3-B objek",
         "dinov3l_obj": "DINOv3-L objek", "dinov3l_ctx": "DINOv3-L konteks", "clip": "CLIP-L/14", "siglip2": "SigLIP2-So400m",
         "siglip2g": "SigLIP2-giant", "aimv2": "AIMv2-L",
         # latest large backbones: paradigm (SSL / VLM / supervised) x architecture (CNN / ViT)
         "dinov3cnxl": "DINOv3 ConvNeXt-L", "dinov3hp": "DINOv3-H+", "dinov3_7b": "DINOv3-7B", "pecoreg": "PE-Core-G",
         "clipcnxxxl": "CLIP ConvNeXt-XXL", "cnxv2h": "ConvNeXt V2-H (IN-22k)"}
for k, v in names.items():
    if os.path.exists(f"{OUT}/emb_{k}.npy"): bank[v] = tta(k)
import warnings; warnings.filterwarnings("ignore")
deb, iters = {}, {}
for v in bank:
    if v == "handcrafted" or "konteks" in v: continue
    deb[v], iters[v] = to_chance(bank[v], style)
if os.path.exists(f"{OUT}/max_reps.csv"):  # checkpoint (preemption-safe restart)
    R = pd.read_csv(f"{OUT}/max_reps.csv")
else:
    rows = []
    for v, x in bank.items(): rows += evaluate(v, x, extra=dict(debiased=False, inlp_iter=0, probe=probe(x, style)))
    for v, x in deb.items(): rows += evaluate(v + " + INLP", x, extra=dict(debiased=True, inlp_iter=iters[v], probe=probe(x, style)))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/max_reps.csv", index=False)
print(R.groupby("rep", sort=False).mean(numeric_only=True).round(3).to_string(), flush=True)

# ---------- 2. fusion search (label-free pick) ----------
def boot_stab(z, k, n=4):
    zm = z[main]; base = kmeans(zm, k, 0).labels_; out = []
    for s in range(n):
        idx = np.sort(rng.choice(len(zm), int(.8 * len(zm)), replace=False))
        out.append(ari(base[idx], kmeans(zm[idx], k, s + 1).labels_))
    return float(np.mean(out))
vlm = [v for v in ("PE-Core-G", "SigLIP2-giant", "SigLIP2-So400m", "CLIP ConvNeXt-XXL", "CLIP-L/14", "AIMv2-L") if v in deb]
ssl = [v for v in ("DINOv3-7B", "DINOv3-H+", "DINOv3-L", "DINOv3 ConvNeXt-L", "DINOv3-B", "DINOv2-L") if v in deb]
cands = {f"{a} ⊕ {b}": nz(np.hstack([deb[a], deb[b]])) for a, b in product(vlm, ssl)}
cands.update({v: deb[v] for v in vlm + ssl})
FZ_PART = f"{OUT}/max_fusion_partial.csv"
frows = pd.read_csv(FZ_PART).to_dict("records") if os.path.exists(FZ_PART) else []
done_reps = {r["rep"] for r in frows}
for name, x in cands.items():
    if name in done_reps: continue
    z = reduce(x[main], x, 32)
    for k in (10, 12, 14, 16, 20):
        y = kmeans(z[main], k, 0).predict(z)
        frows.append(dict(rep=name, k=k, stab=boot_stab(z, k), sil=silhouette_score(z[main], y[main]),
                          db=davies_bouldin_score(z[main], y[main]), **full_metrics(y)))
    print("fusion", name, flush=True); pd.DataFrame(frows).to_csv(FZ_PART, index=False)
Fz = pd.DataFrame(frows); Fz.to_csv(f"{OUT}/max_fusion.csv", index=False)
ok = Fz[Fz.stab >= 0.95]; pick = (ok if len(ok) else Fz).sort_values(["sil", "stab"], ascending=False).iloc[0]
print("LABEL-FREE PICK:", pick.rep, "k", pick.k, flush=True)
x_final = cands[pick.rep]; np.save(f"{OUT}/x_final.npy", x_final)
z = reduce(x_final[main], x_final, 32); np.save(f"{OUT}/z_final.npy", z); K = int(pick.k)

# ---------- 3. clustering methods on the picked representation ----------
def turtle(spaces, k, outer=600, inner=10, gamma=10.0, seed=0):
    """TURTLE (Gadetsky et al., ICML 2024), simplified: find the labeling that is most linearly separable in every
    representation space. Task heads give soft labels tau; an unrolled inner loop fits a linear probe per space;
    the outer loss is the probes' cross-entropy on tau minus gamma * entropy of the label marginal. Label-free."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(seed)
    Z = [torch.tensor(s, dtype=torch.float32, device=dev) for s in spaces]
    heads = [torch.nn.Linear(s.shape[1], k).to(dev) for s in Z]
    opt = torch.optim.Adam([p for h in heads for p in h.parameters()], lr=1e-3)
    for _ in range(outer):
        tau = torch.softmax(sum(h(s) for h, s in zip(heads, Z)), 1); loss = 0
        for s in Z:
            w = torch.zeros(s.shape[1], k, device=dev, requires_grad=True)
            for _ in range(inner):
                g, = torch.autograd.grad(-(tau * torch.log_softmax(s @ w, 1)).sum(1).mean(), w, create_graph=True); w = w - 0.5 * g
            loss = loss - (tau * torch.log_softmax(s @ w, 1)).sum(1).mean()
        pm = tau.mean(0); loss = loss + gamma * (pm * torch.log(pm + 1e-8)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return torch.softmax(sum(h(s) for h, s in zip(heads, Z)), 1).argmax(1).cpu().numpy()
def assign_all(ym):  # nearest centroid (in z) of a main-only partition, for every row incl. external
    C = np.stack([z[main][ym == c].mean(0) for c in np.unique(ym)]); return np.argmin(((z[:, None] - C[None]) ** 2).sum(-1), 1)
parts = pick.rep.split(" ⊕ ")
spaces = [reduce(deb[p][main], deb[p], 32)[main] for p in parts]
meths = {"K-Means": lambda s: kmeans(z[main], K, s).labels_,
         "Agglomerative-Ward": lambda s: AgglomerativeClustering(K).fit_predict(z[main]),
         "GMM-diag": lambda s: GaussianMixture(K, covariance_type="diag", random_state=s).fit_predict(z[main]),
         "Spectral-kNN": lambda s: SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15, random_state=s, assign_labels="cluster_qr").fit_predict(z[main]),
         "TURTLE": lambda s: turtle(spaces, K, seed=s)}
mrows, mlabels = [], {}
for mname, fn in meths.items():
    ys = [fn(s) for s in range(3)]; mlabels[mname] = ys[0]
    for s, ym in enumerate(ys):
        mrows.append(dict(method=mname, seed=s, n_clusters=len(np.unique(ym)), seed_ari=np.mean([ari(ym, o) for o in ys if o is not ym]),
                          sil=silhouette_score(z[main], ym) if len(np.unique(ym)) > 1 else np.nan, **full_metrics(assign_all(ym))))
    print("method", mname, flush=True)
# DECMCV protocol (Huang et al., 2025): K-Means + Agglomerative + BIRCH voting, disagreeing images are discarded.
from sklearn.cluster import Birch
from scipy.optimize import linear_sum_assignment
def decmcv(k, s=0):
    base = kmeans(z[main], k, s).labels_
    def al(o):
        C = np.zeros((k, k)); np.add.at(C, (base, o), 1); r, c = linear_sum_assignment(-C); m = dict(zip(c, r)); return np.array([m[v] for v in o])
    votes = [al(AgglomerativeClustering(k).fit_predict(z[main])), al(Birch(n_clusters=k, threshold=0.3).fit_predict(z[main]))]
    return base, (votes[0] == base) & (votes[1] == base)
for kk in (K, 50):
    ym, keep = decmcv(kk)
    y = assign_all(ym); kept_all = np.zeros(len(z), bool); kept_all[np.where(main)[0][keep]] = True
    mrows.append(dict(method=f"DECMCV-voting K={kk} (buang tidak-sepakat)", seed=0, n_clusters=kk, coverage=float(keep.mean()),
                      purity_bdc_kept=purity(y, cat & kept_all), purity_bdc=purity(y, cat), purity_ext=purity(y, (~main) & (lab != "")),
                      xsrc_agree=M.xsrc_agree(y, lab, src, shared), sil=silhouette_score(z[main], ym)))
Mt = pd.DataFrame(mrows); Mt.to_csv(f"{OUT}/max_methods.csv", index=False)
agg = Mt[~Mt.method.str.startswith("DECMCV")].groupby("method").mean(numeric_only=True)
best_m = agg[agg.seed_ari >= 0.9].sort_values("sil", ascending=False).index[0] if (agg.seed_ari >= 0.9).any() else "K-Means"
np.save(f"{OUT}/y_method_pick.npy", assign_all(mlabels[best_m]))
print(agg.round(3).to_string(), "\nLABEL-FREE METHOD PICK:", best_m, flush=True)

# ---------- 4. style-proxy robustness ----------
side = np.array([max(r["width"], r["height"]) for r in items])
st = np.c_[[r["border_white"] for r in items], [r["border_std"] for r in items], [r["colorfulness"] for r in items], np.log(side)]
st = (st - st[main].mean(0)) / st[main].std(0)
proxies = {"default (putih>=0.5 x res 200/800)": style, "putih>=0.4": style_groups(items, 0.4)[0], "putih>=0.6": style_groups(items, 0.6)[0],
           "res 256/1024": style_groups(items, 0.5, (256, 1024))[0], "k-means-6 statistik piksel": kmeans(st[main], 6, 0).predict(st)}
prow = []
for pname, sp in proxies.items():
    for rep in dict.fromkeys(parts + ["CLIP-L/14"]):
        if rep not in bank: continue
        for deb_flag in (False, True):
            x = to_chance(bank[rep], sp)[0] if deb_flag else bank[rep]
            zz = reduce(x[main], x, 32); y = kmeans(zz[main], K, 0).predict(zz)
            prow.append(dict(proxy=pname, rep=rep, inlp=deb_flag, style_nmi_proxy=nmi(sp[main], y[main]), **full_metrics(y)))
P = pd.DataFrame(prow); P.to_csv(f"{OUT}/max_proxy.csv", index=False); print(P.round(3).to_string(), flush=True)
json.dump(dict(rep=pick.rep, parts=parts, K=K, method=best_m, inlp_iters={p: iters.get(p) for p in parts}),
          open(f"{OUT}/final_config.json", "w"), indent=1)
print("ALL DONE")
