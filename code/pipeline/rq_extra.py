"""(4a) Automatic, label-free style discovery + (3) SOTA frozen-feature clustering baselines. K=16 throughout.
  extra_style.csv      auto-style proxy (AdaIN stats -> PCA -> K-Means, k by silhouette) vs heuristic proxy:
                       raw vs INLP(heuristic) vs INLP(auto) per backbone, and the final fusion rebuilt with the auto proxy
  extra_baselines.csv  TURTLE (port of the official mlbio-epfl/turtle loop) and TEMI (port of HHU-MMBS/TEMI losses)
"""
import copy, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from common import *
import rq1_metrics as M
import warnings; warnings.filterwarnings("ignore")

DEV = "cuda" if torch.cuda.is_available() else "cpu"; K = 16
items = load_items(); main, style, lab, src, cat, shared = M.setup(items)
white = np.array([r["border_white"] >= 0.5 for r in items])
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
    out = [(y[cat & (lab == c) & ~white] == np.bincount(y[cat & (lab == c) & white]).argmax()).mean()
           for c in np.unique(lab[cat]) if (cat & (lab == c) & white).sum() >= 15 and (cat & (lab == c) & ~white).sum() >= 15]
    return float(np.mean(out))
def purity(y, m): return float(pd.crosstab(y[m], lab[m]).max(axis=1).sum() / m.sum())
def full_metrics(y):
    return dict(**M.score_labels(y, main, style, lab, src, cat, shared), xstyle_bdc=xstyle_bdc(y),
                purity_bdc=purity(y, cat), purity_ext=purity(y, (~main) & (lab != "")))
def spectral_all(x, seed=0):
    z = reduce(x[main], x, 32); zm = z[main]
    ym = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15, random_state=seed, assign_labels="cluster_qr").fit_predict(zm)
    C = np.stack([zm[ym == c].mean(0) for c in range(K)]); y = ((z[:, None] - C[None]) ** 2).sum(-1).argmin(1); y[np.where(main)[0]] = ym
    return y, silhouette_score(zm, ym)

# ---------------- (4a) automatic style discovery ----------------
st = np.load(f"{OUT}/emb_stylestats.npy"); st = (st - st[main].mean(0)) / (st[main].std(0) + 1e-6)
zs = reduce(st[main], st, 32)
cand = {k: kmeans(zs[main], k, 0) for k in range(3, 9)}
sil = {k: silhouette_score(zs[main], f.labels_) for k, f in cand.items()}
ks = max(sil, key=sil.get); auto = cand[ks].predict(zs)
print(f"AUTO STYLE: k={ks} (silhouette {sil[ks]:.3f}) | NMI(auto, heuristic)={nmi(style[main], auto[main]):.3f} | "
      f"NMI(auto, source | external)={nmi(src[~main], auto[~main]):.3f} | NMI(auto, object | BDC catalog)={nmi(lab[cat], auto[cat]):.3f}", flush=True)
reps = {"SigLIP2-So400m": "siglip2", "SigLIP2-giant": "siglip2g", "PE-Core-G": "pecoreg", "CLIP-L/14": "clip", "DINOv3-H+": "dinov3hp", "DINOv3-7B": "dinov3_7b"}
rows, deb_auto, deb_heur = [], {}, {}
DONE_STYLE = os.path.exists(f"{OUT}/extra_style.csv")  # checkpoint: style section already computed
for name, key in (reps.items() if not DONE_STYLE else [("SigLIP2-So400m", "siglip2"), ("DINOv3-H+", "dinov3hp")]):
    if DONE_STYLE: deb_heur[name] = to_chance(tta(key), style)[0]; continue
    x = tta(key); deb_heur[name], nh = to_chance(x, style); deb_auto[name], na = to_chance(x, auto)
    for variant, xv, n in [("raw", x, 0), ("INLP heuristik", deb_heur[name], nh), ("INLP gaya-otomatis", deb_auto[name], na)]:
        y, s = spectral_all(xv)
        rows.append(dict(rep=name, variant=variant, inlp_iter=n, probe_auto=probe(xv, auto), probe_heur=probe(xv, style),
                         style_nmi_auto=nmi(auto[main], y[main]), sil=s, **full_metrics(y)))
    print("style", name, flush=True)
for tag, parts in [] if DONE_STYLE else [("Fusi (INLP gaya-otomatis keduanya)", [deb_auto["SigLIP2-So400m"], deb_auto["DINOv3-H+"]]),
                   ("Fusi (SigLIP2 INLP gaya-otomatis + DINOv3-H+ mentah)", [deb_auto["SigLIP2-So400m"], tta("dinov3hp")]),
                   ("Fusi (INLP heuristik keduanya) = final", [deb_heur["SigLIP2-So400m"], deb_heur["DINOv3-H+"]])]:
    y, s = spectral_all(nz(np.hstack(parts)))
    rows.append(dict(rep=tag, variant="fusion", style_nmi_auto=nmi(auto[main], y[main]), sil=s, **full_metrics(y)))
S = pd.read_csv(f"{OUT}/extra_style.csv") if DONE_STYLE else pd.DataFrame(rows)
if not DONE_STYLE: S.to_csv(f"{OUT}/extra_style.csv", index=False)
print(S[["rep", "variant", "inlp_iter", "probe_auto", "style_nmi", "style_nmi_auto", "xsrc_agree", "ext_src_given_obj", "ext_obj_nmi", "purity_bdc", "purity_ext", "xstyle_bdc"]].round(3).to_string(index=False), flush=True)

# ---------------- (3) SOTA baselines on frozen features ----------------
def turtle(Zs, seed, T=6000, M_in=10, gamma=10., lr=1e-3):
    """Port of the official TURTLE loop (mlbio-epfl/turtle, run_turtle.py): weight-normed linear task encoder per
    space, labels = mean of per-space softmax, cold-start Adam inner loop on detached labels (first-order
    hypergradient), outer CE + gamma * per-space marginal entropy. Trained on D_main, predicts every row."""
    torch.manual_seed(seed); np.random.seed(seed)
    Ztr = [torch.tensor(z[main], device=DEV) for z in Zs]; dims = [z.shape[1] for z in Zs]
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
        (err - gamma * sum(torch.special.entr(p.mean(0)).sum() for p in per)).backward(); opt.step()
    with torch.no_grad(): return encode([torch.tensor(z, device=DEV) for z in Zs])[0].argmax(1).cpu().numpy()

class Head(nn.Module):  # DINO-style head as in TEMI (nlayers=2, hidden 512, bottleneck 256)
    def __init__(self, d):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 256))
        self.last = nn.utils.weight_norm(nn.Linear(256, K, bias=False)); self.last.weight_g.data.fill_(1); self.last.weight_g.requires_grad = False
    def forward(self, x): return self.last(F.normalize(self.mlp(x), dim=-1))

def temi(X, seed, heads=50, epochs=100, bs=256, knn=50, beta=0.6, temp=0.1):
    """Port of TEMI (Adaloglou et al., BMVC 2023; HHU-MMBS/TEMI-official-BMVC2023): kNN positive pairs on frozen
    features, student/teacher heads (EMA 0.996 -> 1), loss = sim_weight(pt1, pt2) * -log sum((ps*pt)^beta / pk),
    pk = EMA(0.9) of teacher probs; the head with the lowest final-epoch loss is used."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    Xa = torch.tensor(X, device=DEV); Xm = Xa[torch.tensor(np.where(main)[0], device=DEV)]
    nn_idx = (Xm @ Xm.T).topk(knn + 1, dim=1).indices[:, 1:]
    S = nn.ModuleList([Head(X.shape[1]) for _ in range(heads)]).to(DEV)
    Tm = nn.ModuleList([Head(X.shape[1]) for _ in range(heads)]).to(DEV)  # deepcopy fails on weight_norm modules
    Tm.load_state_dict(S.state_dict()); Tm.requires_grad_(False)
    pk = torch.full((heads, K), 1 / K, device=DEV)
    opt = torch.optim.AdamW(S.parameters(), lr=5e-4, weight_decay=1e-4)
    steps = epochs * ((len(Xm) + bs - 1) // bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-6); step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(Xm), generator=g).to(DEV); last = torch.zeros(heads, device=DEV)
        for b in perm.split(bs):
            nb = nn_idx[b, torch.randint(0, knn, (len(b),), generator=g).to(DEV)]; x1, x2 = Xm[b], Xm[nb]
            with torch.no_grad():
                t1 = torch.stack([F.softmax(h(x1) / temp, -1) for h in Tm]); t2 = torch.stack([F.softmax(h(x2) / temp, -1) for h in Tm])
                pk = 0.9 * pk + 0.1 * torch.cat([t1, t2], 1).mean(1)
                w = (t1 * t2).sum(-1).mean(0)  # sim_weight averaged over heads
            s1 = torch.stack([F.softmax(h(x1) / temp, -1) for h in S]); s2 = torch.stack([F.softmax(h(x2) / temp, -1) for h in S])
            bmi = lambda ps, pt: -((((ps * pt) ** beta) / pk[:, None]).sum(-1)).log()
            per_head = 0.5 * ((w * bmi(s1, t2)).mean(1) + (w * bmi(s2, t1)).mean(1))
            opt.zero_grad(); per_head.sum().backward(); opt.step(); sched.step(); step += 1; last += per_head.detach()
            mom = 1 - (1 - 0.996) * (np.cos(np.pi * step / steps) + 1) / 2
            with torch.no_grad():
                for ps_, pt_ in zip(S.parameters(), Tm.parameters()): pt_.mul_(mom).add_((1 - mom) * ps_.detach())
    best = int(last.argmin())
    with torch.no_grad(): return Tm[best](Xa).argmax(1).cpu().numpy()

sig, dino = tta("siglip2"), tta("dinov3hp")
fin = np.load(f"{OUT}/x_final.npy")
cfgs = {"TURTLE (SigLIP2 + DINOv3-H+ mentah)": lambda s: turtle([sig, dino], s),
        "TURTLE (SigLIP2 + DINOv3-H+ ter-INLP)": lambda s: turtle([deb_heur["SigLIP2-So400m"], deb_heur["DINOv3-H+"]], s),
        "TEMI (DINOv3-H+ mentah)": lambda s: temi(dino, s),
        "TEMI (representasi final ter-INLP)": lambda s: temi(fin, s),
        "Spectral-kNN (usulan)": lambda s: spectral_all(fin, s)[0]}
BPART = f"{OUT}/extra_baselines_partial.csv"
brow = pd.read_csv(BPART).to_dict("records") if os.path.exists(BPART) else []
for name, fn in cfgs.items():
    if name in {r["method"] for r in brow}: continue
    ys = [fn(s) for s in range(3)]
    for s, y in enumerate(ys):
        brow.append(dict(method=name, seed=s, n_found=len(np.unique(y[main])), seed_ari=np.mean([ari(y[main], o[main]) for o in ys if o is not y]), **full_metrics(y)))
    print("baseline", name, flush=True); pd.DataFrame(brow).to_csv(BPART, index=False)
Bt = pd.DataFrame(brow); Bt.to_csv(f"{OUT}/extra_baselines.csv", index=False)
print(Bt.groupby("method", sort=False)[["n_found", "seed_ari", "purity_bdc", "purity_ext", "xsrc_agree", "ext_src_given_obj", "ext_obj_nmi", "style_nmi"]].mean().round(3).to_string())
print("EXTRA DONE")
