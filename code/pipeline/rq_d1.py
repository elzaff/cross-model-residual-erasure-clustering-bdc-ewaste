"""Round 8 D1: recent frozen-feature clustering methods on the final v4 features, K=16, 3 seeds, trained on D_main only
and applied to every row (external, Iliev, Shubha) so all methods are scored on the same xsrc protocol.
TURTLE and TEMI are the ports from rq_extra.py (same hyper-parameters); DECMCV = K-Means + Ward + BIRCH voting (rq_max.py).
Writes OUT/r8/d1_baselines.csv."""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score
from stack import *
import warnings; warnings.filterwarnings("ignore")

DEV = "cuda" if torch.cuda.is_available() else "cpu"; K = 16; R8 = f"{OUT}/r8"; os.makedirs(R8, exist_ok=True)
sig, pe, dino = feat("siglip2"), feat("pecoreg"), feat("dinov3hp"); cs, cp = cmre(sig, dino), cmre(pe, dino)
x4 = fuse(cs, cp, dino); z4 = reduce(x4[mainX], x4, 32); zm = z4[mainX]; ym4 = v4_main()
def nearest(ym):  # D_main partition -> every row by nearest centroid in z4
    ks = np.unique(ym); C = np.stack([zm[ym == c].mean(0) for c in ks]); y = ks[((z4[:, None] - C[None]) ** 2).sum(-1).argmin(1)]; y[np.where(mainX)[0]] = ym; return y

def turtle(Zs, seed, T=6000, M_in=10, gamma=10., lr=1e-3):  # port from rq_extra.py
    torch.manual_seed(seed); np.random.seed(seed)
    Ztr = [torch.tensor(z[mainX], device=DEV) for z in Zs]; dims = [z.shape[1] for z in Zs]
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

class Head(nn.Module):
    def __init__(self, d):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 256))
        self.last = nn.utils.weight_norm(nn.Linear(256, K, bias=False)); self.last.weight_g.data.fill_(1); self.last.weight_g.requires_grad = False
    def forward(self, x): return self.last(F.normalize(self.mlp(x), dim=-1))

def temi(X, seed, heads=50, epochs=100, bs=256, knn=50, beta=0.6, temp=0.1):  # port from rq_extra.py
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    Xa = torch.tensor(X, device=DEV); Xm = Xa[torch.tensor(np.where(mainX)[0], device=DEV)]
    nn_idx = (Xm @ Xm.T).topk(knn + 1, dim=1).indices[:, 1:]
    S = nn.ModuleList([Head(X.shape[1]) for _ in range(heads)]).to(DEV); Tm = nn.ModuleList([Head(X.shape[1]) for _ in range(heads)]).to(DEV)
    Tm.load_state_dict(S.state_dict()); Tm.requires_grad_(False); pk = torch.full((heads, K), 1 / K, device=DEV)
    opt = torch.optim.AdamW(S.parameters(), lr=5e-4, weight_decay=1e-4)
    steps = epochs * ((len(Xm) + bs - 1) // bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-6); step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(Xm), generator=g).to(DEV); last = torch.zeros(heads, device=DEV)
        for b in perm.split(bs):
            nb = nn_idx[b, torch.randint(0, knn, (len(b),), generator=g).to(DEV)]; x1, x2 = Xm[b], Xm[nb]
            with torch.no_grad():
                t1 = torch.stack([F.softmax(h(x1) / temp, -1) for h in Tm]); t2 = torch.stack([F.softmax(h(x2) / temp, -1) for h in Tm])
                pk = 0.9 * pk + 0.1 * torch.cat([t1, t2], 1).mean(1); w = (t1 * t2).sum(-1).mean(0)
            s1 = torch.stack([F.softmax(h(x1) / temp, -1) for h in S]); s2 = torch.stack([F.softmax(h(x2) / temp, -1) for h in S])
            bmi = lambda ps, pt: -((((ps * pt) ** beta) / pk[:, None]).sum(-1)).log()
            per_head = 0.5 * ((w * bmi(s1, t2)).mean(1) + (w * bmi(s2, t1)).mean(1))
            opt.zero_grad(); per_head.sum().backward(); opt.step(); sched.step(); step += 1; last += per_head.detach()
            mom = 1 - (1 - 0.996) * (np.cos(np.pi * step / steps) + 1) / 2
            with torch.no_grad():
                for ps_, pt_ in zip(S.parameters(), Tm.parameters()): pt_.mul_(mom).add_((1 - mom) * ps_.detach())
    best = int(last.argmin())
    with torch.no_grad(): return Tm[best](Xa).argmax(1).cpu().numpy()

def decmcv(seed):
    base = kmeans(zm, K, seed).labels_
    def al(o):
        C = np.zeros((K, K)); np.add.at(C, (base, o), 1); r, c = linear_sum_assignment(-C); mp = dict(zip(c, r)); return np.array([mp[v] for v in o])
    keep = (al(AgglomerativeClustering(K).fit_predict(zm)) == base) & (al(Birch(n_clusters=K, threshold=0.3).fit_predict(zm)) == base)
    return nearest(base), float(keep.mean())

CFG = {"Spectral-kNN (usulan v4)": lambda s: (nearest(ym4 if s == 0 else spectral(zm, K, s)), 1.0),
       "TURTLE (3 ruang CMRE v4)": lambda s: (turtle([cs, cp, dino], s), 1.0),
       "TEMI (fitur fusi v4)": lambda s: (temi(x4, s), 1.0),
       "DECMCV voting (z v4)": decmcv}
PART = f"{R8}/d1_baselines.csv"; rows = pd.read_csv(PART).to_dict("records") if os.path.exists(PART) else []
for name, fn in CFG.items():
    if name in {r["method"] for r in rows}: continue
    res = [fn(s) for s in range(3)]; ys = [r[0] for r in res]
    for s, (y, cov) in enumerate(res):
        ymm = y[mainX]
        rows.append(dict(method=name, seed=s, n_found=len(np.unique(ymm)), seed_ari=np.mean([ari(ymm, o[mainX]) for o in ys if o is not y]),
                         sil=silhouette_score(zm, ymm) if len(np.unique(ymm)) > 1 else np.nan, coverage=cov,
                         **{f"xsrc_{d}": xsrc(y, d) for d in SETS}, purity_bdc=purity_bdc(y)))
    pd.DataFrame(rows).to_csv(PART, index=False); print("D1", name, flush=True)
print(pd.DataFrame(rows).groupby("method", sort=False).mean(numeric_only=True).round(3).to_string(), "\nD1 DONE", flush=True)
