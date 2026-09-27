"""Label-free rank rules for the CMRE residual (post hoc check of the frozen r=64).

Rules: Horn parallel analysis (column permutation, 95th percentile), Marchenko-Pastur
bulk edge with noise level from the median eigenvalue, and Gavish-Donoho optimal hard
threshold for unknown noise. Only `main` images are used; no labels enter the rules.
"""
import csv

import numpy as np
from scipy import integrate, optimize
from sklearn.linear_model import Ridge

from compare_aimv2_fusions import ROOT, cmre, fuse, score, tta  # sets EXP_OUT
from common import load_items
import rq1_metrics as metrics

OUT = ROOT / "results" / "modal" / "cmre_rank_rules.csv"
N_PERM = 20


def residual(v, reference, main):
    reg = Ridge(alpha=1.0).fit(reference[main], v[main])
    res = v[main] - reg.predict(reference[main])
    return res - res.mean(0)


def eigvals(x):
    return np.sort(np.linalg.eigvalsh(x.T @ x / len(x)))[::-1]


def parallel_analysis(res, rng):
    null = np.stack([eigvals(np.column_stack([rng.permutation(c) for c in res.T]))
                     for _ in range(N_PERM)])
    above = eigvals(res) > np.percentile(null, 95, axis=0)
    return int(np.argmin(above)) if not above.all() else len(above)


def mp_median(beta):
    lo, hi = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    dens = lambda t: np.sqrt((hi - t) * (t - lo)) / (2 * np.pi * beta * t)
    return optimize.brentq(lambda m: integrate.quad(dens, lo, m)[0] - 0.5, lo, hi)


def marchenko_pastur(res):
    n, p = res.shape
    beta = min(p, n) / max(p, n)
    lam = eigvals(res)
    sigma2 = np.median(lam) / mp_median(beta)
    return int((lam > sigma2 * (1 + np.sqrt(beta)) ** 2).sum())


def gavish_donoho(res):
    n, p = res.shape
    beta = min(p, n) / max(p, n)
    s = np.linalg.svd(res, compute_uv=False)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return int((s > omega * np.median(s)).sum())


def project_out(v, reference, main, r):
    basis = np.linalg.svd(residual(v, reference, main), full_matrices=False)[2][:r].astype(np.float32)
    x = v - (v @ basis.T) @ basis
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    items = load_items()
    main_mask, style, labels, source, cat, shared = metrics.setup(items)
    sig, pe, dino = (tta(n) for n in ("siglip2", "pecoreg", "dinov3hp"))
    rng = np.random.default_rng(0)
    ranks = {}
    for name, v in (("siglip2", sig), ("pecore", pe)):
        res = residual(v, dino, main_mask)
        ranks[name] = {"parallel": parallel_analysis(res, rng),
                       "mp": marchenko_pastur(res), "gd": gavish_donoho(res)}
        print(name, res.shape, ranks[name], flush=True)
    rows = []
    for rule in ("parallel", "mp", "gd"):
        rs, rp = ranks["siglip2"][rule], ranks["pecore"][rule]
        feats = fuse(project_out(sig, dino, main_mask, rs), project_out(pe, dino, main_mask, rp), dino)
        row = score(f"rule={rule} r_sig={rs} r_pe={rp}", feats, main_mask, style, labels, source, cat, shared)
        rows.append({"rule": rule, "r_siglip2": rs, "r_pecore": rp, **row})
    frozen = fuse(cmre(sig, dino, main_mask)[0], cmre(pe, dino, main_mask)[0], dino)
    rows.append({"rule": "frozen r=64", "r_siglip2": 64, "r_pecore": 64,
                 **score("frozen r=64", frozen, main_mask, style, labels, source, cat, shared)})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
