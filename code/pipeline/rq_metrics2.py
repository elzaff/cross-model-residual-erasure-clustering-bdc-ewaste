"""Non-naive cross-source metrics. Definitions LOCKED before first run (26 Sep 2026).

Labeled pool: BDC main (filename labels), Kaan, Bangladesh, Karan. A (class, source) cell
counts only with >= 20 images. Y = cluster (main = Spectral partition, others = nearest
centroid, as in v4); C = object class; S = source.

- xsrc     : original metric (Bangla in Kaan-majority cluster, 5 shared classes), for continuity.
- xtv_bk   : 1 - TV(P(Y|C,Bangla), P(Y|C,Kaan)), macro over the 5 shared classes.
             Symmetric; subtypes split the same way in both sources are not penalised.
- xtv_all  : same, macro over every (class, source pair) with both cells valid.
- leak_cmi : I(Y;S|C) / H(S|C) over valid cells (0 = cluster carries no source info beyond
             class). leak_p = permutation p-value, S shuffled within class, 200 perms.
- cramer_v : mean over classes of Cramer's V between Y and S.
- knn_r1   : partition-free. Cross-source Recall@1 on the clustering space (PCA-32): query
             source s, gallery = other source t, classes present in both (>= 2 classes),
             macro over classes then over ordered pairs.
- ami_all / ami_ext : AMI(Y, C) on all labeled / external labeled images. Every cross-source
             score must be read with AMI: merging classes inflates agreement, AMI exposes it.
"""
import csv
from itertools import combinations, permutations

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

from compare_aimv2_fusions import ROOT, cmre, fuse, normalize, tta  # sets EXP_OUT
from common import SHARED, load_items, reduce
from rq_cca import cca_pair
import rq1_metrics as metrics

OUT = ROOT / "results" / "modal" / "metrics2_results.csv"
SOURCES = ("bdc", "kaan", "bangla", "karan")
MIN_N = 20


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def table(y, s, m):
    return np.array([[np.sum(m & (y == a) & (s == b)) for b in np.unique(s[m])] for a in np.unique(y[m])])


def cond_mi(y, s, c):
    """I(Y;S|C) and H(S|C) in nats."""
    mi = hs = 0.0
    for k in np.unique(c):
        m = c == k
        tab = table(y, s, m)
        mi += m.mean() * (entropy(tab.sum(1)) + entropy(tab.sum(0)) - entropy(tab.ravel()))
        hs += m.mean() * entropy(tab.sum(0))
    return mi, hs


def tv_agree(y, lab, src, pool, c, a, b):
    ks = np.unique(y)
    pa = np.array([np.sum(pool & (lab == c) & (src == a) & (y == k)) for k in ks], float)
    pb = np.array([np.sum(pool & (lab == c) & (src == b) & (y == k)) for k in ks], float)
    return 1 - 0.5 * np.abs(pa / pa.sum() - pb / pb.sum()).sum()


def evaluate(y, z, lab, src, pool, shared, rng, n_perm=200):
    valid = {(c, s) for c in np.unique(lab[pool]) for s in SOURCES
             if np.sum(pool & (lab == c) & (src == s)) >= MIN_N}
    classes = sorted({c for c, _ in valid})
    multi = [c for c in classes if sum((c, s) in valid for s in SOURCES) >= 2]
    keep = pool & np.array([(l, s) in valid for l, s in zip(lab, src)]) & np.isin(lab, multi)
    out = {"xsrc": metrics.xsrc_agree(y, lab, src, shared)}
    out["xtv_bk"] = float(np.mean([tv_agree(y, lab, src, pool, c, "bangla", "kaan") for c in SHARED]))
    out["xtv_all"] = float(np.mean([tv_agree(y, lab, src, pool, c, a, b) for c in multi
                                    for a, b in combinations(SOURCES, 2) if {(c, a), (c, b)} <= valid]))
    yk, sk, ck = y[keep], src[keep], lab[keep]
    mi, hs = cond_mi(yk, sk, ck)
    null = []
    for _ in range(n_perm):
        s_perm = sk.copy()
        for c in multi:
            s_perm[ck == c] = rng.permutation(s_perm[ck == c])
        null.append(cond_mi(yk, s_perm, ck)[0])
    out["leak_cmi"] = mi / hs
    if n_perm:
        out["leak_p"] = (1 + np.sum(np.array(null) >= mi)) / (n_perm + 1)
    vs = []
    for c in multi:
        tab = table(yk, sk, ck == c)
        vs.append(np.sqrt(chi2_contingency(tab)[0] / (tab.sum() * (min(tab.shape) - 1))) if min(tab.shape) > 1 else 0.0)
    out["cramer_v"] = float(np.mean(vs))
    recalls = []
    for a, b in permutations(SOURCES, 2):
        cls = [c for c in classes if {(c, a), (c, b)} <= valid]
        if len(cls) < 2:
            continue
        q, g = pool & (src == a) & np.isin(lab, cls), pool & (src == b) & np.isin(lab, cls)
        nn = NearestNeighbors(n_neighbors=1).fit(z[g]).kneighbors(z[q], return_distance=False)[:, 0]
        hit = lab[g][nn] == lab[q]
        recalls.append(np.mean([hit[lab[q] == c].mean() for c in cls]))
    out["knn_r1"] = float(np.mean(recalls))
    ext = pool & (src != "bdc")
    out["ami_all"] = adjusted_mutual_info_score(lab[pool], y[pool])
    out["ami_ext"] = adjusted_mutual_info_score(lab[ext], y[ext])
    return out


def partition(feats, main, k):
    z = reduce(feats[main], feats, 32)
    p = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, random_state=0,
                           assign_labels="cluster_qr").fit_predict(z[main])
    centers = np.stack([z[main][p == c].mean(0) for c in np.unique(p)])
    y = ((z[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
    y[main] = p
    return y, z


def inlp_x(x, main, style, n_max=8):
    xf, s, g = x[main], style[main], len(np.unique(style[main]))
    P = np.eye(x.shape[1], dtype=np.float32)
    for _ in range(n_max):
        W = LogisticRegression(max_iter=300).fit(xf @ P, s).coef_.astype(np.float32)
        Q, _ = np.linalg.qr(W.T)
        P = P @ (np.eye(P.shape[0], dtype=np.float32) - Q @ Q.T)
        if cross_val_score(LogisticRegression(max_iter=300, class_weight="balanced"), xf @ P, s,
                           cv=5, scoring="balanced_accuracy").mean() <= 1 / g:
            break
    return normalize(x @ P)


def main():
    items = load_items()
    main_mask, style, lab, src_raw, cat, shared = metrics.setup(items)
    src = np.where(main_mask, "bdc", src_raw)
    pool = (lab != "") & (lab != "upload") & np.isin(src, SOURCES)
    sig, pe, dino = (tta(n) for n in ("siglip2", "pecoreg", "dinov3hp"))
    v4 = fuse(cmre(sig, dino, main_mask)[0], cmre(pe, dino, main_mask)[0], dino)
    raw = fuse(sig, pe, dino)
    variants = [("CMRE v4", v4, k) for k in (3, 8, 13, 16, 20)] + [
        ("raw fusion", raw, 16), ("raw fusion", raw, 3),
        ("DINOv3-H+ only", dino, 16), ("SigLIP2 only", sig, 16), ("PE-Core only", pe, 16),
        ("CCA + DINO k=32", fuse(cca_pair(sig, dino, main_mask, 32)[0], cca_pair(pe, dino, main_mask, 32)[0], dino), 16),
        ("INLP x3", fuse(*(inlp_x(x, main_mask, style) for x in (sig, pe, dino))), 16),
    ]
    rows = []
    for name, feats, k in variants:
        y, z = partition(feats, main_mask, k)
        row = {"method": name, "K": k, **evaluate(y, z, lab, src, pool, shared, np.random.default_rng(0))}
        print({key: (round(v, 3) if isinstance(v, float) else v) for key, v in row.items()}, flush=True)
        rows.append(row)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
