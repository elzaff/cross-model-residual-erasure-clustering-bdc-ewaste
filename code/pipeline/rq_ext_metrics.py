"""Non-naive metrics (rq_metrics2 definitions) on Iliev and Shubha. Locked before first run.

All transforms (CMRE, INLP, CCA, PCA-32, Spectral K=16) are fit on BDC main only; external
images are projected to the nearest main centroid. Class mapping = posthoc_label locked on
24 Sep; cells with < 20 images are dropped (Iliev printer n=15). Only pairs involving the
external set E are scored, against reference sources R = bdc, kaan, bangla, karan:
- xsrc     : share of E images in the Kaan-majority cluster of their class (old metric).
- xtv      : 1 - TV(P(Y|c,E), P(Y|c,t)), macro over valid (class, t).
- leak_cmi : per t, I(Y;S|C)/H(S|C) on E∪t over shared classes; mean over t. cramer_v likewise.
- knn_r1   : Recall@1 E->t and t->E (>= 2 shared classes), macro over classes then directions.
- ami      : AMI(Y, class) on labeled E images.
Bootstrap: 300 replicates resampling every (class, source) cell; paired deltas vs CMRE v4.
"""
import csv

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from compare_aimv2_fusions import ROOT, cmre, emb, fuse, normalize  # sets EXP_OUT
from common import load_items
from rq_cca import cca_pair
from rq_metrics2 import MIN_N, cond_mi, inlp_x, partition, table, tv_agree
import rq1_metrics as metrics

OUT = ROOT / "results" / "modal" / "ext_metrics_summary.csv"
EXT = ROOT / "data" / "ext_eval"
REFS = ("bdc", "kaan", "bangla", "karan")
N_BOOT = 300
KEYS = ("xsrc", "xtv", "leak_cmi", "cramer_v", "knn_r1", "ami")
LOWER_IS_BETTER = {"leak_cmi", "cramer_v"}


def ext_scores(y, z, lab, src, idx, e):
    y, z, lab, src = y[idx], z[idx], lab[idx], src[idx]
    valid = {(c, s) for c in np.unique(lab) for s in (e,) + REFS if np.sum((lab == c) & (src == s)) >= MIN_N}
    ecls = sorted(c for c, s in valid if s == e)
    out = {}
    kaan_major = {c: np.bincount(y[(src == "kaan") & (lab == c)]).argmax() for c in ecls if (c, "kaan") in valid}
    out["xsrc"] = float(np.mean([(y[(src == e) & (lab == c)] == k).mean() for c, k in kaan_major.items()]))
    out["xtv"] = float(np.mean([tv_agree(y, lab, src, np.ones(len(y), bool), c, e, t)
                                for c in ecls for t in REFS if (c, t) in valid]))
    leaks, vs, recalls = [], [], []
    for t in REFS:
        cls = [c for c in ecls if (c, t) in valid]
        if not cls:
            continue
        m = np.isin(src, [e, t]) & np.isin(lab, cls)
        mi, hs = cond_mi(y[m], src[m], lab[m])
        leaks.append(mi / hs)
        for c in cls:
            tab = table(y[m], src[m], lab[m] == c)
            tab = tab[tab.sum(1) > 0]
            vs.append(np.sqrt(chi2_contingency(tab)[0] / (tab.sum() * (min(tab.shape) - 1))) if min(tab.shape) > 1 else 0.0)
        if len(cls) < 2:
            continue
        for a, b in ((e, t), (t, e)):
            q, g = (src == a) & np.isin(lab, cls), (src == b) & np.isin(lab, cls)
            nn = NearestNeighbors(n_neighbors=1).fit(z[g]).kneighbors(z[q], return_distance=False)[:, 0]
            hit = lab[g][nn] == lab[q]
            recalls.append(np.mean([hit[lab[q] == c].mean() for c in cls]))
    out["leak_cmi"], out["cramer_v"], out["knn_r1"] = float(np.mean(leaks)), float(np.mean(vs)), float(np.mean(recalls))
    me = (src == e) & np.isin(lab, ecls)
    out["ami"] = adjusted_mutual_info_score(lab[me], y[me])
    return out


def load_ext(e):
    ni = pd.read_csv(EXT / e / "items.csv", keep_default_na=False)
    ctrl = (ni.set == "control").values
    keep = ~ctrl & (ni.posthoc_label != "EXCLUDE").values
    X = {}
    for k in ("siglip2", "pecoreg", "dinov3hp"):
        old = normalize(emb(k) + emb(k + "_flip"))
        new = normalize(np.load(EXT / e / f"emb_{k}.npy") + np.load(EXT / e / f"emb_{k}_flip.npy")).astype(np.float32)
        cos = (old[ni.cls.values[ctrl].astype(int)] * new[ctrl]).sum(1)
        print(e, "control", k, "cos min", round(float(cos.min()), 4), flush=True)
        assert cos.min() > 0.98, "embedding mismatch"
        X[k] = np.vstack([old, new[keep]])
    return X, ni.posthoc_label.values[keep]


def main():
    items = load_items()
    main0, style0, lab0, src0, cat, shared = metrics.setup(items)
    rows = []
    for e in ("iliev", "shubha"):
        X, lab_e = load_ext(e)
        n_e = len(lab_e)
        main = np.r_[main0, np.zeros(n_e, bool)]
        style = np.r_[style0, np.full(n_e, style0[0])]
        lab = np.r_[lab0, lab_e]
        src = np.r_[np.where(main0, "bdc", src0), np.full(n_e, e)]
        sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
        feats = {
            "CMRE v4": fuse(cmre(sig, dino, main)[0], cmre(pe, dino, main)[0], dino),
            "INLP x3": fuse(*(inlp_x(x, main, style) for x in (sig, pe, dino))),
            "DINOv3-H+ only": dino,
            "CCA + DINO k=32": fuse(cca_pair(sig, dino, main, 32)[0], cca_pair(pe, dino, main, 32)[0], dino),
            "raw fusion": fuse(sig, pe, dino),
        }
        parts = {name: partition(x, main, 16) for name, x in feats.items()}
        pool = np.where((lab != "") & (lab != "upload") & np.isin(src, (e,) + REFS))[0]
        point = {name: ext_scores(y, z, lab, src, pool, e) for name, (y, z) in parts.items()}
        for name, p in point.items():
            print(e, name, {k: round(v, 3) for k, v in p.items()}, flush=True)
        cells = {}
        for i in pool:
            cells.setdefault((lab[i], src[i]), []).append(i)
        rng = np.random.default_rng(0)
        boot = {name: [] for name in parts}
        for _ in range(N_BOOT):
            idx = np.concatenate([rng.choice(v, len(v)) for v in cells.values()])
            for name, (y, z) in parts.items():
                boot[name].append(ext_scores(y, z, lab, src, idx, e))
        for name in parts:
            for key in KEYS:
                vals = np.array([r[key] for r in boot[name]])
                d = vals - np.array([r[key] for r in boot["CMRE v4"]])
                worse = d > 0 if key in LOWER_IS_BETTER else d < 0
                rows.append({"set": e, "method": name, "metric": key, "estimate": point[name][key],
                             "lo": np.percentile(vals, 2.5), "hi": np.percentile(vals, 97.5),
                             "delta_vs_cmre": point[name][key] - point["CMRE v4"][key],
                             "d_lo": np.percentile(d, 2.5), "d_hi": np.percentile(d, 97.5),
                             "share_baseline_worse": worse.mean()})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
