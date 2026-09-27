"""Layer 5: stratified bootstrap CIs and paired deltas for the rq_metrics2 metrics.

Partitions are frozen (fit once, as in the pipeline). Each replicate resamples the labeled
evaluation images with replacement within every (class, source) cell and re-scores all
methods on the same replicate, so deltas are paired. The CI therefore covers evaluation-
sample variability only, not refitting of representation or clustering.
summary = harmonic mean of xtv_all and ami_all (reported as a summary, never used to pick K).
"""
import csv

import numpy as np

from compare_aimv2_fusions import ROOT, cmre, fuse, tta  # sets EXP_OUT
from common import SHARED, load_items
from rq_cca import cca_pair
from rq_metrics2 import SOURCES, evaluate, inlp_x, partition
import rq1_metrics as metrics

OUT = ROOT / "results" / "modal" / "boot2_summary.csv"
N_BOOT = 500
KEYS = ("xsrc", "xtv_bk", "xtv_all", "leak_cmi", "cramer_v", "knn_r1", "ami_all", "summary")
LOWER_IS_BETTER = {"leak_cmi", "cramer_v"}


def scored(y, z, lab, src, idx, rng):
    shared = np.isin(src[idx], ["kaan", "bangla", "karan"]) & np.isin(lab[idx], SHARED)
    out = evaluate(y[idx], z[idx], lab[idx], src[idx], np.ones(len(idx), bool), shared, rng, n_perm=0)
    out["summary"] = 2 * out["xtv_all"] * out["ami_all"] / (out["xtv_all"] + out["ami_all"])
    return out


def main():
    items = load_items()
    main_mask, style, lab, src_raw, cat, shared = metrics.setup(items)
    src = np.where(main_mask, "bdc", src_raw)
    pool = np.where((lab != "") & (lab != "upload") & np.isin(src, SOURCES))[0]
    sig, pe, dino = (tta(n) for n in ("siglip2", "pecoreg", "dinov3hp"))
    feats = {
        "CMRE v4": fuse(cmre(sig, dino, main_mask)[0], cmre(pe, dino, main_mask)[0], dino),
        "INLP x3": fuse(*(inlp_x(x, main_mask, style) for x in (sig, pe, dino))),
        "DINOv3-H+ only": dino,
        "CCA + DINO k=32": fuse(cca_pair(sig, dino, main_mask, 32)[0], cca_pair(pe, dino, main_mask, 32)[0], dino),
        "raw fusion": fuse(sig, pe, dino),
    }
    parts = {name: partition(x, main_mask, 16) for name, x in feats.items()}
    rng = np.random.default_rng(0)
    point = {name: scored(y, z, lab, src, pool, rng) for name, (y, z) in parts.items()}
    cells = {}
    for i in pool:
        cells.setdefault((lab[i], src[i]), []).append(i)
    boot = {name: [] for name in parts}
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(v, len(v)) for v in cells.values()])
        for name, (y, z) in parts.items():
            boot[name].append(scored(y, z, lab, src, idx, rng))
        if b % 50 == 0:
            print("boot", b, flush=True)
    rows = []
    for name in parts:
        for key in KEYS:
            vals = np.array([r[key] for r in boot[name]])
            d = vals - np.array([r[key] for r in boot["CMRE v4"]])
            worse = d > 0 if key in LOWER_IS_BETTER else d < 0  # baseline worse than CMRE
            rows.append({"method": name, "metric": key, "estimate": point[name][key],
                         "lo": np.percentile(vals, 2.5), "hi": np.percentile(vals, 97.5),
                         "delta_vs_cmre": point[name][key] - point["CMRE v4"][key],
                         "d_lo": np.percentile(d, 2.5), "d_hi": np.percentile(d, 97.5),
                         "share_baseline_worse": worse.mean()})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['method']:16s} {r['metric']:9s} {r['estimate']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]"
              f"  d={r['delta_vs_cmre']:+.3f} [{r['d_lo']:+.3f},{r['d_hi']:+.3f}] worse={r['share_baseline_worse']:.2f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
