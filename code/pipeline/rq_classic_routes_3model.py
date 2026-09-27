"""Compare flip-derived EPO with final CMRE on the same three-model protocol.

Fit all projections and clustering on D_main; keep DINOv3 unchanged in the
primary comparison. The x3 route is an exploratory control that also erases
flip directions from DINOv3. Labels are used only by the scoring functions.
Run: python code/pipeline/rq_classic_routes_3model.py
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from rq_divide import ROOT, build, fuse, score_all  # configures final-v4 embeddings
from compare_aimv2_fusions import cmre, normalize
from common import SHARED, emb
from rq_metrics2 import partition


ILIEV_CLASSES = ("battery", "keyboard", "microwave", "mobile", "mouse", "pcb",
                 "player", "printer", "television", "washing_machine")


def epo_flip(v, name, main, r=64):
    original, flipped = emb(name), emb(f"{name}_flip")
    train_rows = np.flatnonzero(main)
    assert len(v) >= len(original) == len(flipped) and train_rows.max() < len(original)
    delta = original[train_rows] - flipped[train_rows]
    basis = PCA(r, random_state=0).fit(delta).components_.astype(np.float32)
    return normalize(v - (v @ basis.T) @ basis)


def locked_iliev_xsrc(S, clusters):
    """Original ten-class Iliev metric used in the manuscript (no min-cell cutoff)."""
    label, source = S["lab"], S["src"]
    shares = []
    for category in ILIEV_CLASSES:
        kaan = clusters[(label == category) & (source == "kaan")]
        iliev = clusters[(label == category) & (source == "iliev")]
        assert len(kaan) and len(iliev), f"missing Iliev pair: {category}"
        shares.append(np.mean(iliev == np.bincount(kaan).argmax()))
    return float(np.mean(shares))


def paired_xsrc_intervals(S, assignments, repeats=2000):
    """Resample target photos within class; hold Kaan references and clusters fixed."""
    labels, sources = S["lab"], S["src"]
    shubha_classes = tuple(c for c in sorted(set(labels[sources == "shubha"]))
                           if np.any((labels == c) & (sources == "kaan")))
    specs = (("BDC", "bangla", SHARED), ("iliev", "iliev", ILIEV_CLASSES),
             ("shubha", "shubha", shubha_classes))
    rng, rows = np.random.default_rng(0), []
    cmre_y, epo_y = assignments["v4 CMRE x2 + DINO"], assignments["EPO-flip x2 + DINO"]
    for dataset, target, classes in specs:
        cells = [np.flatnonzero((labels == category) & (sources == target)) for category in classes]
        assert all(len(cell) for cell in cells)
        refs = [[np.bincount(y[(labels == category) & (sources == "kaan")]).argmax()
                 for category in classes] for y in (cmre_y, epo_y)]
        observed = [np.mean([(y[cell] == ref).mean() for cell, ref in zip(cells, ref_ids)])
                    for y, ref_ids in zip((cmre_y, epo_y), refs)]
        deltas = []
        for _ in range(repeats):
            samples = [rng.choice(cell, size=len(cell), replace=True) for cell in cells]
            estimates = [np.mean([(y[sample] == ref).mean() for sample, ref in zip(samples, ref_ids)])
                         for y, ref_ids in zip((cmre_y, epo_y), refs)]
            deltas.append(estimates[1] - estimates[0])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append(dict(set=dataset, classes=len(classes), n_target=sum(map(len, cells)),
                         cmre=observed[0], epo_flip=observed[1], delta_epo_minus_cmre=observed[1] - observed[0],
                         delta_lo=lo, delta_hi=hi))
    return rows


def main():
    S = build()
    X, main_mask = S["X"], S["main"]
    sig, pe, dino = (X[name] for name in ("siglip2", "pecoreg", "dinov3hp"))
    sig_epo = epo_flip(sig, "siglip2", main_mask)
    pe_epo = epo_flip(pe, "pecoreg", main_mask)
    routes = (
        ("v4 CMRE x2 + DINO", fuse(cmre(sig, dino, main_mask)[0], cmre(pe, dino, main_mask)[0], dino)),
        ("EPO-flip x2 + DINO", fuse(sig_epo, pe_epo, dino)),
        ("EPO-flip x3", fuse(sig_epo, pe_epo, epo_flip(dino, "dinov3hp", main_mask))),
        ("mentah", fuse(sig, pe, dino)),
    )
    rows, assignments = [], {}
    for method, features in routes:
        clusters, reduced = partition(features, main_mask, 16)
        assignments[method] = clusters
        for dataset, metrics in score_all(S, clusters, reduced).items():
            row = dict(m=method, set=dataset, **metrics,
                       xsrc10_locked=locked_iliev_xsrc(S, clusters) if dataset == "iliev" else None)
            rows.append(row)
            print(row, flush=True)
    output = ROOT / "results" / "classic" / "classic_routes_3model.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}", flush=True)
    interval_output = output.with_name("classic_routes_3model_paired_ci.csv")
    intervals = paired_xsrc_intervals(S, assignments)
    pd.DataFrame(intervals).to_csv(interval_output, index=False)
    print(f"wrote {interval_output}: {intervals}", flush=True)


if __name__ == "__main__":
    main()
