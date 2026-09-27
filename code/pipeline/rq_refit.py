"""Test B: re-fit the whole v4 pipeline FROM SCRATCH on non-BDC labeled e-waste (Kaan + Bangladesh + Karan + Iliev,
BDC classes only). CMRE, PCA-32, SCMax K and Spectral are all fitted on this pool; BDC is not used. Labels are read only
for scoring: K=10 -> Hungarian 1:1 macro P/R/F1; K=SCMax -> majority (many-to-one) mapping; plus NMI, ARI and
xsrc = share of each non-Kaan source's class-c images in the Kaan-majority cluster of c (macro over class, source).
Run: python code/pipeline/rq_refit.py"""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (accuracy_score, adjusted_rand_score, normalized_mutual_info_score,
                             precision_recall_fscore_support)
from rq_divide import ROOT, build, cmre, fuse
from rq_labeled import CLS
from rq_metrics2 import partition
from rq_scmax import scmax

SOURCES = ("kaan", "bangla", "karan", "iliev")


def mapped(t, y, hungarian):
    if hungarian:
        C = pd.crosstab(y, t); r, c = linear_sum_assignment(-C.values)
        m = dict(zip(C.index[r], C.columns[c]))
    else:
        m = pd.Series(t).groupby(y).agg(lambda s: s.value_counts().idxmax()).to_dict()
    return np.array([m.get(v, "none") for v in y])


def pool_feats():
    S = build(); X, lab, src = S["X"], S["lab"], S["src"]
    P = np.isin(src, SOURCES) & np.isin(lab, CLS)
    sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
    feats = {"v4 (CMRE -> Spectral)": fuse(cmre(sig, dino, P)[0], cmre(pe, dino, P)[0], dino),
             "fusi mentah -> Spectral": fuse(sig, pe, dino), "DINOv3 saja -> Spectral": dino}
    return P, lab[P], src[P], feats


def main(picks_by=None):  # picks_by: {method: [K per SCMax seed]} from modal_refit.py; None = run SCMax here
    P, t, s, feats = pool_feats()
    print("pool", P.sum(), pd.Series(s).value_counts().to_dict(), flush=True)
    rows = []
    for name, x in feats.items():
        picks = picks_by[name] if picks_by else [max(scmax(x[P], sd), key=lambda L: L["nnc"])["K"] for sd in (3407, 0, 1)]
        for rule, K in (("K=10 (jumlah kelas)", 10), (f"K=SCMax {picks}", int(np.median(picks)))):
            y = partition(x, P, K)[0][P]; p = mapped(t, y, K == 10)
            Pm, Rm, Fm, _ = precision_recall_fscore_support(t, p, labels=CLS, average="macro", zero_division=0)
            km = {c: np.bincount(y[(s == "kaan") & (t == c)]).argmax() for c in CLS if ((s == "kaan") & (t == c)).any()}
            xs = [np.mean(y[(s == o) & (t == c)] == k) for c, k in km.items() for o in SOURCES[1:] if ((s == o) & (t == c)).sum() >= 20]
            rows.append(dict(method=name, K_rule=rule, K=K, macro_p=Pm, macro_r=Rm, macro_f1=Fm, acc=accuracy_score(t, p),
                             nmi=normalized_mutual_info_score(t, y), ari=adjusted_rand_score(t, y), xsrc=float(np.mean(xs))))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "classic" / "refit_nonbdc.csv", index=False)


if __name__ == "__main__":
    main()
