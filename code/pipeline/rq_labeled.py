"""v4 as a classifier on labeled sets: macro precision / recall / F1.
Cluster -> class map = majority BDC filename label per cluster, learned on BDC-train (D_main) ONLY, then frozen and
applied to Kaan, Bangladesh, Karan, Iliev and Shubha (held-out labels). BDC-train scores are in-sample (optimistic).
Only the 10 BDC classes are scored (non-BDC / OOD rows dropped). BDC test labels are never read.
Run: python code/pipeline/rq_labeled.py"""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from rq_divide import ROOT, build, cmre, fuse
from rq_metrics2 import partition

CLS = ["battery", "keyboard", "microwave", "mobile", "mouse", "pcb", "player", "printer", "television", "washing_machine"]
OUT = ROOT / "results" / "classic"


def main():
    S = build(); X, main, lab, src = S["X"], S["main"], S["lab"], S["src"]
    sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
    feats = {"v4 (CMRE -> Spectral)": fuse(cmre(sig, dino, main)[0], cmre(pe, dino, main)[0], dino),
             "fusi mentah -> Spectral": fuse(sig, pe, dino), "DINOv3 saja -> Spectral": dino}
    sets = {"BDC-train (in-sample)": "bdc", "Kaan": "kaan", "Bangladesh": "bangla", "Karan": "karan",
            "Iliev": "iliev", "Shubha": "shubha"}
    rows, per, rng = [], [], np.random.default_rng(0)
    for name, x in feats.items():
        y = partition(x, main, 16)[0]
        fit = main & np.isin(lab, CLS)
        cmap = pd.Series(lab[fit]).groupby(y[fit]).agg(lambda s: s.value_counts().idxmax()).to_dict()
        pred = np.array([cmap.get(c, "none") for c in y])
        for sname, s in sets.items():
            m = (src == s) & np.isin(lab, CLS); t, p = lab[m], pred[m]; labels = sorted(set(t))
            P, R, F, _ = precision_recall_fscore_support(t, p, labels=labels, average="macro", zero_division=0)
            bs = [precision_recall_fscore_support(t[i], p[i], labels=labels, average="macro", zero_division=0)[2]
                  for i in (rng.integers(0, len(t), len(t)) for _ in range(1000))]
            rows.append(dict(method=name, set=sname, n=int(m.sum()), n_classes=len(labels), acc=accuracy_score(t, p),
                             macro_p=P, macro_r=R, macro_f1=F, f1_lo=np.percentile(bs, 2.5), f1_hi=np.percentile(bs, 97.5)))
            rep = classification_report(t, p, labels=labels, output_dict=True, zero_division=0)
            per += [dict(method=name, set=sname, cls=c, **{k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")})
                    for c in labels]
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "labeled_eval.csv", index=False)
    pd.DataFrame(per).round(4).to_csv(OUT / "labeled_eval_perclass.csv", index=False)


if __name__ == "__main__":
    main()
