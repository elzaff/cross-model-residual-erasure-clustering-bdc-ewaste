"""BDC12-matched evaluation on the FULL Custom Bangladeshi E-Waste dataset (prep_bangfull.py + embed_new.py):
frozen v4 (CMRE/PCA-32/Spectral K=16 fitted on D_main; cluster -> class = BDC-train majority). Semantic transfer on the
710 in-taxonomy images (and our 449 deduplicated subset): Top-1 micro / macro (balanced), Top-3 and MRR over classes
ranked by the nearest centroid of each class's clusters. Out-of-taxonomy: 1,443 non-e-waste images (positive) vs 710,
AUROC/AUPRC of kNN distance (k=10, PCA-32, gallery = D_main) and nearest-centroid distance.
Run (after `modal volume get bdc-max /out/bangfull data/ext_eval/`): python code/pipeline/rq_bangfull.py"""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from rq_divide import ROOT, cmre, fuse  # sets EXP_OUT first
import rq1_metrics as metrics
from common import load_items
from rq_ext_metrics import load_ext
from rq_labeled import CLS
from rq_metrics2 import partition


def main():
    items = load_items(); main0, _, lab0, _, _, _ = metrics.setup(items); N = len(items)
    X, lab_e = load_ext("bangfull"); M = pd.read_csv(ROOT / "data" / "ext_eval" / "bangfull_meta.csv")
    assert len(M) == len(lab_e), "meta / embedding row mismatch"
    main = np.r_[main0, np.zeros(len(M), bool)]; lab = np.r_[lab0, lab_e]
    ind = M.cls.isin(["Battery_Waste", "Keyboard", "Mobile", "Mouse", "PCB"]).values; ood, ours = ~ind, M.in_ours.values
    sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
    feats = {"v4 (CMRE)": fuse(cmre(sig, dino, main)[0], cmre(pe, dino, main)[0], dino),
             "fusi mentah": fuse(sig, pe, dino), "DINOv3 saja": dino}
    rows = []
    for name, x in feats.items():
        y, z = partition(x, main, 16); zm = z[main]; fit = main & np.isin(lab, CLS)
        cmap = pd.Series(lab[fit]).groupby(y[fit]).agg(lambda s: s.value_counts().idxmax()).to_dict()
        C = np.stack([zm[y[main] == c].mean(0) for c in range(16)]); ze = z[N:]
        d = ((ze[:, None] - C[None]) ** 2).sum(-1)  # (n_ext, 16)
        cls_d = np.stack([d[:, [c for c in range(16) if cmap.get(c) == k]].min(1) if any(cmap.get(c) == k for c in range(16))
                          else np.full(len(ze), np.inf) for k in CLS], 1)
        rank = np.argsort(cls_d, 1); true = np.array([CLS.index(t) if t in CLS else -1 for t in lab_e])
        pos = np.array([np.where(rank[i] == true[i])[0][0] + 1 if true[i] >= 0 else 0 for i in range(len(ze))])
        for sub, m in (("710 (setara BDC12)", ind), ("449 (dedup kita)", ind & ours)):
            t, p = true[m], rank[m, 0]
            rows.append(dict(method=name, subset=sub, n=int(m.sum()), top1_micro=float((p == t).mean()),
                             top1_macro=float(np.mean([(p[t == c] == c).mean() for c in np.unique(t)])),
                             top3=float((pos[m] <= 3).mean()), mrr=float((1 / pos[m]).mean())))
            print(rows[-1], flush=True)
        knn = NearestNeighbors(n_neighbors=10).fit(zm).kneighbors(ze)[0][:, -1]
        for sname, s in (("kNN k=10", knn), ("jarak centroid", d.min(1))):
            yb, sb = ood[ind | ood].astype(int), s[ind | ood]
            rows.append(dict(method=name, subset=f"OOD 1443 vs 710 [{sname}]", n=int((ind | ood).sum()),
                             auroc=roc_auc_score(yb, sb), auprc=average_precision_score(yb, sb)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "classic" / "bangfull_eval.csv", index=False)


if __name__ == "__main__":
    main()
