"""External object NMI for Tabel 3 rows that were n.d. (EPO-flip, CMRE cepat, CTGL), same protocol as the table:
NMI between cluster and post-hoc label on the 1,869 labelled Kaan/Bangladesh/Karan images of the five shared classes,
external images assigned to the nearest D_main centroid in the PCA-32 space. v4 is recomputed as a control (0.859).
Writes results/classic/ext_nmi_fill.csv."""
import numpy as np, pandas as pd
from rq_divide import ROOT, build, fuse
from rq_classic_routes_3model import epo_flip
from compare_aimv2_fusions import cmre
from rq_metrics2 import partition
from common import SHARED, nmi

S = build()
X, main = S["X"], S["main"]
lab, src = np.asarray(S["lab"]), np.asarray(S["src"])
shared = (~main) & np.isin(lab, SHARED) & np.isin(src, ["kaan", "bangla", "karan"])
print("shared external images:", int(shared.sum()), {s: int((shared & (src == s)).sum()) for s in ("kaan", "bangla", "karan")})
sig, pe, dino = (X[n] for n in ("siglip2", "pecoreg", "dinov3hp"))

routes = {"CMRE usulan (kontrol)": fuse(cmre(sig, dino, main)[0], cmre(pe, dino, main)[0], dino),
          "EPO-flip x2 + DINO": fuse(epo_flip(sig, "siglip2", main), epo_flip(pe, "pecoreg", main), dino),
          "CMRE cepat": fuse(cmre(sig, dino, main)[0], dino)}
rows, reduced_v4 = [], None
for name, feats in routes.items():
    y, z = partition(feats, main, 16)
    if reduced_v4 is None: reduced_v4 = z
    rows.append(dict(method=name, ext_obj_nmi=nmi(lab[shared], y[shared])))

# CTGL: stored D_main labels on the CMRE route; external images go to the nearest CTGL centroid in the same PCA-32 space
yc = np.load(ROOT / "results/classic/ctgl_labels_CMRE.npy")
assert len(yc) == main.sum()
zm = reduced_v4[main]; ids = np.unique(yc); cent = np.vstack([zm[yc == c].mean(0) for c in ids])
y = np.empty(len(main), int); y[main] = yc
y[~main] = ids[np.argmin(((reduced_v4[~main][:, None, :] - cent[None]) ** 2).sum(-1), 1)]
rows.append(dict(method="CTGL", ext_obj_nmi=nmi(lab[shared], y[shared])))

out = pd.DataFrame(rows); out.to_csv(ROOT / "results/classic/ext_nmi_fill.csv", index=False)
print(out.round(4).to_string(index=False))
