"""External object NMI (Tabel 3 protocol) for DIVIDE on the three view pairs behind the "0,65-0,77" range:
raw [SigLIP2+PE | DINOv3], CMRE [CMRE(SigLIP2)+CMRE(PE) | DINOv3] and CMRE-VLM [CMRE(SigLIP2) | CMRE(PE)].
Same training, K=16 and K-Means readout as rq_divide.py; 3 seeds. Writes results/classic/divide_ext_nmi.csv."""
import numpy as np, pandas as pd
from rq_divide import ROOT, build, divide_embed, kmeans_all, fuse, cmre
from rq1_metrics import xsrc_agree
from common import nmi

S = build(); X, main = S["X"], S["main"]
lab, src, shared = S["lab"], S["src"], S["shared"]
sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
cm = [cmre(sig, dino, main)[0], cmre(pe, dino, main)[0]]
VIEWS = {"mentah": [fuse(sig, pe), dino], "CMRE": [fuse(*cm), dino], "CMRE-VLM": cm}
rows = []
for tag, views in VIEWS.items():
    for seed in (0, 1, 2):
        y = kmeans_all(divide_embed(views, main, seed), main, 16)
        rows.append(dict(views=tag, seed=seed, ext_obj_nmi=nmi(lab[shared], y[shared]),
                         xsrc_BDC=xsrc_agree(y, lab, src, shared)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results/classic/divide_ext_nmi.csv", index=False)
print(pd.DataFrame(rows).groupby("views", sort=False)[["ext_obj_nmi", "xsrc_BDC"]].mean().round(4).to_string())
