"""Round 8c: the v4 pipeline refit WITHOUT the 188 BDC test images (they were selected with the team's supervised
SigLIP2 classifier). CMRE, PCA-32 and Spectral K=16 are fitted on the 3,604 train images only; the test images, external
sets, Iliev and Shubha are projected afterwards. Compared with v4: xsrc (BDC/Iliev/Shubha), purity, and agreement of
the partitions on train rows and on the 188 projected test rows. Writes OUT/r8/r8c_notest.csv."""
import numpy as np, pandas as pd
from stack import *
import warnings; warnings.filterwarnings("ignore")

sig, pe, dino = feat("siglip2"), feat("pecoreg"), feat("dinov3hp")
test = np.r_[src == "bdc_test", np.zeros(n_i + n_s, bool)]; trX = mainX & ~test
x4 = fuse(cmre(sig, dino), cmre(pe, dino), dino); y4, _ = project(x4, v4_main())
xt = fuse(cmre(sig, dino, m=trX), cmre(pe, dino, m=trX), dino); yt, _ = project(xt, m=trX)
rows = []
for name, y in (("v4 (train + 188 uji)", y4), ("hanya 3.604 train", yt)):
    rows.append(dict(method=name, **{f"xsrc_{d}": xsrc(y, d) for d in SETS}, purity_bdc=purity_bdc(y)))
rows[1].update(ari_vs_v4_train=ari(y4[trX], yt[trX]), ari_vs_v4_test=ari(y4[test], yt[test]), n_train=int(trX.sum()), n_test=int(test.sum()))
pd.DataFrame(rows).to_csv(f"{OUT}/r8/r8c_notest.csv", index=False); print(pd.DataFrame(rows).round(3).T.to_string(), "\nR8C DONE", flush=True)
