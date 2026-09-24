"""Metric helpers shared by rq1b and later scripts (mirrors rq1.py definitions)."""
import numpy as np
from common import *

def setup(items):
    main = np.array([r["set"] == "main" for r in items])
    src = np.array([r["source"] for r in items]); lab = np.array([r["posthoc_label"] for r in items])
    style = style_groups(items)[0]
    return main, style, lab, src, main & (lab != ""), (~main) & np.isin(lab, SHARED) & np.isin(src, ["kaan", "bangla", "karan"])

def xsrc_agree(y, lab, src, shared):
    a = []
    for c in SHARED:
        kc, bc = shared & (src == "kaan") & (lab == c), shared & (src == "bangla") & (lab == c)
        if kc.sum() and bc.sum(): a.append((y[bc] == np.bincount(y[kc]).argmax()).mean())
    return float(np.mean(a))

def score_labels(y, main, style, lab, src, cat, shared):
    return dict(style_nmi=nmi(style[main], y[main]), obj_nmi_bdc=nmi(lab[cat], y[cat]),
                ext_obj_nmi=nmi(lab[shared], y[shared]), ext_src_given_obj=cond_nmi(y[shared], src[shared], lab[shared]),
                xsrc_agree=xsrc_agree(y, lab, src, shared))

def score(x, xf, k, dim, main, style, lab, src, cat, shared):
    z = reduce(x[main], x, dim)
    fits = [kmeans(z[main], k, s) for s in (0, 1, 2)]
    y = fits[0].predict(z)
    return dict(seed_ari=float(np.mean([ari(fits[0].labels_, f.labels_) for f in fits[1:]])),
                **score_labels(y, main, style, lab, src, cat, shared))
