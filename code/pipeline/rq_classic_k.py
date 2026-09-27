"""Label-free choice of K with classic criteria, per debiasing variant of rq_classic.py (concat fusion, PCA-32, D_main).
Criteria: silhouette of Spectral-kNN (Rousseeuw 1987), eigengap of the normalized kNN Laplacian (von Luxburg 2007),
consensus clustering PAC (Monti et al. 2003; Senbabaoglu et al. 2014), prediction strength (Tibshirani & Walther 2005).
Labels are read only to report xsrc at each K. Run: EXP_OUT=data/embeddings_final_v4 python code/pipeline/rq_classic_k.py"""
import os, numpy as np, pandas as pd
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from rq_classic import *

KS, H = range(6, 25), 50

def pac(zm, k, frac=0.8, lo=0.1, hi=0.9):
    n, rng = len(zm), np.random.default_rng(0); co = np.zeros((n, n), np.float32); cnt = np.zeros((n, n), np.float32)
    for h in range(H):
        i = rng.choice(n, int(frac * n), replace=False)
        O = np.eye(k, dtype=np.float32)[KMeans(k, n_init=1, random_state=h).fit_predict(zm[i])]
        co[np.ix_(i, i)] += O @ O.T; cnt[np.ix_(i, i)] += 1
    iu = np.triu_indices(n, 1); c = co[iu] / np.maximum(cnt[iu], 1)
    return float(((c > lo) & (c < hi)).mean())

def pred_strength(zm, k, reps=5):
    ps = []
    for r in range(reps):
        i = np.random.default_rng(r).permutation(len(zm)); a, b = zm[i[::2]], zm[i[1::2]]
        ka, kb = KMeans(k, n_init=5, random_state=r).fit(a), KMeans(k, n_init=5, random_state=r).fit(b)
        ya, yb = ka.predict(b), kb.labels_  # test half labelled by training centroids vs its own clustering
        ps.append(min(((ya[yb == c][:, None] == ya[yb == c][None]).sum() - (yb == c).sum()) /
                      max((yb == c).sum() * ((yb == c).sum() - 1), 1) for c in range(k)))
    return float(np.mean(ps))

if __name__ == "__main__":
    rows, out = [], os.path.join(ROOT, "results", "classic")
    for a, fa in A.items():
        x = nz(np.hstack(fa())); z = reduce(x[main], x, 32); zm = z[main]; W = knn_sym(zm)
        ev = np.sort(eigsh(laplacian(W, normed=True), k=max(KS) + 2, which="SM")[0])
        for k in KS:
            y = SpectralClustering(k, affinity="precomputed", random_state=0, assign_labels="cluster_qr").fit_predict(W)
            e = evaluate(y, z)
            rows.append(dict(debias=a, K=k, sil=e["sil"], eigengap=float(ev[k] - ev[k - 1]), pac=pac(zm, k),
                             ps=pred_strength(zm, k), xsrc=e["xsrc"], ext_nmi=e["ext_nmi"], purity=e["purity"]))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "grid_k.csv"), index=False)
    D, pick = pd.DataFrame(rows), []
    for a, d in D.groupby("debias", sort=False):
        ok = d[d.ps >= 0.8]
        for crit, kk in [("silhouette", d.loc[d.sil.idxmax(), "K"]), ("eigengap", d.loc[d.eigengap.idxmax(), "K"]),
                         ("PAC", d.loc[d.pac.idxmin(), "K"]), ("PS>=0.8", ok.K.max() if len(ok) else np.nan)]:
            r = d[d.K == kk]
            pick.append(dict(debias=a, criterion=crit, K=kk, xsrc=r.xsrc.iloc[0] if len(r) else np.nan,
                             ext_nmi=r.ext_nmi.iloc[0] if len(r) else np.nan))
    P = pd.DataFrame(pick).round(4); P.to_csv(os.path.join(out, "grid_k_pick.csv"), index=False); print(P.to_string())
