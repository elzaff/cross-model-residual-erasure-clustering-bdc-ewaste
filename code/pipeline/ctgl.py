"""Line-by-line NumPy port of the OFFICIAL CTGL MATLAB code (Liu et al., IEEE TNNLS 2025,
https://github.com/CLiu272/CTGL, files CTGL.m, Demo_CTGL_loop.m, function/*.m). Each function below names the .m file
it mirrors. Only numerically-equivalent substitutions are made:
  * wshrinkObj(mode=3, isWeight=0): MATLAB shrinks FFT slices 1..n3/2+1 and fills the rest by conjugate symmetry;
    here every slice is shrunk (the SVD of conj(M) is the conjugate of the SVD of M, so the result is identical);
  * svd(Lstar) in FusionSum: only the first numC left singular vectors are used -> scipy svds(k=numC);
  * MATLAB kmeans(..., 'Replicate', 10) -> sklearn KMeans(n_init=10).
Data layout follows the authors: data[v] is d_v x n (each COLUMN is a sample).
Validation: python code/pipeline/ctgl.py  (runs Demo_CTGL_loop.m on the bundled BUAA.mat)."""
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans

EPS = np.finfo(float).eps


def L2_distance_1(a, b):  # L2_distance_1.m: squared distances between COLUMNS of a and b
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))]); b = np.vstack([b, np.zeros((1, b.shape[1]))])
    aa, bb = (a * a).sum(0), (b * b).sum(0)
    return np.maximum(aa[:, None] + bb[None, :] - 2 * a.T @ b, 0)


def EProjSimplex_new(v, k=1):  # EProjSimplex_new.m: min 1/2||x-v||^2 s.t. x>=0, 1'x=k
    n = len(v); v0 = v - v.mean() + k / n
    if v0.min() >= 0: return v0
    lam, ft = 0.0, 1
    while True:
        v1 = v0 - lam; pos = v1 > 0; f = v1[pos].sum() - k
        if abs(f) <= 1e-10: break
        lam -= f / (-pos.sum()); ft += 1
        if ft > 100: break
    return np.maximum(v0 - lam, 0)


def constructW_PKN(X, k=9):  # constructW_PKN.m (issymmetric = 1)
    n = X.shape[1]; D = L2_distance_1(X, X); idx = np.argsort(D, 1, kind="stable"); W = np.zeros((n, n))
    for i in range(n):
        idd = idx[i, 1:k + 2]; di = D[i, idd]
        W[i, idd] = (di[k] - di) / (k * di[k] - di[:k].sum() + EPS)
    return (W + W.T) / 2


def fun_alm(A, b):  # fun_alm.m
    rho, mu, n = 1.5, 1.0, A.shape[0]; alpha = np.ones(n); v = np.ones(n) / n
    for _ in range(10):
        z = v - A.T @ v / mu + alpha / mu
        mm = (alpha / mu - z) + (A @ z - b) / mu
        v = EProjSimplex_new(-mm)
        alpha = alpha + mu * (v - z); mu = rho * mu
    return v


def wshrinkObj_mode3(X, rho):  # wshrinkObj.m with isWeight=0, mode=3 (t-SVD singular-value thresholding)
    Y = np.moveaxis(X, 0, 2)  # shiftdim(X,1): n x n x (m+1) -> n x (m+1) x n
    S = np.moveaxis(np.fft.fft(Y, axis=2), 2, 0)  # frontal slices, each n x (m+1)
    U, s, Vh = np.linalg.svd(S, full_matrices=False)
    S = (U * np.maximum(s - rho, 0)[:, None, :]) @ Vh
    return np.moveaxis(np.real(np.fft.ifft(np.moveaxis(S, 0, 2), axis=2)), 2, 0)  # shiftdim(Y,2)


def _spectral(S_list, numC, seed):  # body of FusionSum.m (Fusion.m computes the same Lstar)
    Lstar = np.zeros_like(S_list[0])
    for S in S_list:
        d = 1 / np.sqrt(np.abs(S.sum(1))); d[~np.isfinite(d)] = 0
        Lstar += d[:, None] * S * d[None, :]
    Lstar /= len(S_list)
    U = svds(Lstar, k=numC, random_state=seed)[0]
    VV = U / np.maximum(np.sqrt((U * U).sum(1, keepdims=True)), 1e-10)
    return Lstar, VV, KMeans(numC, n_init=10, random_state=seed).fit_predict(VV)


def CTGL(data, numC, k, alpha, lam, pho, maxIter, seed=0):  # CTGL.m (mu = 2, gamma = 1)
    m, n = len(data), data[0].shape[1]; mu, gamma = 2, 1
    G = []
    for X in data:
        X = X / np.sqrt((X ** 2).sum(0, keepdims=True))  # per-sample (column) L2 normalisation
        g = constructW_PKN(X, k); g[np.isnan(g)] = 0; G.append(g)
    W = sum(G) / m  # Fusion.m returns Sstar (mean graph) as the initial W; its labels are unused
    A = [g.copy() for g in G] + [W]
    P = [np.zeros((n, n)) for _ in range(m + 1)]; Q = [np.zeros((n, n)) for _ in range(m + 1)]
    for _ in range(maxIter):
        U = sum(A[:m]); B = P[m] - Q[m] / pho; ed = L2_distance_1(U, U)  # unified graph W
        Wt = np.vstack([EProjSimplex_new(r) for r in (pho * B - ed) / (2 * alpha + pho)])
        W = np.abs((Wt + Wt.T) / 2); A[m] = W
        Pt = wshrinkObj_mode3(np.stack(A, 2), lam / pho); P = [Pt[:, :, i] for i in range(m + 1)]  # tensor
        Q = [Q[i] + pho * (A[i] - P[i]) for i in range(m + 1)]
        pho = pho * mu
        L = np.eye(n) * (1 + pho) + (np.diag(W.sum(0)) - W)
        for v in range(m):  # subgraph propagation; entries outside W's support keep their previous value
            Bv = ((P[v] - Q[v] / pho) + G[v]) / gamma
            for i in range(n):
                idx = np.flatnonzero(W[i] > 0)
                A[v][i, idx] = fun_alm(L[np.ix_(idx, idx)], 2 * Bv[i, idx])
        _, VV, label = _spectral(A, numC, seed)  # FusionSum(A, nCluster, numView+1)
    return label, VV, A


if __name__ == "__main__":  # Demo_CTGL_loop.m on BUAA (k=15, pho=1, maxIter=5, alpha x lambda grid)
    import os, scipy.io as sio
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import normalized_mutual_info_score as nmi
    mat = sio.loadmat(os.path.join(os.path.dirname(__file__), "..", "..", "repos", "CTGL", "Dataset", "BUAA.mat"))
    data = [x.astype(float) for x in mat["data"].ravel()]; y = mat["truelabel"].ravel()[0].ravel().astype(int)
    y = y - y.min(); grid = [0.001, 0.1, 0.5, 1, 3, 5]
    def acc(y, p):
        C = np.zeros((p.max() + 1, y.max() + 1)); np.add.at(C, (p, y), 1); r, c = linear_sum_assignment(-C)
        return C[r, c].sum() / len(y)
    for a in grid:
        for l in grid:
            p = CTGL(data, len(np.unique(y)), 15, a, l, 1, 5)[0]
            print(f"alpha={a} lambda={l} ACC={acc(y, p):.4f} NMI={nmi(y, p):.4f}", flush=True)
