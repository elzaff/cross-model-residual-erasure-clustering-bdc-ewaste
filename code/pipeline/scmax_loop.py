"""SCMax loop (Zhang et al., AAAI-26) with the authors' demo.py defaults, data-free so it can run on Modal.
Moved verbatim from rq_scmax.py; uses the official modules in scmax/."""
import random, numpy as np, torch
from scipy.optimize import linear_sum_assignment
from scmax.auto_encoder import AutoEncoder
from scmax.neighbor_clustering import NeighborClustering
from scmax.feature_optimization import BackFeature

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BS, LR, MSE_EP, BACK_EP, FDIM = 256, 3e-4, 200, 50, 256  # authors' defaults


def nnc(a, b):  # authors' best_mapping: Hungarian-aligned agreement
    D = max(a.max(), b.max()) + 1; C = np.zeros((D, D), np.int64); np.add.at(C, (a, b), 1)
    r, c = linear_sum_assignment(C.max() - C); return C[r, c].sum() / len(a)

def scmax(X, seed):
    # Same seed setup as the authors' demo.py.
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    ae = AutoEncoder(1, [X.shape[1]], FDIM, DEV); opt = torch.optim.Adam(ae.parameters(), lr=LR); mse = torch.nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV); n = len(X)
    def epochs(E, labels=None, K=None):
        bf = BackFeature(BS, K, DEV) if labels is not None else None
        for _ in range(E):
            ae.train(); idx = np.random.permutation(n)
            for s in range(0, n, BS):
                b = idx[s:s + BS]; opt.zero_grad(); xr, z = ae([Xt[b]])
                loss = mse(xr[0], Xt[b]) + (bf.forward_label(z[0], labels[b]) if bf else 0)
                loss.backward(); opt.step()
    def emb_z():
        ae.eval()
        with torch.no_grad(): return ae.forward_all_z([Xt])
    epochs(MSE_EP); Z = emb_z(); cl = NeighborClustering(); labels = np.arange(n); Zp, lp, levels = None, None, []
    while True:
        K, labels = cl.step(Z, labels)
        if Zp is not None:
            Kp, lp = cl.step(Zp, lp)
            if K < 3 or Kp < 3: break
            levels.append(dict(K=K, nnc=nnc(labels, lp), labels=labels.copy()))
        lp = labels; epochs(BACK_EP, labels, K); Zp = emb_z()
    return levels
