"""Core of the proposed pipeline, as used in the paper (images only; no labels, text, or style proxies).

Input: L2-normalised test-time-augmented embeddings (N x D) of the BDC images from SigLIP2-So400m, PE-Core-G and
DINOv3-H+ (extraction: pipeline/embed_kg.py, pipeline/embed_extra.py, pipeline/embed_new.py).
Steps: CMRE debiasing (Eq. 1-3) -> fusion + PCA-32 (Eq. 4) -> Spectral-kNN with the SCMax-selected K (16)
-> kNN out-of-distribution score (Eq. 5). The full experiments live in pipeline/ (see README.md).
"""
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def cmre(v, ref, r=64, alpha=1.0):
    """Cross-Model Residual Erasure: remove from the VLM embeddings v the top-r principal directions of the part
    that the self-supervised reference cannot predict. Returns the cleaned embeddings and the projector P,
    which is applied unchanged to new images (v_new @ P)."""
    residual = v - Ridge(alpha=alpha).fit(ref, v).predict(ref)          # Eq. 1-2
    U = PCA(r, random_state=0).fit(residual).components_                 # top-r residual directions
    P = np.eye(v.shape[1], dtype=v.dtype) - U.T @ U
    return l2(v @ P), P                                                  # Eq. 3


def fuse(parts, dim=32):
    """Concatenate the cleaned embeddings, reduce with PCA and re-normalise (Eq. 4)."""
    x = l2(np.hstack(parts))
    pca = PCA(dim, random_state=0).fit(x)
    return l2(pca.transform(x)), pca


def cluster(z, k=16, n_neighbors=15, seed=0):
    """Spectral clustering on a k-nearest-neighbour graph; k = median SCMax choice over 9 seeds."""
    return SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=n_neighbors,
                              random_state=seed, assign_labels="cluster_qr").fit_predict(z)


class KnnOOD:
    """kNN out-of-distribution detector (Sun et al., 2022): distance to the k-th nearest D_main image (Eq. 5);
    an image is 'unknown' above the q-th percentile of the D_main scores (leave-one-out)."""

    def __init__(self, z_main, k=10, q=95):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(z_main)
        self.threshold = np.percentile(self.nn.kneighbors(z_main)[0][:, k], q)

    def score(self, z):
        return self.nn.kneighbors(z, self.k)[0][:, -1]

    def is_unknown(self, z):
        return self.score(z) > self.threshold


def run(sig, pe, dino, k=16):
    """Full pipeline on D_main: returns the fused features, cluster labels and the OOD detector."""
    sig_clean, _ = cmre(sig, dino)
    pe_clean, _ = cmre(pe, dino)
    z, _ = fuse([sig_clean, pe_clean, dino])
    return z, cluster(z, k), KnnOOD(z)


if __name__ == "__main__":  # smoke test on random embeddings
    g = np.random.default_rng(0)
    sig, pe, dino = (l2(g.normal(size=(300, d))).astype(np.float32) for d in (128, 160, 96))
    z, y, ood = run(sig, pe, dino, k=5)
    assert z.shape == (300, 32) and len(set(y)) == 5
    assert ood.is_unknown(z).mean() <= 0.06          # training images are rarely flagged
    print("ewaste_cmre smoke test ok")
