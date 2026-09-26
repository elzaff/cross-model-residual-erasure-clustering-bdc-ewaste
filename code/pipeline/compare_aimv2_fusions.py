"""Screen AIMv2 fusion variants under the final PCA-32 + Spectral K=16 protocol."""
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
os.environ["EXP_OUT"] = str(ROOT / "data" / "embeddings_final_v4")

from common import ari, emb, load_items, reduce  # noqa: E402
import rq1_metrics as metrics  # noqa: E402


def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def tta(name):
    return normalize(emb(name) + emb(name + "_flip"))


def cmre(v, reference, main):
    reg = Ridge(alpha=1.0).fit(reference[main], v[main])
    residual = v[main] - reg.predict(reference[main])
    pca = PCA(64, random_state=0).fit(residual)
    r = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5) + 1)
    basis = pca.components_[:r].astype(np.float32)
    return normalize(v - (v @ basis.T) @ basis), r


def fuse(*parts):
    return normalize(np.hstack(parts))


def score(name, features, main, style, labels, source, cat, shared):
    z = reduce(features[main], features, 32)
    partitions = [SpectralClustering(16, affinity="nearest_neighbors", n_neighbors=15,
                                     random_state=seed, assign_labels="cluster_qr").fit_predict(z[main])
                  for seed in range(3)]
    rows = []
    for partition in partitions:
        centers = np.stack([z[main][partition == c].mean(0) for c in np.unique(partition)])
        assignment = ((z[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        assignment[np.where(main)[0]] = partition
        row = metrics.score_labels(assignment, main, style, labels, source, cat, shared)
        for key, mask in (("purity_bdc", cat), ("purity_ext", (~main) & (labels != ""))):
            groups = np.unique(assignment[mask])
            row[key] = sum(np.unique(labels[mask & (assignment == c)], return_counts=True)[1].max()
                           for c in groups) / mask.sum()
        rows.append(row)
    result = {"method": name, "K": 16, "seed_ari": float(np.mean([ari(partitions[0], p) for p in partitions[1:]]))}
    result.update({key: float(np.mean([row[key] for row in rows])) for key in rows[0]})
    result["xsrc_sd_seed"] = float(np.std([row["xsrc_agree"] for row in rows]))
    print(name, {k: round(v, 4) for k, v in result.items() if isinstance(v, float)}, flush=True)
    return result


def main():
    items = load_items()
    main_mask, style, labels, source, cat, shared = metrics.setup(items)
    aim, sig, pe, dino = (tta(name) for name in ("aimv2", "siglip2", "pecoreg", "dinov3hp"))
    assert len(items) == len(aim) == len(sig) == len(pe) == len(dino)
    sig_cmre, r_sig = cmre(sig, dino, main_mask)
    pe_cmre, r_pe = cmre(pe, dino, main_mask)
    print(f"CMRE effective ranks: SigLIP2={min(r_sig, 64)}, PE={min(r_pe, 64)}", flush=True)
    variants = {
        "AIMv2": aim,
        "Final: CMRE(Sig)+CMRE(PE)+DINO": fuse(sig_cmre, pe_cmre, dino),
        "AIM+Sig(raw)+DINO": fuse(aim, sig, dino),
        "AIM+PE(raw)+DINO": fuse(aim, pe, dino),
        "AIM+Sig(raw)+PE(raw)+DINO": fuse(aim, sig, pe, dino),
        "AIM+CMRE(Sig)+DINO": fuse(aim, sig_cmre, dino),
        "AIM+CMRE(PE)+DINO": fuse(aim, pe_cmre, dino),
        "AIM+CMRE(Sig)+CMRE(PE)+DINO": fuse(aim, sig_cmre, pe_cmre, dino),
    }
    results = [score(name, features, main_mask, style, labels, source, cat, shared)
               for name, features in variants.items()]
    output = ROOT / "results" / "modal" / "aimv2_fusion_results.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
