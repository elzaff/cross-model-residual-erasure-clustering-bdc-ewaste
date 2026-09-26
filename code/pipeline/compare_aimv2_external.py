"""Compare the BDC-selected four-encoder fusion with frozen v4 on Iliev and Shubha."""
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from common import OUT, ari, load_items, reduce
import rq1_metrics as metrics

CLASSES = {
    "iliev": ["battery", "keyboard", "microwave", "mobile", "mouse", "pcb", "player", "printer", "television", "washing_machine"],
    "shubha": ["keyboard", "mobile", "mouse", "pcb"],
}
SHUBHA_OOD = ["Remote", "Dryer", "Headphone", "Modem", "Pendrive"]


def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def tta(name, directory=OUT):
    def load(suffix):
        x = np.load(f"{directory}/emb_{name}{suffix}.npy").astype(np.float32)
        return normalize(x)
    return normalize(load("") + load("_flip"))


def cmre(v, reference, main):
    residual = v[main] - Ridge(alpha=1.0).fit(reference[main], v[main]).predict(reference[main])
    pca = PCA(64, random_state=0).fit(residual)
    r = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5) + 1)
    basis = pca.components_[:r].astype(np.float32)
    return normalize(v - (v @ basis.T) @ basis)


def fuse(*parts):
    return normalize(np.hstack(parts))


def evaluate(tag, name, features, main, frozen, lab, src, cat, ext_lab, ext_cls, classes):
    z = reduce(features[main], features, 32)
    if frozen is None:
        partitions = [SpectralClustering(16, affinity="nearest_neighbors", n_neighbors=15,
                                         random_state=seed, assign_labels="cluster_qr").fit_predict(z[main])
                      for seed in range(3)]
        partition = partitions[0]
        stability = float(np.mean([ari(partition, p) for p in partitions[1:]]))
    else:
        partition, stability = frozen, 1.0
    centers = np.stack([z[main][partition == c].mean(0) for c in range(16)])
    distances2 = ((z[:, None] - centers[None]) ** 2).sum(-1)
    labels = distances2.argmin(1)
    labels[np.where(main)[0]] = partition
    n_old = len(lab)
    predicted = labels[n_old:]
    majority = {c: np.bincount(labels[:n_old][(src == "kaan") & (lab == c)]).argmax() for c in classes}
    matched = np.array([predicted[i] == majority.get(c, -1) for i, c in enumerate(ext_lab)])
    xsrc = float(np.mean([matched[ext_lab == c].mean() for c in classes if (ext_lab == c).any()]))
    object_majority = {k: pd.Series(lab[cat][labels[:n_old][cat] == k]).mode()[0]
                       for k in np.unique(labels[:n_old][cat])}
    lenient = float(np.mean([np.mean([object_majority.get(k) == c for k in predicted[ext_lab == c]])
                             for c in classes if (ext_lab == c).any()]))
    mapped = np.isin(ext_lab, classes)
    purity = float(pd.crosstab(predicted[mapped], ext_lab[mapped]).max(axis=1).sum() / mapped.sum())
    if tag == "iliev":
        in_dist = np.isin(ext_cls, np.unique(ext_cls[mapped])) | (ext_cls == "Laptop")
        ood_score = np.sqrt(distances2.min(1))[n_old:]
        auroc = float(roc_auc_score(~in_dist, ood_score))
    else:
        in_dist, out_dist = mapped, np.isin(ext_cls, SHUBHA_OOD)
        neighbors = NearestNeighbors(n_neighbors=10).fit(z[main])
        ood_score = neighbors.kneighbors(z[n_old:], 10)[0][:, 9]
        selected = in_dist | out_dist
        auroc = float(roc_auc_score(out_dist[selected], ood_score[selected]))
    result = dict(set=tag, method=name, xsrc=xsrc, xsrc_lenient=lenient, purity=purity,
                  ood_auroc=auroc, seed_ari=stability)
    print(result, flush=True)
    return result, matched


def main():
    items = load_items()
    main_old, _, labels_old, source, cat, _ = metrics.setup(items)
    assignments = pd.read_csv(f"{OUT}/final_v4/assignments.csv", dtype={"id": str})
    frozen = dict(zip(assignments.id, assignments.cluster))
    frozen_main = np.array([frozen[r["id"]] for r, is_main in zip(items, main_old) if is_main])
    rows = []
    for tag, classes in CLASSES.items():
        meta = pd.read_csv(f"{OUT}/{tag}/items.csv", keep_default_na=False)
        control = (meta["set"] == "control").values
        keep = ~control
        if tag == "shubha":
            keep &= (meta.posthoc_label != "EXCLUDE").values
        ext_lab, ext_cls = meta.posthoc_label.values[keep], meta.cls.values[keep]
        main = np.r_[main_old, np.zeros(keep.sum(), dtype=bool)]
        embeddings = {}
        for name in ("aimv2", "siglip2", "pecoreg", "dinov3hp"):
            old, new = tta(name), tta(name, f"{OUT}/{tag}")
            cosine = (old[meta.cls.values[control].astype(int)] * new[control]).sum(1)
            if cosine.min() < 0.98:
                raise ValueError(f"{tag} {name}: control cosine minimum {cosine.min():.4f}")
            embeddings[name] = np.vstack([old, new[keep]])
        aim, sig, pe, dino = (embeddings[k] for k in ("aimv2", "siglip2", "pecoreg", "dinov3hp"))
        sig_c, pe_c = cmre(sig, dino, main), cmre(pe, dino, main)
        variants = (("v4 frozen", fuse(sig_c, pe_c, dino), frozen_main),
                    ("v4 recomputed", fuse(sig_c, pe_c, dino), None),
                    ("AIMv2+CMRE(Sig)+CMRE(PE)+DINO", fuse(aim, sig_c, pe_c, dino), None))
        matched = {}
        for name, features, partition in variants:
            result, matched[name] = evaluate(tag, name, features, main, partition,
                                             labels_old, source, cat, ext_lab, ext_cls, classes)
            rows.append(result)
        for baseline in ("v4 frozen", "v4 recomputed"):
            rng = np.random.default_rng(0)
            differences = []
            for _ in range(2000):
                per_class = []
                for c in classes:
                    ids = np.flatnonzero(ext_lab == c)
                    if len(ids):
                        sample = rng.choice(ids, len(ids), replace=True)
                        per_class.append((matched[variants[-1][0]][sample].astype(int) -
                                          matched[baseline][sample].astype(int)).mean())
                differences.append(np.mean(per_class))
            print(tag, "versus", baseline, "paired_delta_xsrc", round(float(np.mean(differences)), 4),
                  "bootstrap_95", np.percentile(differences, [2.5, 97.5]), flush=True)
    path = f"{OUT}/aimv2_external_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Saved", path, flush=True)


if __name__ == "__main__":
    main()
