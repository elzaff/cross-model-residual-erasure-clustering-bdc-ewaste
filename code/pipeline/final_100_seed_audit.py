"""Audit random_state=0..99 for frozen CMRE v4 Spectral-kNN at fixed K=16.

Run locally: python code/pipeline/final_100_seed_audit.py
Inputs are the frozen PCA-32 features and seed-0 final assignments. No model is
retrained, no new data split is drawn, and SCMax is not rerun here.
"""

from __future__ import annotations

import csv
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

from common import SHARED
import rq1_metrics as metrics


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "modal"
OUTPUT = INPUT / "final_seed_100"
K = 16
N_SEEDS = 100


def align_and_jaccard(reference: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    overlap = np.zeros((K, K), dtype=np.int64)
    np.add.at(overlap, (reference, labels), 1)
    ref_ids, run_ids = linear_sum_assignment(-overlap)
    mapping = np.empty(K, dtype=np.int64)
    mapping[run_ids] = ref_ids
    aligned = mapping[labels]
    ref_sizes = np.bincount(reference, minlength=K)
    run_sizes = np.bincount(aligned, minlength=K)
    intersection = np.bincount(reference[reference == aligned], minlength=K)
    union = ref_sizes + run_sizes - intersection
    return aligned, intersection / union


def run_one(seed: int, z: np.ndarray, main: np.ndarray, reference: np.ndarray,
            lab: np.ndarray, src: np.ndarray, style: np.ndarray,
            cat: np.ndarray, shared: np.ndarray) -> tuple[dict, np.ndarray]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        labels = SpectralClustering(
            K, affinity="nearest_neighbors", n_neighbors=15,
            random_state=seed, assign_labels="cluster_qr",
        ).fit_predict(z[main])
    counts = np.bincount(labels, minlength=K)
    if len(counts) != K or np.any(counts == 0):
        raise ValueError(f"Expected {K} non-empty clusters, got {counts.tolist()}")
    aligned, jaccard = align_and_jaccard(reference, labels)

    # final_max.py keeps Spectral labels on D_main and sends other images to
    # nearest D_main centroid in the same frozen PCA-32 coordinate system.
    zm = z[main]
    centroids = np.stack([zm[labels == c].mean(axis=0) for c in range(K)])
    y = ((z[:, None] - centroids[None]) ** 2).sum(axis=-1).argmin(axis=1)
    y[np.flatnonzero(main)] = labels
    score = metrics.score_labels(y, main, style, lab, src, cat, shared)
    purity_bdc = pd.crosstab(y[cat], lab[cat]).max(axis=1).sum() / cat.sum()
    external = (~main) & (lab != "")
    purity_external = pd.crosstab(y[external], lab[external]).max(axis=1).sum() / external.sum()
    row = dict(
        seed=seed, status="ok", seconds=round(time.perf_counter() - started, 3),
        warnings=" | ".join(sorted(set(str(w.message) for w in seen))),
        clusters=int(np.count_nonzero(counts)), min_cluster_size=int(counts.min()),
        max_cluster_size=int(counts.max()),
        ari_vs_reference=adjusted_rand_score(reference, labels),
        matched_main_agreement=float((aligned == reference).mean()),
        mean_cluster_jaccard=float(jaccard.mean()),
        min_cluster_jaccard=float(jaccard.min()),
        max_cluster_jaccard=float(jaccard.max()),
        max_abs_cluster_size_change=int(np.max(np.abs(np.bincount(aligned, minlength=K) - np.bincount(reference, minlength=K)))),
        purity_bdc=float(purity_bdc), purity_external=float(purity_external), **score,
        error="",
    )
    return row, labels


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    z = np.load(INPUT / "z_final_v4.npy")
    assignments = pd.read_csv(INPUT / "final_v4" / "assignments.csv", keep_default_na=False)
    assert z.shape == (len(assignments), 32)
    main_mask = (assignments["set"] == "main").to_numpy()
    assert int(main_mask.sum()) == 3792
    reference = assignments.loc[main_mask, "cluster"].to_numpy(dtype=np.int64)
    assert sorted(np.unique(reference).tolist()) == list(range(K))
    lab = assignments["posthoc_label"].to_numpy()
    src = assignments["source"].to_numpy()
    style = assignments["style"].to_numpy()
    cat = main_mask & (lab != "")
    shared = (~main_mask) & np.isin(lab, SHARED) & np.isin(src, ["kaan", "bangla", "karan"])

    rows_path = OUTPUT / "runs.csv"
    labels_path = OUTPUT / "main_labels.npy"
    label_matrix = np.load(labels_path) if labels_path.exists() else np.full((N_SEEDS, main_mask.sum()), -1, dtype=np.int16)
    assert label_matrix.shape == (N_SEEDS, main_mask.sum())
    existing = {int(r["seed"]): r for r in csv.DictReader(rows_path.open(encoding="utf-8"))} if rows_path.exists() else {}
    for seed in range(N_SEEDS):
        if seed in existing and existing[seed]["status"] == "ok" and np.all(label_matrix[seed] >= 0):
            continue
        try:
            row, labels = run_one(seed, z, main_mask, reference, lab, src, style, cat, shared)
            if seed == 0 and not np.array_equal(labels, reference):
                raise AssertionError("Seed 0 differs from frozen final_v4 assignments")
            label_matrix[seed] = labels
        except Exception:
            row = dict(seed=seed, status="error", error=traceback.format_exc())
            label_matrix[seed] = -1
        existing[seed] = row
        pd.DataFrame([existing[i] for i in sorted(existing)]).to_csv(rows_path, index=False)
        np.save(labels_path, label_matrix)
        print(f"seed {seed:02d}: {row['status']} | ARI={row.get('ari_vs_reference', float('nan')):.6f} | {row.get('seconds', float('nan')):.1f}s", flush=True)

    # Separate all-pair distribution from reference-relative metrics.
    successful = [s for s in range(N_SEEDS) if existing[s]["status"] == "ok"]
    pairs = [(a, b, adjusted_rand_score(label_matrix[a], label_matrix[b]))
             for i, a in enumerate(successful) for b in successful[i + 1:]]
    pd.DataFrame(pairs, columns=["seed_a", "seed_b", "ari"]).to_csv(OUTPUT / "pairwise_ari.csv", index=False)
    print(f"complete: {len(successful)}/{N_SEEDS} successes; {len(pairs)} pairwise ARI values", flush=True)


if __name__ == "__main__":
    main()
