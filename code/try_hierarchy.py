"""Build a strictly nested, post hoc hierarchy over the final 16 CMRE clusters."""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / "results" / "modal"
OUT = MODAL / "hierarchy"
NAMES = [
    "Aki", "Baterai laptop", "Turntable", "TV CRT",
    "Baterai ponsel/kamera", "Pemutar CD/radio", "Mesin cuci atas",
    "TV layar datar", "Mesin cuci depan", "Laptop", "Keyboard",
    "Microwave", "Printer", "Mouse", "Ponsel", "PCB/campuran",
]
LEVELS = (16, 13, 12, 8)
BOOTSTRAPS = 200


def clades(tree):
    nodes = {i: frozenset([i]) for i in range(16)}
    for step, (left, right, _, _) in enumerate(tree):
        nodes[16 + step] = nodes[int(left)] | nodes[int(right)]
    return nodes


def partition(tree, k):
    raw = cut_tree(tree, n_clusters=k).ravel()
    ordered = sorted(set(raw), key=lambda group: np.flatnonzero(raw == group)[0])
    return np.array([ordered.index(group) for group in raw], dtype=int)


def xsrc_score(rows, mapped):
    scores = []
    for category in ("battery", "keyboard", "mobile", "mouse", "pcb"):
        kaan = (rows.source == "kaan") & (rows.posthoc_label == category)
        bangla = (rows.source == "bangla") & (rows.posthoc_label == category)
        if kaan.any() and bangla.any():
            target = np.bincount(mapped[kaan]).argmax()
            scores.append((mapped[bangla] == target).mean())
    return float(np.mean(scores))


def main_purity(rows, mapped):
    valid = (rows["set"] == "main") & rows.posthoc_label.notna()
    table = pd.crosstab(mapped[valid], rows.loc[valid, "posthoc_label"])
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def main():
    rows = pd.read_csv(MODAL / "final_v4" / "assignments.csv")
    vectors = np.load(MODAL / "z_final_v4.npy")
    if len(rows) != len(vectors) or not np.array_equal(rows.id, np.arange(len(rows))):
        raise ValueError("Embedding and assignments do not align by id")

    main_mask = (rows["set"] == "main").to_numpy()
    labels = rows.loc[main_mask, "cluster"].to_numpy(dtype=int)
    groups = [vectors[main_mask][labels == i] for i in range(16)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("At least one final cluster has no main images")

    centroids = np.stack([group.mean(axis=0) for group in groups])
    tree = linkage(centroids, method="average", metric="cosine", optimal_ordering=True)
    original_clades = clades(tree)
    alternative_clades = [
        set(clades(linkage(centroids, method=method, metric="cosine")).values())
        for method in ("single", "complete")
    ]

    rng = np.random.default_rng(20260926)
    support = Counter()
    for _ in range(BOOTSTRAPS):
        sampled = np.stack([
            group[rng.integers(len(group), size=len(group))].mean(axis=0)
            for group in groups
        ])
        recovered = set(clades(linkage(sampled, method="average", metric="cosine")).values())
        support.update(clade for clade in original_clades.values() if len(clade) > 1 and clade in recovered)

    OUT.mkdir(parents=True, exist_ok=True)
    merges = []
    for step, (_, _, distance, _) in enumerate(tree):
        members = original_clades[16 + step]
        merges.append({
            "step": step + 1,
            "k_after": 15 - step,
            "node": 16 + step,
            "leaf_clusters": ",".join(map(str, sorted(members))),
            "leaf_names": " + ".join(NAMES[i] for i in sorted(members)),
            "n_main": sum(len(groups[i]) for i in members),
            "cosine_distance": distance,
            "bootstrap_clade_support": support[members] / BOOTSTRAPS,
            "single_complete_agree": all(members in other for other in alternative_clades),
        })
    pd.DataFrame(merges).to_csv(OUT / "merges.csv", index=False)

    mapping = pd.DataFrame({"cluster": np.arange(16), "name": NAMES})
    evaluation = []
    assigned = rows.cluster.to_numpy(dtype=int)
    for k in LEVELS:
        parent = np.arange(16) if k == 16 else partition(tree, k)
        mapping[f"parent_k{k}"] = parent
        mapped = parent[assigned]
        evaluation.append({
            "k": k,
            "main_bdc_purity": main_purity(rows, mapped),
            "kaan_bangla_xsrc": xsrc_score(rows, mapped),
        })
    mapping.to_csv(OUT / "levels.csv", index=False)
    pd.DataFrame(evaluation).to_csv(OUT / "evaluation.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(
        tree, orientation="right", ax=ax,
        labels=[f"C{i:02d}  {name}" for i, name in enumerate(NAMES)],
        leaf_font_size=9, color_threshold=0, above_threshold_color="#235789",
    )
    ax.set_xlabel("Jarak kosinus antarsentroid (average linkage)")
    ax.set_title("Hierarki post hoc dari 16 cluster final CMRE")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "dendrogram.png", dpi=180)
    plt.close(fig)

    print(pd.DataFrame(merges).head(8).to_string(index=False))
    print("\nEvaluasi post hoc:")
    print(pd.DataFrame(evaluation).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
