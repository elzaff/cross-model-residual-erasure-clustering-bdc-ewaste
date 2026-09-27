"""Summarize native SCMax merge levels and compare partitions at matched K."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from common import SHARED
import rq1_metrics as metrics


root = Path(__file__).resolve().parents[2]
input_dir = root / "results/modal"
out = input_dir / "scmax_hierarchy_audit"
out.mkdir(parents=True, exist_ok=True)
batch = input_dir / "scmax_source_audit_batches"
levels = pd.concat([pd.read_csv(p) for p in sorted(batch.glob("scmax_levels_source_audit_*.csv"))],
                   ignore_index=True)
levels = levels[levels.seed.between(0, 99)].sort_values(["seed", "K"], ascending=[True, False])
levels["merge_level"] = levels.groupby("seed").cumcount() + 1
summary = levels.groupby("merge_level").agg(
    runs=("seed", "nunique"), K_min=("K", "min"), K_median=("K", "median"), K_max=("K", "max"),
    nnc_mean=("nnc", "mean"), silhouette_euclidean_mean=("sil", "mean"),
    xsrc_mean=("xsrc_agree", "mean"), purity_bdc_mean=("purity_bdc", "mean"),
    source_leak_mean=("ext_src_given_obj", "mean"))
summary.to_csv(out / "levels_100_seeds.csv")

results = pd.read_csv(input_dir / "scmax_results_100_source_audit.csv")
matched = results.groupby("K_scmax").agg(
    runs=("seed", "size"), native_xsrc_mean=("scmax_xsrc_agree", "mean"),
    spectral_xsrc_mean=("hybrid_xsrc_agree", "mean"),
    native_purity_bdc_mean=("scmax_purity_bdc", "mean"),
    spectral_purity_bdc_mean=("hybrid_purity_bdc", "mean"),
    native_source_leak_mean=("scmax_ext_src_given_obj", "mean"),
    spectral_source_leak_mean=("hybrid_ext_src_given_obj", "mean"))
matched.to_csv(out / "native_vs_spectral_by_K.csv")

path = out / "seed1_levels.npz"
if path.exists():
    saved = np.load(path)
    z = np.load(input_dir / "z_final_v4.npy")
    assign = pd.read_csv(input_dir / "final_v4/assignments.csv", keep_default_na=False)
    main = assign["set"].eq("main").to_numpy()
    zm = z[main]
    final = assign.loc[main, "cluster"].to_numpy(dtype=int)
    lab = assign.posthoc_label.to_numpy()
    src = assign.source.to_numpy()
    style = assign["style"].to_numpy()
    cat = main & (lab != "")
    shared = (~main) & np.isin(lab, SHARED) & np.isin(src, ["kaan", "bangla", "karan"])
    external = (~main) & (lab != "")
    local_metrics = []

    def evaluate(name, labels):
        centers = np.stack([zm[labels == c].mean(axis=0) for c in np.unique(labels)])
        y = ((z[:, None] - centers[None]) ** 2).sum(axis=-1).argmin(axis=1)
        y[main] = labels
        score = metrics.score_labels(y, main, style, lab, src, cat, shared)
        score.update(method=name, K=int(np.unique(labels).size),
                     purity_bdc=pd.crosstab(y[cat], lab[cat]).max(axis=1).sum() / cat.sum(),
                     purity_external=pd.crosstab(y[external], lab[external]).max(axis=1).sum() / external.sum())
        local_metrics.append(score)

    fine_rows = []
    for key in saved.files:
        native = saved[key]
        K = np.unique(native).size
        evaluate(f"native_{key}", native)
        row = dict(level=key, K=K, ari_native_final16=adjusted_rand_score(native, final),
                   native_cosine_silhouette=silhouette_score(zm, native, metric="cosine"))
        if K <= 25:
            spectral = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15,
                                          random_state=0, assign_labels="cluster_qr").fit_predict(zm)
            ward = AgglomerativeClustering(K).fit_predict(zm)
            evaluate(f"spectral_K{K}", spectral)
            evaluate(f"ward_K{K}", ward)
            pd.crosstab(pd.Series(native, name="native"), pd.Series(spectral, name="spectral")).to_csv(
                out / f"local_{key}_native_vs_spectral.csv")
            row.update(ari_native_spectral=adjusted_rand_score(native, spectral),
                       ari_native_ward=adjusted_rand_score(native, ward),
                       ari_spectral_ward=adjusted_rand_score(spectral, ward),
                       spectral_cosine_silhouette=silhouette_score(zm, spectral, metric="cosine"),
                       ward_cosine_silhouette=silhouette_score(zm, ward, metric="cosine"))
        fine_rows.append(row)
    pd.DataFrame(fine_rows).to_csv(out / "seed1_partition_comparison.csv", index=False)
    pd.DataFrame(local_metrics).to_csv(out / "seed1_local_metrics.csv", index=False)
    print(pd.DataFrame(fine_rows).round(4).to_string(index=False))
    print(pd.DataFrame(local_metrics).round(4).to_string(index=False))

print(summary.round(4).to_string())
print(matched.round(4).to_string())
