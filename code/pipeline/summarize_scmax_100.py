"""Summarize the frozen-feature, 100-seed SCMax repeatability run."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "modal"
AUDIT = os.environ.get("SCMAX_SOURCE_AUDIT") == "1"
files = (sorted((RESULTS / "scmax_source_audit_batches").glob("scmax_results_source_audit_*.csv"))
         if AUDIT else [RESULTS / "scmax_results_live.csv", *sorted((RESULTS / "scmax_repeat_batches").glob("*.csv"))])
rows = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
rows = rows[rows.feat.eq("CMRE(sig)+CMRE(pe)+dino") & rows.seed.between(0, 99)]
assert not rows.seed.duplicated().any(), "Duplicate seed in source files."
rows = rows.sort_values("seed").reset_index(drop=True)
assert rows.seed.tolist() == list(range(100)), "Expected exactly seeds 0–99."
rows.to_csv(RESULTS / ("scmax_results_100_source_audit.csv" if AUDIT else "scmax_results_100.csv"), index=False)

assignments = pd.read_csv(RESULTS / "final_v4" / "assignments.csv")
if "set" in assignments:
    main = assignments["set"].eq("main").to_numpy()
else:  # public release keeps id/cluster but omits source metadata
    assert len(assignments) == 7565 and np.array_equal(assignments.id, np.arange(7565))
    main = assignments.id.to_numpy() < 3792
z = np.load(RESULTS / "z_final_v4.npy")[main]
reference = assignments.loc[main, "cluster"].to_numpy()
partitions = {
    int(k): SpectralClustering(int(k), affinity="nearest_neighbors", n_neighbors=15,
                               random_state=0, assign_labels="cluster_qr").fit_predict(z)
    for k in sorted(rows.K_scmax.unique())
}
assert adjusted_rand_score(reference, partitions[16]) > 0.999999
ari = {(a, b): adjusted_rand_score(partitions[a], partitions[b])
       for a in partitions for b in partitions}
chosen = rows.K_scmax.astype(int).to_numpy()
pairwise = np.array([ari[(chosen[i], chosen[j])] for i in range(100) for j in range(i + 1, 100)])


def describe(values):
    values = np.asarray(values)
    return (f"{values.mean():.3f} ± {values.std(ddof=1):.3f}; "
            f"median {np.median(values):.3f}; persentil 2,5–97,5 "
            f"{np.quantile(values, .025):.3f}–{np.quantile(values, .975):.3f}; "
            f"rentang {values.min():.3f}–{values.max():.3f}")


lines = [
    "# Audit ulang 100 seed SCMax terhadap kode penulis" if AUDIT else "# Audit 100 seed SCMax pada fitur CMRE beku",
    "",
    ("Seed 0–9 dijalankan pada [Modal run 1](https://modal.com/apps/apasijannn/main/ap-Cji9Qb0yzZ0FAq2f7H9ZLy) "
     "dan seed 10–99 pada [Modal run 2](https://modal.com/apps/apasijannn/main/ap-0yzkv8Yq8DI68bj4s3Qeng). "
     "Seluruhnya memakai GPU L4, fitur CMRE beku, dan pengaturan seed persis seperti demo.py penulis."
     if AUDIT else
     "Seed 0–7 berasal dari checkpoint sebelumnya; seed 8–9 dijalankan pada "
     "[Modal run 1](https://modal.com/apps/apasijannn/main/ap-ebKIp7sM2zYNWKmllOptcc) "
     "dan seed 10–99 pada [Modal run 2](https://modal.com/apps/apasijannn/main/ap-Q7pB7VUVqiuOkNuVZn5v7M). "
     "Seluruhnya memakai GPU L4, kode SCMax dan input yang sama."),
    "",
    "Seed 0–99; SCMax memilih K tanpa label. Spectral clustering memakai K pilihan tiap seed, "
    "fitur PCA-32 yang sama, 15 tetangga, dan random_state=0. Data eksternal "
    "dipetakan ke centroid partisi utama. Ini menguji seed SCMax pada data tetap, "
    "bukan bootstrap data atau 100 set uji independen.",
    "",
    f"- K: median {np.median(chosen):g}; modus {rows.K_scmax.mode().iloc[0]:g}; "
    f"K=16 pada {(chosen == 16).sum()}/100 seed.",
    f"- xsrc hybrid: {describe(rows.hybrid_xsrc_agree)}.",
    f"- NMI objek eksternal hybrid: {describe(rows.hybrid_ext_obj_nmi)}.",
    f"- ARI hybrid antarpasangan seed (4.950 pasangan): {describe(pairwise)}.",
    f"- ARI hybrid terhadap partisi final K=16: "
    f"{describe([ari[(k, 16)] for k in chosen])}.",
    "",
    "| K pilihan | Seed | xsrc hybrid | NMI eksternal | ARI vs K=16 |",
    "|---:|---:|---:|---:|---:|",
]
for k, group in rows.groupby("K_scmax", sort=True):
    lines.append(f"| {int(k)} | {len(group)} | {group.hybrid_xsrc_agree.iloc[0]:.3f} | "
                 f"{group.hybrid_ext_obj_nmi.iloc[0]:.3f} | {ari[(int(k), 16)]:.3f} |")
lines += ["", "Persentil 2,5–97,5% antar-seed adalah sebaran algoritmik, bukan interval "
          "kepercayaan generalisasi. Bootstrap data yang menghitung ulang pipeline "
          "diperlukan untuk menilai perubahan sampel.", ""]
(RESULTS / ("scmax_100_source_audit_report.md" if AUDIT else "scmax_100_report.md")).write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
