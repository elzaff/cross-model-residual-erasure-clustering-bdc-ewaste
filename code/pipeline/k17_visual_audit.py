"""Create contact sheets for the additional K=17 group from local source photos."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import SpectralClustering

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "modal" / "k17_visual_audit"
OUT.mkdir(parents=True, exist_ok=True)

assign = pd.read_csv(ROOT / "results/modal/final_v4/assignments.csv", keep_default_na=False)
main = assign["set"].eq("main").to_numpy()
z = np.load(ROOT / "results/modal/z_final_v4.npy")
k17 = SpectralClustering(
    n_clusters=17, affinity="nearest_neighbors", n_neighbors=15,
    random_state=0, assign_labels="cluster_qr",
).fit_predict(z[main])
base = assign.loc[main].copy().reset_index(drop=True)
base["k17"] = k17

manifest = pd.read_csv(ROOT / "data/manifest.csv", keep_default_na=False)
paths = pd.concat([
    manifest[(manifest.source == source) & (manifest.status == "keep")]
    for source in ("bdc", "bdc_test")
], ignore_index=True).path
assert len(paths) == len(base) == 3792
assert all(Path(a).suffix.lower() == Path(b).suffix.lower() for a, b in zip(base.path, paths))
base["original_path"] = paths

group = base[base.k17 == 11].copy()
assert len(group) == 468
center = z[main][k17 == 11].mean(axis=0)
group["centroid_distance"] = np.linalg.norm(z[main][k17 == 11] - center, axis=1)
group.to_csv(OUT / "group11_members.csv", index=False)

def sheet(rows, name, cols=6):
    rows = list(rows)
    cell_w, cell_h, band = 190, 162, 24
    canvas = Image.new("RGB", (cols * cell_w, ((len(rows) + cols - 1) // cols) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for n, (_, row) in enumerate(rows):
        x, y = (n % cols) * cell_w, (n // cols) * cell_h
        with Image.open(row.original_path) as original:
            thumb = ImageOps.contain(original.convert("RGB"), (cell_w - 8, cell_h - band - 8))
        canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + band + (cell_h - band - thumb.height) // 2))
        draw.text((x + 3, y + 4), f"ID {row.id} | C{row.cluster} | {row.posthoc_label or 'unlabeled'}", fill="black")
    canvas.save(OUT / name)

sheet(group.nsmallest(18, "centroid_distance").iterrows(), "nearest_centroid.jpg")
sheet(group.sample(n=30, random_state=2026).iterrows(), "random_30.jpg")
stratified = pd.concat([g.sample(n=min(6, len(g)), random_state=2026) for _, g in group.groupby("cluster")])
sheet(stratified.sort_values("cluster").iterrows(), "by_k16_origin.jpg")
pcb_group = base[base.k17 == 16].copy()
pcb_center = z[main][k17 == 16].mean(axis=0)
pcb_group["centroid_distance"] = np.linalg.norm(z[main][k17 == 16] - pcb_center, axis=1)
sheet(pcb_group.nsmallest(18, "centroid_distance").iterrows(), "k17_group16_nearest.jpg")
sheet(pcb_group.sample(n=24, random_state=2026).iterrows(), "k17_group16_random.jpg")
print("saved", OUT, "members", len(group), "unlabeled", (group.posthoc_label == "").sum())
