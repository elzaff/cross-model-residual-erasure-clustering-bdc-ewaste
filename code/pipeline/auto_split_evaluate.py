"""Evaluate the already frozen label-free auto-split decision."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from common import SHARED
import rq1_metrics as metrics


root = Path(__file__).resolve().parents[2]
out = root / "results/modal/auto_split"
z = np.load(root / "results/modal/z_final_v4.npy")
assign = pd.read_csv(root / "results/modal/final_v4/assignments.csv", keep_default_na=False)
main = assign["set"].eq("main").to_numpy()
base = assign.loc[main, "cluster"].to_numpy(dtype=int)
selected = np.load(out / "main_labels.npy")
assert selected.shape == base.shape
changed = np.flatnonzero(selected != base)
parent = int(np.unique(base[changed]).item())
assert len(changed) and np.all(selected[changed] == 16)
assert np.array_equal(base[selected != 16], selected[selected != 16])

lab = assign["posthoc_label"].to_numpy()
src = assign["source"].to_numpy()
style = assign["style"].to_numpy()
cat = main & (lab != "")
shared = (~main) & np.isin(lab, SHARED) & np.isin(src, ["kaan", "bangla", "karan"])
external = (~main) & (lab != "")
class_rows = []


def evaluate(name, labels):
    zm = z[main]
    centers = np.stack([zm[labels == c].mean(axis=0) for c in range(labels.max() + 1)])
    y = ((z[:, None] - centers[None]) ** 2).sum(axis=-1).argmin(axis=1)
    y[main] = labels
    score = metrics.score_labels(y, main, style, lab, src, cat, shared)
    for category in SHARED:
        catalog = shared & (src == "kaan") & (lab == category)
        phone = shared & (src == "bangla") & (lab == category)
        if catalog.any() and phone.any():
            majority = np.bincount(y[catalog]).argmax()
            class_rows.append(dict(method=name, category=category, phone_n=int(phone.sum()),
                                   match_n=int((y[phone] == majority).sum())))
    score.update(method=name,
                 purity_bdc=pd.crosstab(y[cat], lab[cat]).max(axis=1).sum() / cat.sum(),
                 purity_external=pd.crosstab(y[external], lab[external]).max(axis=1).sum() / external.sum())
    return score


pd.DataFrame([evaluate("K16", base), evaluate("automatic_split", selected)]).set_index(
    "method").to_csv(out / "evaluation.csv")
pd.DataFrame(class_rows).to_csv(out / "xsrc_by_class.csv", index=False)

manifest = pd.read_csv(root / "data/manifest.csv", keep_default_na=False)
paths = pd.concat([manifest[(manifest.source == source) & (manifest.status == "keep")]
                   for source in ("bdc", "bdc_test")], ignore_index=True).path
assert len(paths) == len(base) == 3792


def contact_sheet(indices, filename):
    indices = np.random.default_rng(2026).choice(indices, min(24, len(indices)), replace=False)
    canvas = Image.new("RGB", (6 * 190, 4 * 160), "white")
    draw = ImageDraw.Draw(canvas)
    for j, index in enumerate(indices):
        x, y = j % 6 * 190, j // 6 * 160
        with Image.open(paths.iloc[index]) as photo:
            thumb = ImageOps.contain(photo.convert("RGB"), (180, 125))
        canvas.paste(thumb, (x + (190 - thumb.width) // 2, y + 25))
        draw.text((x + 4, y + 4), f"ID {int(assign.loc[main].iloc[index].id)}", fill="black")
    canvas.save(out / filename)


contact_sheet(np.flatnonzero(selected == 16), "new_group_random.jpg")
contact_sheet(np.flatnonzero(selected == parent), "parent_remainder_random.jpg")
print(f"Selected C{parent}: new group {len(changed)}; remainder {(selected == parent).sum()}")
print(pd.read_csv(out / "evaluation.csv").round(4).to_string(index=False))
