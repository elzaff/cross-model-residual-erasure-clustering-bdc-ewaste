"""Occlusion sensitivity of the frozen three-backbone CMRE pipeline.

Run with MODAL_PROFILE=riset-kp modal run cmre_focus_modal.py. Each BDC image is
re-embedded after one of 100 equal-sized cells is replaced by neutral gray.
Heat is 1 - cosine(original, occluded) in the final PCA-32 representation.
This measures sensitivity, not attention or a pixel-level explanation.
"""
from pathlib import Path
import sys

import modal

sys.path.insert(0, "/root/exp")
from modal_app import HF, image, vol

app = modal.App("cmre-focus", image=image)
# BDC main images whose object is separated from a visible background (IDs checked against items.csv labels
# and the original files; the first list of 27 Sep named other images). Boxes drawn before any map was computed.
IDS = [1450, 521, 2920, 3037, 156, 1590, 1912, 2671, 1670, 3322]
NAMES = {
    1450: "Microwave (latar putih)", 521: "Ponsel di atas meja", 2920: "Mesin cuci tabung ganda",
    3037: "Mesin cuci front-load", 156: "Laptop (tampak bawah)", 1590: "Microwave (kontras rendah)",
    1912: "Mouse", 2671: "TV CRT", 1670: "Ponsel di atas kain bermotif", 3322: "Baterai silinder",
}
# Approximate object rectangles (x0, y0, x1, y1) as fractions of width and height.
OBJECT_BOX = {
    1450: (.04, .04, .97, .93), 521: (.00, .12, .97, .80), 2920: (.12, .03, .88, .98),
    3037: (.13, .03, .88, 1.0), 156: (.00, .08, .98, .72), 1590: (.08, .20, .98, .85),
    1912: (.33, .22, .86, 1.0), 2671: (.05, .00, .74, 1.0), 1670: (.25, .00, .85, .85),
    3322: (.30, .18, .62, .95),
}
GRID = 10


@app.function(gpu="A100-80GB", cpu=8, memory=65536, volumes={"/vol": vol},
              timeout=3600, secrets=HF)
def compute():
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from transformers import AutoModel, AutoProcessor
    import timm

    def nz(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    def tta(name):
        a = nz(np.load(f"/vol/out/emb_{name}.npy").astype(np.float32))
        b = nz(np.load(f"/vol/out/emb_{name}_flip.npy").astype(np.float32))
        return nz(a + b)

    items = pd.read_csv("/vol/out/items.csv")
    assert all(items.source[i] == "bdc" and items.set[i] == "main" for i in IDS)
    main = items.set.eq("main").to_numpy()
    sig, pe, dino = (tta(name) for name in ("siglip2", "pecoreg", "dinov3hp"))
    basis = []
    for view in (sig, pe):
        residual = view[main] - Ridge(alpha=1.0).fit(dino[main], view[main]).predict(dino[main])
        basis.append(PCA(64, random_state=0).fit(residual).components_.astype(np.float32))

    def fuse(a, b, c):
        return nz(np.hstack([a, b, c]))

    def project(view, vectors):
        u = basis[view]
        return nz(vectors - (vectors @ u.T) @ u)

    raw_all = fuse(sig, pe, dino)
    cmre_all = fuse(project(0, sig), project(1, pe), dino)
    pca_raw = PCA(32, random_state=0).fit(raw_all[main])
    pca_cmre = PCA(32, random_state=0).fit(cmre_all[main])
    frozen = np.load("/vol/out/z_final_v4.npy")
    reconstructed = nz(pca_cmre.transform(cmre_all))
    reconstruction_mae = float(np.abs(reconstructed - frozen).mean())
    print("Frozen v4 reconstruction MAE:", reconstruction_mae, flush=True)
    assert reconstruction_mae < 1e-4

    originals = [Image.open(items.path[i]).convert("RGB") for i in IDS]
    def occlusions(size):
        arrays = []
        for original in originals:
            base = np.asarray(original.resize((size, size), Image.Resampling.BICUBIC))
            arrays.append(base)
            for row in range(GRID):
                for col in range(GRID):
                    covered = base.copy()
                    y0, y1 = round(row * size / GRID), round((row + 1) * size / GRID)
                    x0, x1 = round(col * size / GRID), round((col + 1) * size / GRID)
                    covered[y0:y1, x0:x1] = [124, 116, 104]
                    arrays.append(covered)
        return np.stack(arrays)

    def embed(arrays, model, forward, batch=4):
        out = []
        with torch.inference_mode():
            for start in range(0, len(arrays), batch):
                chunk = arrays[start:start + batch]
                pair = []
                for flip in (False, True):
                    batch_images = chunk[:, :, ::-1].copy() if flip else chunk
                    pair.append(nz(forward(model, batch_images).float().cpu().numpy()))
                out.append(nz(pair[0] + pair[1]))
        return np.vstack(out)

    def norm_tensor(arr, size, mean, std):
        x = torch.from_numpy(arr).to("cuda").permute(0, 3, 1, 2).float() / 255
        if x.shape[-1] != size:
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        return ((x - torch.tensor(mean, device="cuda")[:, None, None]) /
                torch.tensor(std, device="cuda")[:, None, None]).half()

    a384, a448 = occlusions(384), occlusions(448)
    sig_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-naflex", dtype=torch.float16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-naflex")
    def sig_forward(model, arr):
        x = processor(images=list(arr), return_tensors="pt").to("cuda")
        x["pixel_values"] = x["pixel_values"].half()
        output = model.get_image_features(**x)
        return getattr(output, "pooler_output", output)
    sig_new = embed(a384, sig_model, sig_forward)
    del sig_model, processor
    gc.collect(); torch.cuda.empty_cache()

    imn_mean, imn_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    dino_model = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m",
                                          dtype=torch.bfloat16).to("cuda").eval()
    dino_new = embed(a448, dino_model, lambda model, arr:
                     model(pixel_values=norm_tensor(arr, 224, imn_mean, imn_std).to(torch.bfloat16)).pooler_output)
    del dino_model
    gc.collect(); torch.cuda.empty_cache()

    pe_model = timm.create_model("vit_pe_core_gigantic_patch14_448.fb", pretrained=True,
                                 num_classes=0).to("cuda").eval().to(torch.bfloat16)
    cfg = pe_model.pretrained_cfg
    pe_new = embed(a448, pe_model, lambda model, arr:
                   model(norm_tensor(arr, cfg["input_size"][-1], cfg["mean"], cfg["std"]).to(torch.bfloat16)))
    del pe_model
    gc.collect(); torch.cuda.empty_cache()

    fidelity = {}
    for name, old, new in (("SigLIP2", sig, sig_new), ("PE-Core-G", pe, pe_new), ("DINOv3-H+", dino, dino_new)):
        fidelity[name] = float(np.min(np.sum(old[IDS] * new[::GRID * GRID + 1], axis=1)))
    print("Minimum cosine agreement with frozen input embeddings:", fidelity, flush=True)
    assert min(fidelity.values()) > 0.995

    raw_new = nz(pca_raw.transform(fuse(sig_new, pe_new, dino_new)))
    cmre_new = nz(pca_cmre.transform(fuse(project(0, sig_new), project(1, pe_new), dino_new)))
    result = {"ids": IDS, "reconstruction_mae": reconstruction_mae, "fidelity": fidelity}
    for method, vectors in (("raw", raw_new), ("cmre", cmre_new)):
        scores = []
        for position in range(len(IDS)):
            block = vectors[position * (GRID * GRID + 1):(position + 1) * (GRID * GRID + 1)]
            scores.append(np.maximum(0, 1 - block[1:] @ block[0]).reshape(GRID, GRID).tolist())
        result[method] = scores
    return result


@app.local_entrypoint()
def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    import pandas as pd
    from PIL import Image

    result = compute.remote()
    root = Path(__file__).resolve().parents[2]
    items = pd.read_csv(root / "data/embeddings_final_v4/items.csv")
    manifest = pd.read_csv(root / "data/manifest.csv")
    manifest = manifest[manifest.status.eq("keep")]
    records = []
    screen = []
    for row, image_id in enumerate(IDS):
        raw, cmre = (np.asarray(result[method][row]) for method in ("raw", "cmre"))
        x0, y0, x1, y1 = OBJECT_BOX[image_id]
        rr, cc = np.indices((GRID, GRID))
        object_cells = (x0 <= (cc + .5) / GRID) & ((cc + .5) / GRID <= x1) & \
                       (y0 <= (rr + .5) / GRID) & ((rr + .5) / GRID <= y1)
        metrics = {"id": image_id, "name": NAMES[image_id], "object_cell_count": int(object_cells.sum())}
        for method, scores in (("raw", raw), ("cmre", cmre)):
            obj_mean, bg_mean = scores[object_cells].mean(), scores[~object_cells].mean()
            metrics[f"object_mean_{method}"] = float(obj_mean)
            metrics[f"background_mean_{method}"] = float(bg_mean)
            metrics[f"ratio_{method}"] = float(obj_mean / (bg_mean + 1e-9))
            metrics[f"peak_on_object_{method}"] = bool(object_cells[np.unravel_index(scores.argmax(), scores.shape)])
            for r in range(GRID):
                for c in range(GRID):
                    records.append({"id": image_id, "source": "bdc", "method": method,
                                    "row": r, "col": c, "cosine_change": float(scores[r, c])})
        metrics["ratio_gain"] = metrics["ratio_cmre"] - metrics["ratio_raw"]
        metrics["qualifies"] = (not metrics["peak_on_object_raw"] and metrics["peak_on_object_cmre"]
                                and metrics["ratio_gain"] > 0)
        screen.append(metrics)
    pd.DataFrame(records).to_csv(root / "results/classic/cmre_occlusion.csv", index=False)
    ranking = pd.DataFrame(screen).sort_values("ratio_gain", ascending=False)
    ranking.to_csv(root / "results/classic/cmre_occlusion_screen.csv", index=False)
    print(ranking[["id", "name", "ratio_raw", "ratio_cmre", "ratio_gain",
                   "peak_on_object_raw", "peak_on_object_cmre", "qualifies"]].round(3).to_string(index=False))
    chosen = ranking[ranking.qualifies].head(3)
    if chosen.empty:
        print("No BDC candidate met the predeclared background-to-object criterion.")
        return

    height = 4.25 * len(chosen) + 1.7
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11.5, height), squeeze=False)
    for row, choice in enumerate(chosen.itertuples()):
        image_id = choice.id
        position = list(items[items.source.eq("bdc")].index).index(image_id)
        path = manifest[manifest.source.eq("bdc")].iloc[position].path
        original = Image.open(path).convert("RGB").resize((448, 448))
        index = IDS.index(image_id)
        raw, cmre = (np.asarray(result[method][index]) for method in ("raw", "cmre"))
        scale = max(raw.max(), cmre.max())
        for col, scores in enumerate((None, raw, cmre)):
            ax = axes[row, col]
            ax.imshow(original)
            if scores is None:
                x0, y0, x1, y1 = OBJECT_BOX[image_id]
                ax.add_patch(Rectangle((x0 * 448, y0 * 448), (x1 - x0) * 448, (y1 - y0) * 448,
                                       fill=False, edgecolor="#00e5e5", linewidth=2, linestyle="--"))
                ax.set_xlabel(f"BDC ID {image_id} · kotak objek manual", fontsize=9, labelpad=5)
            else:
                expanded = np.asarray(Image.fromarray(scores.astype(np.float32), mode="F").resize(
                    (448, 448), Image.Resampling.BICUBIC))
                ax.imshow(expanded, cmap="inferno", alpha=.55, vmin=0, vmax=scale)
                ratio = choice.ratio_raw if col == 1 else choice.ratio_cmre
                ax.set_xlabel(f"Rasio objek/latar: {ratio:.2f}×", fontsize=9, labelpad=5)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if col == 0: ax.set_ylabel(NAMES[image_id], fontsize=11, fontweight="bold")
    for col, heading in enumerate(("Foto asli", "Fusi mentah", "CMRE final")):
        axes[0, col].set_title(heading, fontsize=13, fontweight="bold", pad=12)
    fig.suptitle("Contoh BDC: puncak sensitivitas bergeser ke objek", fontsize=16, fontweight="bold", y=.985)
    fig.text(.5, .42 / height, "Dipilih dari 10 foto BDC: puncak latar→objek dan rasio objek/latar naik. "
             "Grid 10×10, interpolasi hanya untuk tampilan.", ha="center", fontsize=9)
    fig.text(.5, .18 / height, "Area objek ditandai manual sebelum peta dihitung; contoh terpilih ini bukan hasil rata-rata dataset.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.08, right=.98, top=1 - 1.0 / height, bottom=.9 / height,
                        wspace=.04, hspace=.22)
    output = root / "paper_draft/fig/fig_cmre_occlusion_selected.png"
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved:", output)
    print("Frozen reconstruction MAE:", result["reconstruction_mae"])
    print("Input embedding agreement:", result["fidelity"])
