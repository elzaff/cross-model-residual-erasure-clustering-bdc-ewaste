"""Multi-backbone, image-only feature extraction (runs on a Kaggle GPU kernel).

Reads items_base.csv (id, set, source, posthoc_label, rel) from DATA_ROOT, fetches the GIZ sample
from the public HF dataset, writes OUT/items.csv (+ pixel style stats) and OUT/emb_<name>.npy.
posthoc_label is never given to any model.
"""
import csv, os, sys, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from embed import style_stats, handcrafted, l2  # shared helpers

DATA = os.environ.get("DATA_ROOT", "/kaggle/input/bdc-max-data")
OUT = os.environ.get("EXP_OUT", "/kaggle/working/out"); os.makedirs(OUT, exist_ok=True)
GIZ = os.environ.get("GIZ_ROOT", "/kaggle/working/giz")
DEV = "cuda"
IMN = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]); CLIPN = ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
GIZ_REV = "e59eca6612b8ed8a5aabe67fb63384f2248681ff"

def fetch_giz(rows):
    os.makedirs(GIZ, exist_ok=True)
    missing = [os.path.basename(r["rel"]) for r in rows if r["source"] == "giz" and not os.path.exists(os.path.join(GIZ, os.path.basename(r["rel"])))]
    if not missing: return  # pre-uploaded copies (anonymous HF downloads hit 429 rate limits)
    from huggingface_hub import hf_hub_download, list_repo_files
    files = {os.path.basename(f): f for f in list_repo_files("GIZ/E-Waste-Database", repo_type="dataset", revision=GIZ_REV)}
    for b in missing:
        os.symlink(hf_hub_download("GIZ/E-Waste-Database", files[b], repo_type="dataset", revision=GIZ_REV), os.path.join(GIZ, b))

def path_of(r): return os.path.join(GIZ, os.path.basename(r["rel"])) if r["source"] == "giz" else os.path.join(DATA, r["rel"])

class Imgs(Dataset):
    def __init__(self, rows, size=384): self.rows, self.size = rows, size
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        im = Image.open(path_of(self.rows[i])); im.draft("RGB", (768, 768)); im = im.convert("RGB")
        return np.asarray(im.resize((self.size, self.size), Image.BICUBIC)), style_stats(im)

def collate(b): return np.stack([x[0] for x in b]), [x[1] for x in b]

def norm_t(arr, size, mean, std):
    x = torch.from_numpy(arr).to(DEV).permute(0, 3, 1, 2).float() / 255
    if x.shape[-1] != size: x = torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    return ((x - torch.tensor(mean, device=DEV)[:, None, None]) / torch.tensor(std, device=DEV)[:, None, None]).half()

def main(limit=0):
    from transformers import AutoModel, AutoProcessor, CLIPModel
    rows = list(csv.DictReader(open(os.path.join(DATA, "items_base.csv"), encoding="utf-8")))
    if limit: rows = rows[:limit]
    fetch_giz(rows)
    # DINOv3-L overflows in fp16 (NaN) and T4 has no bf16, so it runs in fp32
    M = {n: AutoModel.from_pretrained(p, dtype=dt).to(DEV).eval() for n, p, dt in
         [("dinov3", os.path.join(DATA, "models/dinov3-vitb16"), torch.float16),
          ("dinov3l", os.path.join(DATA, "models/dinov3-vitl16"), torch.float32),
          ("dinov2l", "facebook/dinov2-large", torch.float16)]}
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.float16).to(DEV).eval()
    sig = {k: (AutoModel.from_pretrained(r, dtype=torch.float16).to(DEV).eval(), AutoProcessor.from_pretrained(r))
           for k, r in [("siglip2", "google/siglip2-so400m-patch16-naflex"), ("siglip2g", "google/siglip2-giant-opt-patch16-384")]}
    try: aim = AutoModel.from_pretrained("apple/aimv2-large-patch14-224", dtype=torch.float16).to(DEV).eval()
    except Exception as e: print("AIMv2 unavailable:", e); aim = None
    F, styles = {}, []
    def add(k, v): F.setdefault(k, []).append(l2(v).cpu())
    dl = DataLoader(Imgs(rows), batch_size=32, num_workers=4, collate_fn=collate)
    with torch.no_grad():
        for bi, (arr, st) in enumerate(dl):
            styles += st
            for flip in (False, True):
                a = arr[:, :, ::-1].copy() if flip else arr; s = "_flip" if flip else ""
                for name in ("dinov3", "dinov3l"):
                    h = M[name](pixel_values=norm_t(a, 224, *IMN).to(M[name].dtype)); cls, pt = h.pooler_output, h.last_hidden_state[:, 5:]  # skip CLS+4 registers
                    add(f"{name}_cls{s}", cls)
                    if not flip:
                        sim = torch.einsum("bpd,bd->bp", l2(pt), l2(cls)); fg = (sim >= sim.quantile(0.6, dim=1, keepdim=True)).float()[..., None]
                        add(f"{name}_obj", (pt * fg).sum(1) / fg.sum(1)); add(f"{name}_ctx", (pt * (1 - fg)).sum(1) / (1 - fg).sum(1))
                add(f"dinov2l_cls{s}", M["dinov2l"](pixel_values=norm_t(a, 224, *IMN)).pooler_output)
                add(f"clip{s}", clip.visual_projection(clip.vision_model(pixel_values=norm_t(a, 224, *CLIPN)).pooler_output))
                for k, (m, proc) in sig.items():
                    sp = proc(images=list(a), return_tensors="pt").to(DEV); sp["pixel_values"] = sp["pixel_values"].half()
                    o = m.get_image_features(**sp); add(f"{k}{s}", getattr(o, "pooler_output", o))
                if aim is not None:
                    add(f"aimv2{s}", aim(pixel_values=norm_t(a, 224, *CLIPN)).last_hidden_state.mean(1))
            F.setdefault("hand", []).append(torch.from_numpy(handcrafted(np.stack([np.asarray(Image.fromarray(x).resize((224, 224))) for x in arr]))))
            if bi % 20 == 0: print(f"{(bi + 1) * 32}/{len(rows)}", flush=True)
    for k, v in F.items(): np.save(os.path.join(OUT, f"emb_{k}.npy"), torch.cat(v).numpy().astype(np.float16))
    with open(os.path.join(OUT, "items.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "set", "source", "path", "posthoc_label"] + list(styles[0])); w.writeheader()
        for r, st in zip(rows, styles):
            w.writerow({"id": r["id"], "set": r["set"], "source": r["source"], "path": path_of(r), "posthoc_label": r["posthoc_label"], **st})
    print("done", len(rows), sorted(F))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
