"""Extract image-only representations for D_main (BDC) and the external post-hoc sets.

Outputs (exp/out/):
  items.csv            one row per image: id, set, source, path, posthoc_label, style stats
  emb_<name>.npy       float16 arrays aligned with items.csv row order

Nothing textual is fed to any model. `posthoc_label` (BDC filename class / external folder
class) is written for evaluation after clustering only.
"""
import csv, os, re, random, sys, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # project root
OUT = os.environ.get("EXP_OUT", os.path.join(ROOT, "results", "pilot_local")); os.makedirs(OUT, exist_ok=True)
MANIFEST = os.path.join(ROOT, "data", "manifest.csv")
GIZ_SAMPLE = 1000
DEV = "cuda"

def norm_label(s):
    s = s.lower().replace(" ", "_")
    return {"battery_waste": "battery"}.get(s, s)

def build_items():
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["status"] == "keep"]
    giz = sorted(r["path"] for r in rows if r["source"] == "giz")
    random.Random(0).shuffle(giz); giz = set(giz[:GIZ_SAMPLE])
    items = []
    for r in rows:
        src, p = r["source"], r["path"]
        if src in ("plusyaml", "trashnetpp") or (src == "giz" and p not in giz):
            continue
        lab = ""
        if src == "bdc":
            m = re.match(r"([A-Za-z_]+?)_\d+\.", os.path.basename(p)); lab = norm_label(m.group(1)) if m else ""
        elif src in ("kaan", "karan"):
            lab = norm_label(r["folder"])
        elif src == "bangla":  # YOLO label file next to images/: first token = class id
            names = ["battery", "keyboard", "light_bulb", "mobile", "mouse", "pcb"]
            lf = os.path.splitext(p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))[0] + ".txt"
            try: lab = names[int(open(lf).read().split()[0])]
            except Exception: lab = ""
        items.append(dict(id=len(items), set="main" if src in ("bdc", "bdc_test") else "ext", source=src, path=p, posthoc_label=lab))
    return items

def style_stats(im):
    """Image-derived acquisition-style cues (no labels): size, border whiteness/uniformity, colourfulness."""
    w, h = im.size
    a = np.asarray(im.resize((128, 128)), dtype=np.float32)
    b = np.concatenate([a[:10].reshape(-1, 3), a[-10:].reshape(-1, 3), a[:, :10].reshape(-1, 3), a[:, -10:].reshape(-1, 3)])
    g = b.mean(1)
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return dict(width=w, height=h, border_white=float((b.min(1) > 225).mean()), border_std=float(g.std()),
                colorfulness=float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())))

class Imgs(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        im = Image.open(self.items[i]["path"])
        im.draft("RGB", (640, 640))
        im = im.convert("RGB")
        st = style_stats(im)
        return np.asarray(im.resize((224, 224), Image.BICUBIC)), st  # whole image, no crop

def collate(b): return np.stack([x[0] for x in b]), [x[1] for x in b]

def to_tensor(arr, mean, std):
    x = torch.from_numpy(arr).to(DEV).permute(0, 3, 1, 2).float() / 255
    return (x - torch.tensor(mean, device=DEV)[:, None, None]) / torch.tensor(std, device=DEV)[:, None, None]

def l2(x): return torch.nn.functional.normalize(x.float(), dim=-1)

def handcrafted(arr):
    """Classical baseline: HSV colour histogram + LBP texture + HOG shape."""
    from skimage.feature import local_binary_pattern, hog
    from skimage.color import rgb2hsv, rgb2gray
    out = []
    for a in arr:
        ch = np.histogramdd(rgb2hsv(a).reshape(-1, 3), bins=(8, 4, 4), range=((0, 1),) * 3)[0].ravel(); ch /= ch.sum()
        g = (rgb2gray(a) * 255).astype(np.uint8)
        lbp = np.histogram(local_binary_pattern(g, 8, 1, "uniform"), bins=10, range=(0, 10))[0].astype(float); lbp /= lbp.sum()
        hg = hog(g, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2))
        out.append(np.concatenate([ch, lbp, hg / (np.linalg.norm(hg) + 1e-8)]))
    return np.array(out, dtype=np.float32)

def run(items):
    from transformers import AutoModel, CLIPModel, AutoProcessor
    IMN = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]); CLIPN = ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.float16)
    models = {
        "dinov3": AutoModel.from_pretrained(os.path.expanduser("~/models/dinov3-vitb16"), dtype=torch.float16),
        "dinov2": AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float16),
        "clip": clip.vision_model, "clip_proj": clip.visual_projection,  # image tower only
        "siglip2": AutoModel.from_pretrained("google/siglip2-so400m-patch16-naflex", dtype=torch.float16),
    }
    sig_proc = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-naflex")
    for m in models.values(): m.to(DEV).eval()
    feats = {k: [] for k in ["dinov3_cls", "dinov3_cls_flip", "dinov3_obj", "dinov3_ctx", "dinov3_patchmean",
                             "dinov2_cls", "dinov2_cls_flip", "clip", "clip_flip", "siglip2", "siglip2_flip", "hand"]}
    styles = []
    dl = DataLoader(Imgs(items), batch_size=32, num_workers=6, collate_fn=collate)
    with torch.no_grad():
        for bi, (arr, st) in enumerate(dl):
            styles += st
            for flip in (False, True):
                a = arr[:, :, ::-1].copy() if flip else arr
                sfx = "_flip" if flip else ""
                h = models["dinov3"](pixel_values=to_tensor(a, *IMN).half())
                cls, pt = h.pooler_output, h.last_hidden_state[:, 1 + 4:]  # skip CLS + 4 register tokens
                feats["dinov3_cls" + sfx].append(l2(cls).cpu())
                if not flip:
                    sim = torch.einsum("bpd,bd->bp", l2(pt), l2(cls))  # patch-to-CLS similarity as saliency proxy
                    fg = (sim >= sim.quantile(0.6, dim=1, keepdim=True)).float()[..., None]
                    feats["dinov3_obj"].append(l2((pt * fg).sum(1) / fg.sum(1)).cpu())
                    feats["dinov3_ctx"].append(l2((pt * (1 - fg)).sum(1) / (1 - fg).sum(1)).cpu())
                    feats["dinov3_patchmean"].append(l2(pt.mean(1)).cpu())
                feats["dinov2_cls" + sfx].append(l2(models["dinov2"](pixel_values=to_tensor(a, *IMN).half()).pooler_output).cpu())
                c = models["clip"](pixel_values=to_tensor(a, *CLIPN).half()).pooler_output
                feats["clip" + sfx].append(l2(models["clip_proj"](c)).cpu())
                sp = sig_proc(images=list(a), return_tensors="pt").to(DEV)
                sp["pixel_values"] = sp["pixel_values"].half()
                sg = models["siglip2"].get_image_features(**sp)
                feats["siglip2" + sfx].append(l2(getattr(sg, "pooler_output", sg)).cpu())
            feats["hand"].append(torch.from_numpy(handcrafted(arr)))
            if bi % 20 == 0: print(f"{(bi + 1) * 32}/{len(items)}", flush=True)
    for k, v in feats.items():
        np.save(os.path.join(OUT, f"emb_{k}.npy"), torch.cat(v).numpy().astype(np.float16))
    return styles

if __name__ == "__main__":
    items = build_items()
    if len(sys.argv) > 1: items = items[: int(sys.argv[1])]  # smoke test
    styles = run(items)
    with open(os.path.join(OUT, "items.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0]) + list(styles[0])); w.writeheader()
        for it, st in zip(items, styles): w.writerow({**it, **st})
    print("done", len(items))
