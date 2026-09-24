"""Embed a new image set with exactly the v4 backbones and preprocessing (images only):
SigLIP2-So400m as in embed_kg.py (384 px arrays, fp16, HF processor) and DINOv3-H+ / PE-Core-G as in embed_extra.py
(448 px arrays, bf16), each with a horizontal-flip copy. Reads OUT/<tag>_items.csv, writes OUT/<tag>/emb_<key>.npy and
OUT/<tag>/items.csv (+ pixel style stats). Usage: python embed_new.py <tag>"""
import os, sys, csv, time, numpy as np, torch
from torch.utils.data import DataLoader
from embed_kg import OUT, Imgs, collate, norm_t, IMN
from embed import l2

TAG = sys.argv[1]; D = f"{OUT}/{TAG}"; os.makedirs(D, exist_ok=True); DEV = "cuda"
rows = list(csv.DictReader(open(f"{OUT}/{TAG}_items.csv", encoding="utf-8")))

def run(size, fns):
    F, styles, t0 = {}, [], time.perf_counter()
    with torch.no_grad():
        for bi, (arr, st) in enumerate(DataLoader(Imgs(rows, size=size), batch_size=32, num_workers=8, collate_fn=collate)):
            styles += st
            for flip in (False, True):
                a = arr[:, :, ::-1].copy() if flip else arr
                for k, f in fns.items(): F.setdefault(k + ("_flip" if flip else ""), []).append(l2(f(a)).cpu())
            if bi % 20 == 0: print(size, f"{(bi + 1) * 32}/{len(rows)}", flush=True)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f"TIME {'+'.join(fns)}: {dt:.1f} s for {len(rows)} images x2 (flip) = {len(rows) / dt:.1f} img/s", flush=True)
    for k, v in F.items(): np.save(f"{D}/emb_{k}.npy", torch.cat(v).float().numpy().astype(np.float16))
    return styles

from transformers import AutoModel, AutoProcessor
import timm
sm, sp = AutoModel.from_pretrained("google/siglip2-so400m-patch16-naflex", dtype=torch.float16).to(DEV).eval(), AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-naflex")
def sig(a):
    x = sp(images=list(a), return_tensors="pt").to(DEV); x["pixel_values"] = x["pixel_values"].half()
    o = sm.get_image_features(**x); return getattr(o, "pooler_output", o)
styles = run(384, {"siglip2": sig}); del sm; torch.cuda.empty_cache()
if os.environ.get("BENCH_DINO_ONLY"):  # timing of DINOv3-H+ alone (round-8 fast variant); writes nothing else
    dm = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m", dtype=torch.bfloat16).to(DEV).eval()
    run(448, {"dinov3hp": lambda a: dm(pixel_values=norm_t(a, 224, *IMN).to(torch.bfloat16)).pooler_output}); sys.exit(0)

BF = torch.bfloat16
dm = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m", dtype=BF).to(DEV).eval()
pm = timm.create_model("vit_pe_core_gigantic_patch14_448.fb", pretrained=True, num_classes=0).to(DEV).eval().to(BF); c = pm.pretrained_cfg
run(448, {"dinov3hp": lambda a: dm(pixel_values=norm_t(a, 224, *IMN).to(BF)).pooler_output,
          "pecoreg": lambda a: pm(norm_t(a, c["input_size"][-1], c["mean"], c["std"]).to(BF))})

with open(f"{D}/items.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "set", "source", "path", "posthoc_label", "cls", "domain"] + list(styles[0])); w.writeheader()
    for r, st in zip(rows, styles): w.writerow({**{k: r[k] for k in ("id", "set", "source", "posthoc_label", "cls", "domain")}, "path": r["rel"], **st})
print("EMBED DONE", TAG, len(rows), flush=True)
