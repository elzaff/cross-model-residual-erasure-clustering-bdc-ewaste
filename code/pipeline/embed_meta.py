"""Data-controlled pair for the "training paradigm, not data" claim (images only):
  Web-DINO 300M / 1B  (Fan et al., 2025) -- DINOv2 SSL trained on MetaCLIP web data, no language
  MetaCLIP L/14, H/14 (Xu et al., 2024)  -- CLIP trained on MetaCLIP-curated data (fullcc2.5b)
Same item order as embed_kg.py; writes OUT/emb_<key>.npy and _flip. Weights are prefetched on CPU (modal_app.prefetch_meta)."""
import csv, os, numpy as np, torch
from torch.utils.data import DataLoader
from embed_kg import DATA, OUT, Imgs, collate, norm_t, fetch_giz, IMN, CLIPN
from embed import l2

DEV = "cuda"; BF = torch.bfloat16
SPECS = [("webdino300m", "facebook/webssl-dino300m-full2b-224", "dino"), ("webdino1b", "facebook/webssl-dino1b-full2b-224", "dino"),
         ("metaclipl", "facebook/metaclip-l14-fullcc2.5b", "clip"), ("metacliph", "facebook/metaclip-h14-fullcc2.5b", "clip")]

def load():
    from transformers import Dinov2Model, CLIPModel
    M = {}
    for key, rid, kind in SPECS:
        if os.path.exists(os.path.join(OUT, f"emb_{key}_flip.npy")): continue
        try:
            if kind == "dino":
                m = Dinov2Model.from_pretrained(rid, dtype=BF).to(DEV).eval()
                M[key] = (lambda x, m=m: m(pixel_values=x).pooler_output, IMN)
            else:
                m = CLIPModel.from_pretrained(rid, dtype=BF).to(DEV).eval()
                M[key] = (lambda x, m=m: m.visual_projection(m.vision_model(pixel_values=x).pooler_output), CLIPN)
            print("loaded", key, flush=True)
        except Exception as e: print("SKIP", key, type(e).__name__, str(e)[:200], flush=True)
    return M

def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "items_base.csv"), encoding="utf-8"))); fetch_giz(rows)
    M = load(); F = {}
    if not M: print("META nothing to extract", flush=True); return
    with torch.no_grad():
        for bi, (arr, _) in enumerate(DataLoader(Imgs(rows, size=224), batch_size=64, num_workers=8, collate_fn=collate)):
            for flip in (False, True):
                a = arr[:, :, ::-1].copy() if flip else arr
                for k, (f, (mean, std)) in M.items():
                    F.setdefault(k + ("_flip" if flip else ""), []).append(l2(f(norm_t(a, 224, mean, std).to(BF)).float()).cpu())
            if bi % 20 == 0: print(f"{(bi + 1) * 64}/{len(rows)}", flush=True)
    for k, v in F.items(): np.save(os.path.join(OUT, f"emb_{k}.npy"), torch.cat(v).numpy().astype(np.float16))
    print("META done", sorted(F), flush=True)

if __name__ == "__main__":
    main()
