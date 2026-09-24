"""Latest large backbones filling a (training paradigm x architecture) grid:
  SSL-CNN   DINOv3 ConvNeXt-L          SSL-ViT   DINOv3 ViT-H+/16, DINOv3 ViT-7B/16
  VLM-CNN   CLIP ConvNeXt-XXL (LAION)  VLM-ViT   Perception Encoder PE-Core-G/14
  SUP-CNN   ConvNeXt V2-H (FCMAE -> IN-22k)
Image towers only. Same item order as embed_kg.py; writes OUT/emb_<key>.npy and _flip. A model that fails to load
(e.g. gated access not granted) is skipped and reported, the others still run."""
import csv, os, numpy as np, torch
from torch.utils.data import DataLoader
from embed_kg import DATA, OUT, Imgs, collate, norm_t, fetch_giz, IMN, CLIPN
from embed import l2

DEV = "cuda"; BF = torch.bfloat16

def load():
    from transformers import AutoModel
    import timm, open_clip
    M = {}
    def hf(key, rid, size):
        m = AutoModel.from_pretrained(rid, dtype=BF).to(DEV).eval()
        M[key] = (lambda x, m=m: m(pixel_values=x).pooler_output, IMN, size)
    def tm(key, name):
        m = timm.create_model(name, pretrained=True, num_classes=0).to(DEV).eval().to(BF); c = m.pretrained_cfg
        M[key] = (lambda x, m=m: m(x), (c["mean"], c["std"]), c["input_size"][-1])
    def oc(key, arch, tag):
        m = open_clip.create_model(arch, pretrained=tag).to(DEV).eval().to(BF); v = m.visual
        M[key] = (lambda x, v=v: v(x), CLIPN, v.image_size[0] if isinstance(v.image_size, (tuple, list)) else v.image_size)
    specs = [("dinov3cnxl", hf, ("facebook/dinov3-convnext-large-pretrain-lvd1689m", 224)),
             ("dinov3hp", hf, ("facebook/dinov3-vith16plus-pretrain-lvd1689m", 224)),
             ("dinov3_7b", hf, ("facebook/dinov3-vit7b16-pretrain-lvd1689m", 224)),
             ("pecoreg", tm, ("vit_pe_core_gigantic_patch14_448.fb",)),
             ("cnxv2h", tm, ("convnextv2_huge.fcmae_ft_in22k_in1k_384",)),
             ("clipcnxxxl", oc, ("convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"))]
    for key, fn, args in specs:
        try: fn(key, *args); print("loaded", key, flush=True)
        except Exception as e: print("SKIP", key, type(e).__name__, str(e)[:200], flush=True)
    return M

def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "items_base.csv"), encoding="utf-8"))); fetch_giz(rows)
    M = load(); F = {}
    with torch.no_grad():
        for bi, (arr, _) in enumerate(DataLoader(Imgs(rows, size=448), batch_size=32, num_workers=8, collate_fn=collate)):
            for flip in (False, True):
                a = arr[:, :, ::-1].copy() if flip else arr
                for k, (f, (mean, std), size) in M.items():
                    F.setdefault(k + ("_flip" if flip else ""), []).append(l2(f(norm_t(a, size, mean, std).to(BF))).cpu())
            if bi % 20 == 0: print(f"{(bi + 1) * 32}/{len(rows)}", flush=True)
    for k, v in F.items(): np.save(os.path.join(OUT, f"emb_{k}.npy"), torch.cat(v).numpy().astype(np.float16))
    print("done", sorted(F))

if __name__ == "__main__":
    main()
