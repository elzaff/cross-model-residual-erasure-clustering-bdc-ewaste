"""Pixel-level debiasing: foreground masking before embedding (early masking, no labels, no text).
The object mask comes from DINOv3-H+ patch tokens: per image, PC1 of the patch features at 448 px, sign chosen so the
outer ring of patches is background (DINOv2-style unsupervised foreground). Background pixels are replaced by the
ImageNet mean colour; images whose mask covers <10% or >90% (scenes, piles, close-ups) are kept unmasked.
Writes OUT/emb_{siglip2,dinov3hp}_fg[_flip].npy, OUT/fg_frac.npy and OUT/fg_examples.jpg."""
import csv, os, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from embed_kg import DATA, OUT, Imgs, collate, norm_t, fetch_giz, IMN
from embed import l2

DEV = "cuda"; BF = torch.bfloat16
GRAY = torch.tensor([124, 116, 104], dtype=torch.uint8)  # ImageNet mean colour

def masks(dino, arr):
    """(B,448,448,3) uint8 -> (B,448,448) bool foreground mask and per-image foreground fraction."""
    h = dino(pixel_values=norm_t(arr, 448, *IMN).to(BF)).last_hidden_state[:, 5:].float()  # skip CLS + 4 registers
    B, P, _ = h.shape; g = int(P ** 0.5)
    h = h - h.mean(1, keepdim=True)
    pc1 = torch.stack([torch.linalg.svd(x, full_matrices=False)[2][0] for x in h])  # (B,d)
    s = torch.einsum("bpd,bd->bp", h, pc1).view(B, g, g)
    ring = torch.ones(g, g, dtype=torch.bool, device=s.device); ring[2:-2, 2:-2] = False
    s = s * torch.where(s[:, ring].mean(1) > 0, -1.0, 1.0)[:, None, None]  # border ring -> background
    m = F.max_pool2d((s > 0).float()[:, None], 3, 1, 1)  # dilate one patch so object edges survive
    frac = m.mean((1, 2, 3))
    keep = (frac < 0.1) | (frac > 0.9); m[keep] = 1.0
    return F.interpolate(m, size=arr.shape[1:3], mode="nearest")[:, 0].bool(), frac

def main():
    from transformers import AutoModel, AutoProcessor
    rows = list(csv.DictReader(open(os.path.join(DATA, "items_base.csv"), encoding="utf-8"))); fetch_giz(rows)
    dino = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m", dtype=BF).to(DEV).eval()
    sig = AutoModel.from_pretrained("google/siglip2-so400m-patch16-naflex", dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-naflex")
    Fe, fracs, ex = {}, [], []
    with torch.no_grad():
        for bi, (arr, _) in enumerate(DataLoader(Imgs(rows, size=448), batch_size=32, num_workers=8, collate_fn=collate)):
            m, frac = masks(dino, arr); fracs.append(frac.cpu())
            a0 = torch.from_numpy(arr); a0 = torch.where(m.cpu()[..., None], a0, GRAY).numpy()
            if len(ex) < 24: ex += [np.hstack([arr[i], a0[i]]) for i in range(0, len(arr), 8)]
            for flip in (False, True):
                a = a0[:, :, ::-1].copy() if flip else a0; s = "_flip" if flip else ""
                Fe.setdefault(f"dinov3hp_fg{s}", []).append(l2(dino(pixel_values=norm_t(a, 224, *IMN).to(BF)).pooler_output.float()).cpu())
                sp = proc(images=list(a), return_tensors="pt").to(DEV); sp["pixel_values"] = sp["pixel_values"].half()
                o = sig.get_image_features(**sp); Fe.setdefault(f"siglip2_fg{s}", []).append(l2(getattr(o, "pooler_output", o).float()).cpu())
            if bi % 20 == 0: print(f"{(bi + 1) * 32}/{len(rows)}", flush=True)
    for k, v in Fe.items(): np.save(os.path.join(OUT, f"emb_{k}.npy"), torch.cat(v).numpy().astype(np.float16))
    fr = torch.cat(fracs).numpy(); np.save(os.path.join(OUT, "fg_frac.npy"), fr)
    Image.fromarray(np.vstack([np.hstack(ex[i:i + 4]) for i in range(0, 24, 4)])).resize((1792, 1344)).save(os.path.join(OUT, "fg_examples.jpg"), quality=85)
    print("FG done", sorted(Fe), "| masked share =", round(float(((fr >= 0.1) & (fr <= 0.9)).mean()), 3), flush=True)

if __name__ == "__main__":
    main()
