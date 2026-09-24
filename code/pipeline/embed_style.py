"""Label-free style descriptor (Huang & Belongie 2017, AdaIN): channel mean/std of VGG-16 relu1_2, relu2_2,
relu3_3, relu4_3 -- the classic Gram/feature-statistics notion of image *style*, independent of any heuristic proxy.
Writes OUT/emb_stylestats.npy (N x 1920, same row order as items.csv)."""
import csv, os, numpy as np, torch, torchvision
from torch.utils.data import DataLoader
from embed_kg import DATA, OUT, Imgs, collate, norm_t, fetch_giz, IMN

def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "items_base.csv"), encoding="utf-8"))); fetch_giz(rows)
    vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.cuda().eval().half()
    taps = {3, 8, 15, 22}  # relu1_2, relu2_2, relu3_3, relu4_3
    out = []
    with torch.no_grad():
        for bi, (arr, _) in enumerate(DataLoader(Imgs(rows, size=256), batch_size=64, num_workers=8, collate_fn=collate)):
            h, feats = norm_t(arr, 256, *IMN), []
            for i, layer in enumerate(vgg):
                h = layer(h)
                if i in taps: feats += [h.mean((2, 3)), h.std((2, 3))]
                if i == max(taps): break
            out.append(torch.cat(feats, 1).float().cpu())
            if bi % 20 == 0: print(f"{(bi + 1) * 64}/{len(rows)}", flush=True)
    np.save(os.path.join(OUT, "emb_stylestats.npy"), torch.cat(out).numpy().astype(np.float32)); print("done style")

if __name__ == "__main__":
    main()
