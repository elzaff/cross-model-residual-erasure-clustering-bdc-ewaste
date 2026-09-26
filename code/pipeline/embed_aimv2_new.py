"""Add AIMv2 embeddings for an existing external image set (Iliev or Shubha)."""
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from embed import l2
from embed_kg import CLIPN, OUT, Imgs, collate, norm_t

tag = sys.argv[1]
rows = list(csv.DictReader(open(f"{OUT}/{tag}_items.csv", encoding="utf-8")))
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224", dtype=torch.float16).to("cuda").eval()
features = {"": [], "_flip": []}

with torch.no_grad():
    for images, _ in DataLoader(Imgs(rows), batch_size=32, num_workers=4, collate_fn=collate):
        for suffix, batch in (("", images), ("_flip", images[:, :, ::-1].copy())):
            x = norm_t(batch, 224, *CLIPN)
            features[suffix].append(l2(model(pixel_values=x).last_hidden_state.mean(1)).cpu())

for suffix, chunks in features.items():
    path = f"{OUT}/{tag}/emb_aimv2{suffix}.npy"
    np.save(path, torch.cat(chunks).numpy().astype(np.float16))
    print(path, len(rows), flush=True)
