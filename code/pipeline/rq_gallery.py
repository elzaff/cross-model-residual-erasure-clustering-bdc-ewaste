"""Round 8 G: cluster galleries of the final v4 partition without detected faces (OpenCV Haar frontal face), dark
(mean grey < 50) or small (< 150 px) images, 160 px tiles; faces.csv = per-image face count for the privacy audit.
Writes OUT/r8/gallery_clean/ and OUT/r8/faces.csv."""
import os, numpy as np, pandas as pd, cv2
from PIL import Image
from common import OUT

R8 = f"{OUT}/r8"; os.makedirs(f"{R8}/gallery_clean", exist_ok=True)
def sheet(paths, fn, cols=8, T=160):
    S = Image.new("RGB", (cols * T, max(1, (len(paths) + cols - 1) // cols) * T), "white")
    for i, p in enumerate(paths):
        im = Image.open(p); im.draft("RGB", (2 * T, 2 * T)); im = im.convert("RGB"); im.thumbnail((T - 4, T - 4))
        S.paste(im, ((i % cols) * T + 2, (i // cols) * T + 2))
    S.save(fn, quality=85)
fd = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def check(pth):
    im = Image.open(pth); im.draft("L", (400, 400)); a = np.asarray(im.convert("L"))
    if max(a.shape) > 400: a = cv2.resize(a, None, fx=400 / max(a.shape), fy=400 / max(a.shape))
    return len(fd.detectMultiScale(a, 1.1, 6, minSize=(24, 24))), float(a.mean())
A4 = pd.read_csv(f"{OUT}/final_v4/assignments.csv", dtype={"id": str}); A4 = A4[(A4.set == "main") | ((A4.source == "giz") & A4.ood)].copy()
fb = [check(q) for q in A4.path]; A4["faces"], A4["bright"] = [a for a, _ in fb], [b for _, b in fb]
A4[["id", "source", "cluster", "ood", "faces", "bright", "side"]].to_csv(f"{R8}/faces.csv", index=False)
m4 = A4[A4.set == "main"]; print("FACES main", int((m4.faces > 0).sum()), "per cluster", m4.groupby("cluster").faces.apply(lambda f: int((f > 0).sum())).to_dict(), flush=True)
good = (A4.faces == 0) & (A4.bright >= 50) & (A4.side >= 150)
for c in range(16):
    s_ = A4[(A4.set == "main") & (A4.cluster == c) & good].sort_values("dist")
    sheet(list(s_.path[:16]) + list(s_.iloc[16:].sample(min(8, max(0, len(s_) - 16)), random_state=0).path), f"{R8}/gallery_clean/c{c:02d}.jpg", T=160)
gz = A4[(A4.source == "giz") & good]; sheet(list(gz.sample(min(24, len(gz)), random_state=0).path), f"{R8}/gallery_clean/giz_ood.jpg", T=160)
print("GALLERY DONE", flush=True)
