"""Build deduplicated image manifest: BDC qualifying e-waste (main) + external e-waste sources (aux).

Near-duplicates = pHash Hamming distance <= THR. BDC images are kept first, so any
external image duplicating BDC is dropped (BDC stays the primary dataset).
Labels/folder names are recorded only for post-hoc analysis, never as model input.
"""
import os, csv, numpy as np, imagehash
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ROOT = r"C:\Users\ender\OneDrive - Institut Teknologi Sepuluh Nopember\Non Academic\satdat semifinal\datasets"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.csv")
THR = 6  # verified visually: pairs at d<=6 were identical/resized copies
SOURCES = [  # (name, relative dir, license, url) -- order = keep priority
    ("bdc", r"00_qualifying_electronic\1_Electronic", "BDC 2026 (panitia)", ""),
    ("bdc_test", None, "BDC 2026 (panitia)", ""),  # test images our qualifying model predicted Electronic
    ("giz", r"02_huggingface\giz_ewaste_database", "CC BY 4.0", "https://huggingface.co/datasets/GIZ/E-Waste-Database"),
    ("bangla", r"04_mendeley\custom_bangladeshi_ewaste\filtered_ewaste", "CC BY 4.0", "https://data.mendeley.com/datasets/77383kmdnw/1"),
    ("kaan", r"01_kaggle\kaan_balanced_waste", "CDLA-Permissive-1.0", "https://www.kaggle.com/datasets/kaanerkez/waste-classfication-dataset"),
    ("karan", r"01_kaggle\garbage_karan", "Apache 2.0", "https://www.kaggle.com/datasets/karansolanki01/garbage-classification"),
    ("plusyaml", r"01_kaggle\garbage_plusyaml_ewaste", "CC0", "https://www.kaggle.com/datasets/engrbasit62/garbage-dataset-plusyaml"),
    ("trashnetpp", r"04_mendeley\trashnet_plus_wastevision\filtered_ewaste", "CC BY 4.0", "https://data.mendeley.com/datasets/mr67c82zw7/1"),
]
EXT = (".jpg", ".jpeg", ".png")
BDC_TEST = r"C:\Users\ender\OneDrive - Institut Teknologi Sepuluh Nopember\Non Academic\satdat_asli\BDC2026\test"
# Qualifying-round SigLIP2 classifier output (class 1 = Electronic): a model prediction, not a ground-truth file.
BDC_TEST_PRED = r"C:\Users\ender\OneDrive - Institut Teknologi Sepuluh Nopember\Non Academic\satdat_asli\SATRIA DATA - EKSPERIMEN\experiments\06_siglip2_naflex_final514\submission asli\submission_SD2026040000100.csv"

def list_files(rel):
    if rel is None:  # bdc_test
        with open(BDC_TEST_PRED, encoding="utf-8") as fh:
            return [os.path.join(BDC_TEST, f"{r['id']}.jpg") for r in csv.DictReader(fh) if r["predicted"] == "1"]
    return [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(ROOT, rel)) for f in sorted(fs) if f.lower().endswith(EXT)]

def phash(path):
    try:
        im = Image.open(path)
        im.draft("RGB", (256, 256))  # fast JPEG downscale on decode (GIZ images are huge)
        return imagehash.phash(im.convert("RGB")), im.size
    except Exception:
        return None, None

rows = []
for name, rel, lic, url in SOURCES:
    for p in list_files(rel):
        rows.append(dict(source=name, path=p, folder=os.path.basename(os.path.dirname(p)), license=lic, url=url))
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(phash, [x["path"] for x in rows]))

kept_bits, kept_idx = [], []
for i, (row, (h, size)) in enumerate(zip(rows, res)):
    row["phash"] = str(h) if h is not None else ""
    if h is None:
        row["status"], row["dup_of"] = "unreadable", ""
        continue
    bits = np.packbits(h.hash.flatten())
    if kept_bits:
        d = np.unpackbits(np.bitwise_xor(np.array(kept_bits), bits), axis=1).sum(1)
        j = int(d.argmin())
        if d[j] <= THR:
            row["status"], row["dup_of"] = "duplicate", rows[kept_idx[j]]["path"]
            continue
    row["status"], row["dup_of"] = "keep", ""
    kept_bits.append(bits); kept_idx.append(i)

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["source", "status", "path", "folder", "phash", "dup_of", "license", "url"])
    w.writeheader(); w.writerows(rows)

from collections import Counter
c = Counter((x["source"], x["status"]) for x in rows)
for name, *_ in SOURCES:
    print(f"{name:11s} keep={c[(name,'keep')]:5d} dup={c[(name,'duplicate')]:5d} bad={c[(name,'unreadable')]}")
print("total keep", sum(v for (s, st), v in c.items() if st == "keep"))
