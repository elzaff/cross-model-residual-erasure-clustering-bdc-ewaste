"""Prepare the held-out validation sets (protocol locked in RECAP section 3.11 before any download). CPU only.
  iliev       Roboflow e-waste-dataset-r0ojc (Iliev, CC BY 4.0) via the Roboflow REST export (key from Modal secret, never
              printed): images with exactly one annotation, bbox crop with shortest side >= 64 px, <= 30 per class (seed 0),
              pHash dedup (Hamming <= 6) against every image already in items.csv and among the kept crops
  officehome  flwrlabs/office-home (HF parquet): 20 electrical/electronic classes, domains Product and Real World
Writes DATA/new/<tag>/imgs/*.jpg and OUT/<tag>_items.csv (id, set, source, rel, posthoc_label, cls, domain).
Usage: python prep_new.py iliev|officehome"""
import os, sys, io, csv, json, glob, time, random, zipfile, urllib.request
import numpy as np, imagehash
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

DATA, OUT = os.environ.get("DATA_ROOT", "/vol/data"), os.environ.get("EXP_OUT", "/vol/out")
TAG = sys.argv[1]; ROOT = f"{DATA}/new/{TAG}"; IMG = f"{ROOT}/imgs"; os.makedirs(IMG, exist_ok=True)
PER_CLASS, MIN_SIDE, HAM = 30, 64, 6
ILIEV_MAP = {"Battery": "battery", "Smartphone": "mobile", "Bar-Phone": "mobile", "Computer-Keyboard": "keyboard",
             "Computer-Mouse": "mouse", "PCB": "pcb", "Microwave": "microwave", "Printer": "printer",
             "Washing-Machine": "washing_machine", "CRT-TV": "television", "Flat-Panel-TV": "television", "Music-Player": "player"}
OH_CLASSES = ["Alarm_Clock", "Batteries", "Calculator", "Computer", "Desk_Lamp", "Drill", "Fan", "Kettle", "Keyboard", "Laptop",
              "Monitor", "Mouse", "Oven", "Printer", "Radio", "Refrigerator", "Speaker", "TV", "Telephone", "Webcam"]

def write_rows(rows):
    with open(f"{OUT}/{TAG}_items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "set", "source", "rel", "posthoc_label", "cls", "domain"]); w.writeheader(); w.writerows(rows)
    print(TAG, "rows", len(rows), flush=True)

def api(url):
    for _ in range(60):
        try: return json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as e: print("api retry", type(e).__name__, flush=True); time.sleep(10)
    raise RuntimeError("Roboflow API unreachable")

SHUBHA_MAP = {"Mouse": "mouse", "Keyboard": "keyboard", "Mobile": "mobile", "PCB": "pcb", "Computer": "EXCLUDE", "Electronics": "EXCLUDE"}
RF = {"iliev": ("electronic-waste-detection", "e-waste-dataset-r0ojc", 30, ILIEV_MAP),
      "shubha": ("shubha-to6ii", "e-waste-1sn3k", 60, SHUBHA_MAP)}  # shubha: fresh test set, protocol RECAP 3.11

def roboflow():
    ws, pj, PER_CLASS, CMAP = RF[TAG]; key = os.environ["ROBOFLOW_API_KEY"]
    raw = f"{ROOT}/raw"
    if not glob.glob(f"{raw}/**/_annotations.coco.json", recursive=True):
        info = api(f"https://api.roboflow.com/{ws}/{pj}?api_key={key}")
        ver = max(int(v["id"].rsplit("/", 1)[-1]) for v in info["versions"]); print("version", ver, flush=True)
        while True:  # the first call starts the export; poll until the signed link exists
            r = api(f"https://api.roboflow.com/{ws}/{pj}/{ver}/coco?api_key={key}")
            if r.get("export", {}).get("link"): break
            print("export in progress", r.get("progress"), flush=True); time.sleep(15)
        zp = f"{ROOT}/raw.zip"; urllib.request.urlretrieve(r["export"]["link"], zp)
        zipfile.ZipFile(zp).extractall(raw); os.remove(zp)
    cand = {}
    for ann in glob.glob(f"{raw}/**/_annotations.coco.json", recursive=True):
        J = json.load(open(ann)); cats = {c["id"]: c["name"] for c in J["categories"]}
        per = {}
        for a in J["annotations"]: per.setdefault(a["image_id"], []).append(a)
        for im in J["images"]:
            a = per.get(im["id"], [])
            if len(a) == 1 and min(a[0]["bbox"][2:]) >= MIN_SIDE:
                cand.setdefault(cats[a[0]["category_id"]], []).append((os.path.join(os.path.dirname(ann), im["file_name"]), a[0]["bbox"]))
    print("classes", len(cand), "candidates", sum(map(len, cand.values())), flush=True)
    items = list(csv.DictReader(open(f"{OUT}/items.csv", encoding="utf-8")))
    if TAG != "iliev" and os.path.exists(f"{OUT}/iliev_items.csv"):  # later test sets are also deduplicated against Iliev
        items += [dict(path=r["rel"]) for r in csv.DictReader(open(f"{OUT}/iliev_items.csv", encoding="utf-8")) if r["set"] == "new"]
    def ph(p):
        try: return imagehash.phash(Image.open(p).convert("RGB"))
        except Exception: return None
    with ThreadPoolExecutor(16) as ex: S = np.array([h.hash.ravel() for h in ex.map(ph, [r["path"] for r in items]) if h is not None])
    print("existing hashes", len(S), flush=True)
    kept_h = []
    def dup(h):
        v = h.hash.ravel()
        return (S != v).sum(1).min() <= HAM or (len(kept_h) and (np.array(kept_h) != v).sum(1).min() <= HAM)
    rows, rng = [], random.Random(0)
    for c in sorted(cand):
        lst = cand[c][:]; rng.shuffle(lst); kept = 0
        for p, (x, y, w, h) in lst:
            if kept >= PER_CLASS: break
            full = Image.open(p).convert("RGB"); crop = full.crop((int(x), int(y), int(x + w), int(y + h)))
            hs = [imagehash.phash(full), imagehash.phash(crop)]
            if any(dup(hh) for hh in hs): continue
            kept_h.append(hs[1].hash.ravel()); kept += 1
            fn = f"{IMG}/{c}_{kept:03d}.jpg"; crop.save(fn, quality=95)
            rows.append(dict(id=f"{TAG}_{c}_{kept:03d}", set="new", source=TAG, rel=fn, posthoc_label=CMAP.get(c, ""), cls=c, domain=TAG))
        print(c, kept, "/", len(lst), flush=True)
    # 20 D_main images re-embedded as a control: their new embeddings must match the stored ones (same preprocessing)
    for i in random.Random(1).sample([i for i, r in enumerate(items) if r.get("set") == "main"], 20):
        rows.append(dict(id=items[i]["id"], set="control", source="control", rel=items[i]["path"], posthoc_label=items[i]["posthoc_label"], cls=str(i), domain="control"))
    write_rows(rows)

def officehome():
    import pandas as pd
    from huggingface_hub import snapshot_download
    d = snapshot_download("flwrlabs/office-home", repo_type="dataset", allow_patterns=["*.parquet", "README.md"])
    names = [l.split(": ", 1)[1].strip() for l in open(f"{d}/README.md") if l.strip().startswith("'") and "': " in l]
    rows = []
    for f in sorted(glob.glob(f"{d}/**/*.parquet", recursive=True)):
        df = pd.read_parquet(f)
        for i, r in df.iterrows():
            dom, c = str(r["domain"]), names[int(r["label"])]
            if c not in OH_CLASSES or not any(k in dom.lower() for k in ("product", "real")): continue
            dm = "product" if "product" in dom.lower() else "realworld"; n = len(rows)
            fn = f"{IMG}/{dm}_{c}_{n:05d}.jpg"; Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(fn, quality=95)
            rows.append(dict(id=f"oh_{n:05d}", set="new", source=dm, rel=fn, posthoc_label=c, cls=c, domain=dm))
    print({dm: sum(r["domain"] == dm for r in rows) for dm in ("product", "realworld")}, flush=True)
    write_rows(rows)

officehome() if TAG == "officehome" else roboflow()
