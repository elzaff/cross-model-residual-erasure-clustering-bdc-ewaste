"""Stage the FULL Custom Bangladeshi E-Waste dataset (Afrin & Azmi 2025, CC BY 4.0, 2,157 images, 12 classes) for
embed_new.py, to match the BDC12 protocol: 710 in-taxonomy images (5 BDC classes, no dedup) + 1,443 out-of-taxonomy
images (7 non-e-waste classes). Adds 20 already-embedded Bangladesh rows as controls (cosine check).
Writes <stage>/new/bangfull/imgs/*, <stage>/bangfull_items.csv and data/ext_eval/bangfull_meta.csv (in_ours = kept by
our pHash dedup, dup_of_bdc = removed as a near-duplicate of a BDC image).
Then: modal volume put bdc-max <stage>/new /data/new ; put bangfull_items.csv /out/ ; modal run modal_app.py::embed_new --tag bangfull"""
import csv, glob, os, shutil, sys, yaml, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(os.path.dirname(ROOT), "satdat semifinal", "datasets", "04_mendeley", "custom_bangladeshi_ewaste", "source")
STAGE = sys.argv[1]
MAP = {"Battery_Waste": "battery", "Keyboard": "keyboard", "Mobile": "mobile", "Mouse": "mouse", "PCB": "pcb"}

names = yaml.safe_load(open(os.path.join(SRC, "data.yaml")))["names"]
man = pd.read_csv(os.path.join(ROOT, "data", "manifest.csv"), keep_default_na=False)
man = man[man.source == "bangla"]; base = man.path.map(os.path.basename)
keep, dup_bdc = set(base[man.status == "keep"]), set(base[(man.status == "duplicate") & ~man.dup_of.str.contains("custom_bangladeshi")])
os.makedirs(os.path.join(STAGE, "new", "bangfull", "imgs"), exist_ok=True)
rows, meta = [], []
for img in sorted(glob.glob(os.path.join(SRC, "*", "images", "*"))):
    split, fn = img.split(os.sep)[-3], os.path.basename(img)
    lab = os.path.join(os.path.dirname(os.path.dirname(img)), "labels", os.path.splitext(fn)[0] + ".txt")
    ln = [l for l in open(lab) if l.strip()]
    if not ln: continue  # a few label files are empty -> unlabeled image, skipped
    cls = names[int(ln[0].split()[0])]
    i = f"bangfull_{len(rows):04d}"; shutil.copy(img, os.path.join(STAGE, "new", "bangfull", "imgs", i + os.path.splitext(fn)[1]))
    rows.append(dict(id=i, set="new", source="bangfull", rel=f"/vol/data/new/bangfull/imgs/{i}{os.path.splitext(fn)[1]}",
                     posthoc_label=MAP.get(cls, ""), cls=cls, domain="bangfull"))
    meta.append(dict(id=i, cls=cls, split=split, in_ours=fn in keep, dup_of_bdc=fn in dup_bdc))
items = pd.read_csv(os.path.join(ROOT, "data", "embeddings_final_v4", "items.csv"), keep_default_na=False)
for _, r in items[items.source == "bangla"].sample(20, random_state=0).iterrows():  # controls: must re-embed identically
    rows.append(dict(id=str(r.id), set="control", source="control", rel=r.path, posthoc_label=r.posthoc_label, cls=str(r.id), domain="control"))
with open(os.path.join(STAGE, "bangfull_items.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
M = pd.DataFrame(meta); M.to_csv(os.path.join(ROOT, "data", "ext_eval", "bangfull_meta.csv"), index=False)
print(len(rows), "rows;", M.cls.value_counts().to_dict()); print("in-taxonomy", M.cls.isin(MAP).sum(),
      "| ours kept", (M.in_ours & M.cls.isin(MAP)).sum(), "| dup of BDC", (M.dup_of_bdc & M.cls.isin(MAP)).sum())
