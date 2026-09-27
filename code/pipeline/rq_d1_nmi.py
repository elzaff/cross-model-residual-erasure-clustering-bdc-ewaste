"""External object NMI (Tabel 3 protocol, 1,869 Kaan/Bangladesh/Karan images, five shared classes) for the D1 baselines
TURTLE, TEMI and DECMCV. Reuses the definitions in rq_d1.py (everything before its run loop), 3 seeds, mean reported.
The xsrc of each run is printed as a check against results/modal/r8/d1_baselines.csv.
Writes results/classic/d1_ext_nmi.csv."""
import os
# stack.py expects D_main embeddings plus iliev/, shubha/ and final_v4/ under one folder: a scratch folder of
# hardlinks/junctions to data/embeddings_final_v4, data/ext_eval and results/modal/final_v4 (set before import)
os.environ["EXP_OUT"] = os.environ.get("D1_OUT") or os.path.join(os.environ.get("TEMP", "/tmp"), "claude", "d1_out_v2")
from pathlib import Path

code = Path(__file__).with_name("rq_d1.py").read_text(encoding="utf-8").split("\nCFG = ")[0]
exec(code)  # defines turtle, temi, decmcv, nearest, cs, cp, dino, x4, shared, lab, N, ...

CFG = {"TURTLE": lambda s: turtle([cs, cp, dino], s), "TEMI": lambda s: temi(x4, s), "DECMCV": lambda s: decmcv(s)[0]}
OUT_CSV = Path(ROOT) / "results/classic/d1_ext_nmi.csv"  # saved after every run; finished runs are skipped on restart
rows = pd.read_csv(OUT_CSV).to_dict("records") if OUT_CSV.exists() else []
for name, fn in CFG.items():
    for s in range(3):
        if any(r["method"] == name and r["seed"] == s for r in rows): continue
        y = fn(s)
        rows.append(dict(method=name, seed=s, ext_obj_nmi=nmi(lab[shared], y[:N][shared]), xsrc_BDC=xsrc(y, "BDC")))
        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
out = pd.DataFrame(rows)
print(out.groupby("method", sort=False)[["ext_obj_nmi", "xsrc_BDC"]].mean().round(4).to_string())
