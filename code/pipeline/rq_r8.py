"""Round 8: analyses on the frozen v4 pipeline (no refit of the final partition, labels only for post-hoc scoring).
  C1 c1_paired.csv     paired bootstrap of delta-xsrc (v4 minus comparator, same resampled crops) on BDC/Iliev/Shubha
  C2 c2_perclass.csv   per-class xsrc for every method and set
  C3 c3_removed.csv    what CMRE removes: Spearman of the top residual PCs with pixel statistics, probes on the removed
                       vs kept part (style group / object / Kaan-vs-Bangladesh source); c3_*_pc1_{low,high}.jpg
  C5 c5_failures.csv   Bangladesh images outside the Kaan-majority cluster of their class; c5_failures.jpg
  C7 c7_weee.csv       is the author WEEE grouping of the 16 clusters tighter / more EU-6-consistent than random merges?
  C8 c8_alpha.csv      Ridge alpha sensitivity of CMRE
  C9 c9_time.csv       CPU seconds per post-embedding step (v4 and light version)
  D4 d4_reference.csv  CMRE with other self-supervised / supervised references (BDC only: those embeddings cover D_main+ext)
Writes OUT/r8/."""
import os, time, numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from stack import *
import warnings; warnings.filterwarnings("ignore")

R8 = f"{OUT}/r8"; os.makedirs(R8, exist_ok=True)
def save(rows, fn): pd.DataFrame(rows).to_csv(f"{R8}/{fn}", index=False)
def sheet(paths, fn, cols=8, T=112):
    S = Image.new("RGB", (cols * T, max(1, (len(paths) + cols - 1) // cols) * T), "white")
    for i, p in enumerate(paths):
        im = Image.open(p); im.draft("RGB", (2 * T, 2 * T)); im = im.convert("RGB"); im.thumbnail((T - 4, T - 4))
        S.paste(im, ((i % cols) * T + 2, (i // cols) * T + 2))
    S.save(fn, quality=85)
paths = np.array([r["path"] for r in items]); side = np.array([max(r["width"], r["height"]) for r in items])
NAMES = ["Aki", "Baterai laptop", "Turntable", "TV CRT", "Baterai ponsel/kamera", "Pemutar CD/radio", "Mesin cuci atas", "TV layar datar",
         "Mesin cuci depan", "Laptop", "Keyboard", "Microwave", "Printer", "Mouse", "Ponsel", "PCB/campuran"]
WEEE = np.array(["baterai", "baterai", "kecil", "layar", "baterai", "kecil", "besar", "layar", "besar", "layar", "ti", "kecil", "ti", "ti", "ti", "komponen"])
G2EU = {"baterai": "baterai", "kecil": "5", "layar": "2", "besar": "4", "ti": "6", "komponen": "komponen"}
EU6 = {1: "Refrigerator Freezer Air-Conditioner Dehumidifier Cooled-Dispenser Cooling-Display",  # copied from rq_iliev.py
       2: "CRT-Monitor CRT-TV Flat-Panel-Monitor Flat-Panel-TV Laptop Tablet",
       3: "Compact-Fluorescent-Lamps Straight-Tube-Fluorescent-Lamp LED-Bulb",
       4: "Washing-Machine Dishwasher Tumble-Dryer Stove Oven Range-Hood Photovoltaic-Panel Server Boiler Street-Lamp Electric-Bicycle Rotary-Mower",
       6: "Smartphone Bar-Phone Computer-Keyboard Computer-Mouse Printer Router Network-Switch HDD SSD USB-Flash-Drive Telephone-Set Calculator Smart-Watch Desktop-PC"}
EU6 = {c: str(k) for k, v in EU6.items() for c in v.split()} | {"Battery": "baterai", "PCB": "komponen"}
tm = lambda: time.perf_counter()

sig, pe, dino = feat("siglip2"), feat("pecoreg"), feat("dinov3hp"); ym4 = v4_main(); print("stack", len(sig), "Iliev", n_i, "Shubha", n_s, flush=True)

# ---------------- C1 + C2 + C9 ----------------
Y, T9 = {}, []
for name, f in methods(sig, pe, dino).items():
    t0 = tm(); x = f(); t1 = tm(); y, z = project(x, ym4 if name == "usulan v4" else None); Y[name] = y
    T9.append(dict(step=f"{name}: fitur (CMRE/INLP/fusi)", sec=t1 - t0)); T9.append(dict(step=f"{name}: PCA-32 + klaster + proyeksi", sec=tm() - t1))
    if name == "usulan v4":
        z4 = z; dz = float(np.abs(z[:N] - np.load(f"{OUT}/z_final_v4.npy")).max()); print("SANITY z vs z_final_v4", round(dz, 6), "OK" if dz < 1e-3 else "!! MISMATCH", flush=True)
    print("C1", name, {d: round(xsrc(y, d), 3) for d in SETS}, flush=True)
save([dict(set=d, method=m, cls=c, n=int((SETS[d][1] == c).sum()), xsrc=v) for d in SETS for m, y in Y.items() for c, v in per_class(y, d).items()], "c2_perclass.csv")
g = np.random.default_rng(0); c1 = []
for d, (idx, _, _) in SETS.items():
    boots = [g.integers(0, len(idx), len(idx)) for _ in range(2000)]
    Bm = {m: np.array([xsrc(y, d, b) for b in boots]) for m, y in Y.items()}
    for m in Y:
        if m == "usulan v4": continue
        dl = Bm["usulan v4"] - Bm[m]
        c1.append(dict(set=d, vs=m, v4=xsrc(Y["usulan v4"], d), other=xsrc(Y[m], d), delta=xsrc(Y["usulan v4"], d) - xsrc(Y[m], d),
                       lo=np.percentile(dl, 2.5), hi=np.percentile(dl, 97.5), p_delta_le0=float((dl <= 0).mean())))
save(c1, "c1_paired.csv"); print(pd.DataFrame(c1).round(3).to_string(index=False), flush=True)
zm = z4[mainX]
t0 = tm(); spectral(zm); T9.append(dict(step="usulan v4: Spectral-kNN K=16 pada D_main", sec=tm() - t0))
t0 = tm(); nnm = NearestNeighbors(n_neighbors=11).fit(zm); dmain = nnm.kneighbors(zm)[0][:, 10]; thr = np.percentile(dmain, 95)
sI = nnm.kneighbors(z4[SETS["Iliev"][0]], 10)[0][:, 9]; T9.append(dict(step="kNN-OOD: fit + skor D_main + skor Iliev", sec=tm() - t0))
save([dict(**r, cpu=os.cpu_count(), n_main=int(main.sum())) for r in T9], "c9_time.csv"); print(pd.DataFrame(T9).round(2).to_string(index=False), flush=True)

# ---------------- C8 alpha sweep ----------------
c8 = []
for a in (0.01, 0.1, 1, 10, 100, 1000):
    y, _ = project(fuse(cmre(sig, dino, a), cmre(pe, dino, a), dino))
    c8.append(dict(alpha=a, **{f"xsrc_{d}": xsrc(y, d) for d in SETS}, purity_bdc=purity_bdc(y), ari_vs_v4=ari(ym4, y[mainX])))
    print("C8", c8[-1], flush=True)
save(c8, "c8_alpha.csv")

# ---------------- C7 WEEE grouping vs random merges ----------------
K = 16; S = np.bincount(ym4, minlength=K); C = np.stack([zm[ym4 == c].mean(0) for c in range(K)]); Dc = pairwise_distances(C)
iu = np.triu_indices(K, 1); W = (S[:, None] * S[None])[iu]
def within(gr): same = (gr[:, None] == gr[None])[iu]; return float((W * Dc[iu] * same).sum() / (W * same).sum())
perm = [g.permutation(WEEE) for _ in range(100000)]
c7 = []
obs = within(WEEE); nl = np.array([within(p) for p in perm]); c7.append(dict(test="jarak centroid intra-grup (makin kecil makin rapat)", obs=obs, null_mean=nl.mean(), p=float((nl <= obs).mean())))
Dm = pairwise_distances(zm); sil = lambda gr: silhouette_score(Dm, gr[ym4], metric="precomputed")
obs = sil(WEEE); nl = np.array([sil(p) for p in perm[:500]]); c7.append(dict(test="silhouette 6 grup WEEE pada citra D_main", obs=obs, null_mean=nl.mean(), p=float((nl >= obs).mean())))
ward = fcluster(linkage(C, "ward"), 6, "maxclust"); obs = ari(WEEE, ward); nl = np.array([ari(p, ward) for p in perm[:20000]])
c7.append(dict(test="ARI(pengelompokan Ward 16 centroid jadi 6, WEEE)", obs=obs, null_mean=nl.mean(), p=float((nl >= obs).mean())))
yI = Y["usulan v4"][SETS["Iliev"][0]]; code = np.array([EU6.get(c, "5") for c in NI.cls.values]); ok = np.isin(code, list(G2EU.values()))
for tag, mk in (("semua", ok), ("in-distribution kNN", ok & (sI <= thr))):
    acc = lambda gr: float((np.array([G2EU[v] for v in gr[yI[mk]]]) == code[mk]).mean())
    obs = acc(WEEE); nl = np.array([acc(p) for p in perm[:20000]])
    c7.append(dict(test=f"Iliev: kategori EU-6 crop = kategori grup klasternya ({tag}, n={int(mk.sum())})", obs=obs, null_mean=nl.mean(), p=float((nl >= obs).mean())))
save(c7, "c7_weee.csv"); print(pd.DataFrame(c7).round(4).to_string(index=False), flush=True)

# ---------------- C5 failure gallery ----------------
yb = Y["usulan v4"]; km = {c: np.bincount(yb[:N][KA[c]]).argmax() for c in SHARED}; c5, fp = [], []
for c in SHARED:
    ii = np.where(bang & (lab == c))[0]; bad = ii[yb[ii] != km[c]]; vc = pd.Series(yb[bad]).value_counts()
    c5.append(dict(cls=c, kaan_cluster=NAMES[km[c]], n=len(ii), n_fail=len(bad), wrong_clusters="; ".join(f"{NAMES[k]}:{v}" for k, v in vc.head(4).items())))
    fp += list(paths[g.permutation(bad)[:8]])
save(c5, "c5_failures.csv"); print(pd.DataFrame(c5).to_string(index=False), flush=True)
sheet(fp, f"{R8}/c5_failures.jpg")  # up to 8 per class in SHARED order (battery, keyboard, mobile, mouse, pcb)

# ---------------- C3 what CMRE removes ----------------
st = {k: np.array([r[k] for r in items]) for k in ("border_white", "border_std", "colorfulness")} | {"log_sisi": np.log(side)}
def probe(x, t, m): return float(cross_val_score(LogisticRegression(max_iter=500, class_weight="balanced"), x[m], t[m], cv=5, scoring="balanced_accuracy").mean())
srcm = shared & np.isin(src, ["kaan", "bangla"]); c3 = []
for bn, v in (("SigLIP2", sig), ("PE-Core-G", pe)):
    cl, U = cmre(v, dino, basis=True); rem, kept, raw = v[:N] @ U.T, cl[:N], v[:N]
    c3.append(dict(backbone=bn, what="energi dibuang (mean ||proj||^2)", value=float((rem[main] ** 2).sum(1).mean())))
    for j in range(5):
        for k, s in st.items(): c3.append(dict(backbone=bn, what=f"spearman PC{j + 1} vs {k}", value=float(spearmanr(rem[main, j], s[main])[0])))
    for tn, t, m in (("gaya 6 grup", style, main), ("objek BDC", lab, cat), ("sumber Kaan vs Bangladesh", src, srcm)):
        for pn, x in (("mentah", raw), ("bagian dibuang (64 dim)", rem), ("bagian disimpan", kept)): c3.append(dict(backbone=bn, what=f"probe {tn} | {pn}", value=probe(x, t, m)))
    o = np.where(main)[0][np.argsort(rem[main, 0])]
    sheet(list(paths[o[:16]]), f"{R8}/c3_{bn}_pc1_low.jpg"); sheet(list(paths[o[-16:]]), f"{R8}/c3_{bn}_pc1_high.jpg")
    print("C3", bn, flush=True)
save(c3, "c3_removed.csv"); print(pd.DataFrame(c3).round(3).to_string(index=False), flush=True)

# ---------------- D4 other references (BDC only) ----------------
REFS = {"dinov3hp": "DINOv3-H+ (v4)", "dinov2l_cls": "DINOv2-L", "dinov3l_cls": "DINOv3-L", "dinov3_cls": "DINOv3-B", "dinov3cnxl": "DINOv3-ConvNeXt-L",
        "dinov3_7b": "DINOv3-7B", "webdino300m": "Web-DINO-300M", "webdino1b": "Web-DINO-1B", "cnxv2h": "ConvNeXt V2-H (supervisi)"}
d4, s, p = [], sig[:N], pe[:N]
for k, nm in REFS.items():
    try: R = nz(emb(k) + emb(k + "_flip")) if os.path.exists(f"{OUT}/emb_{k}_flip.npy") else emb(k)
    except FileNotFoundError: print("D4 skip", k, flush=True); continue
    if len(R) != N: print("D4 skip", k, len(R), flush=True); continue
    for var, x in (("CMRE", lambda: fuse(cmre(s, R, m=main), cmre(p, R, m=main), R)), ("mentah", lambda: fuse(s, p, R)), ("referensi saja", lambda: R)):
        y, _ = project(x(), m=main); d4.append(dict(ref=nm, variant=var, xsrc_BDC=xsrc(y, "BDC"), purity_bdc=purity_bdc(y), obj_nmi=nmi(lab[cat], y[cat])))
    print("D4", nm, [round(r["xsrc_BDC"], 3) for r in d4[-3:]], flush=True); save(d4, "d4_reference.csv")

print("R8 DONE", flush=True)
