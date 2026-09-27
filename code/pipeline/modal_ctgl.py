"""Run the CTGL port (ctgl.py) on Modal CPU for BDC D_main; views are built locally (rq_ctgl.py rules), only the
D_main rows go up, only labels come back -> results/classic/ctgl_labels_{tag}.npy, then `python rq_ctgl.py` scores.
Run: MODAL_PROFILE=apasijannn modal run code/pipeline/modal_ctgl.py"""
import io, pathlib, modal, numpy as np

HERE = pathlib.Path(__file__).parent
app = modal.App("bdc-ctgl")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "scipy", "scikit-learn")
         .add_local_file(HERE / "ctgl.py", "/root/ctgl.py"))


@app.function(image=image, cpu=8, memory=32768, timeout=3 * 3600)
def run(blob: bytes, k: int):
    import sys, time; sys.path.insert(0, "/root"); from ctgl import CTGL
    z = np.load(io.BytesIO(blob)); t = time.time()
    y = CTGL([z[f"v{i}"].T.astype(float) for i in range(len(z.files))], k, k=15, alpha=0.5, lam=0.5, pho=1, maxIter=5)[0]
    print("CTGL done", round(time.time() - t), "s"); return y


@app.local_entrypoint()
def main(views: str = "CMRE,mentah"):
    from rq_ctgl import K, ROOT, build, cmre
    S = build(); X, m = S["X"], S["main"]; sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
    V = {"mentah": [sig, pe, dino], "CMRE": [cmre(sig, dino, m)[0], cmre(pe, dino, m)[0], dino]}
    tags = views.split(","); blobs = []
    for t in tags:
        b = io.BytesIO(); np.savez(b, **{f"v{i}": v[m].astype(np.float32) for i, v in enumerate(V[t])}); blobs.append(b.getvalue())
    for t, y in zip(tags, run.starmap([(b, K) for b in blobs])):
        np.save(ROOT / "results" / "classic" / f"ctgl_labels_{t}.npy", y); print(t, "labels saved", np.bincount(y))
