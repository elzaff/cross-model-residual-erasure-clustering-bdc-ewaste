"""Test B on Modal: the 9 SCMax runs (3 feature sets x seeds 3407, 0, 1) of rq_refit.py on L4 GPUs in parallel.
Features are built locally (fits on the non-BDC pool only); only pool rows go up, only the K picks come back, and
rq_refit.main(picks_by) does the scoring locally. Run: MODAL_PROFILE=apasijannn modal run code/pipeline/modal_refit.py"""
import io, pathlib, modal, numpy as np

HERE = pathlib.Path(__file__).parent
app = modal.App("bdc-refit-scmax")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.7.1", "numpy", "scipy", "scikit-learn")
         .add_local_dir(HERE / "scmax", "/root/scmax").add_local_file(HERE / "scmax_loop.py", "/root/scmax_loop.py"))


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=2 * 3600)
def pick(blob: bytes, seed: int) -> int:
    import sys; sys.path.insert(0, "/root"); from scmax_loop import scmax
    X = np.load(io.BytesIO(blob)); levels = scmax(X, seed)
    k = max(levels, key=lambda L: L["nnc"])["K"]; print("seed", seed, "levels", [L["K"] for L in levels], "pick", k); return k


@app.local_entrypoint()
def main():
    from rq_refit import main as score, pool_feats
    P, _, _, feats = pool_feats(); jobs, seeds = [], (3407, 0, 1)
    for name, x in feats.items():
        b = io.BytesIO(); np.save(b, x[P].astype(np.float32)); jobs += [(name, b.getvalue(), sd) for sd in seeds]
    ks = list(pick.starmap([(b, sd) for _, b, sd in jobs]))
    picks = {name: [k for (n, _, _), k in zip(jobs, ks) if n == name] for name in feats}
    print("SCMax picks", picks); score(picks)
