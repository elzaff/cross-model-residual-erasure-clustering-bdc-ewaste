"""Run rq_d1_nmi.py (TURTLE/TEMI/DECMCV) and rq_divide_nmi.py (DIVIDE) on Modal, one L4 each, results on the volume.
The container recreates the local layout under /proj (code, repos/DIVIDE, data/embeddings_final_v4 with iliev/,
shubha/, final_v4/) from the bdc-max volume; results/classic points at /vol/nmi_out so every saved row survives.
Run: MODAL_PROFILE=riset-kp modal run --detach modal_nmi.py
Fetch: MODAL_PROFILE=riset-kp modal volume get bdc-max /nmi_out <local_dir>"""
from pathlib import Path
import sys
import modal

sys.path.insert(0, "/root/exp")  # inside the container modal_app.py lives in /root/exp (added by its own image)
from modal_app import image, vol

HERE = Path(__file__).resolve().parent
# local paths only exist on the laptop; inside the container this file sits in /root and the dirs are already mounted
DIVIDE_DIR = HERE.parents[1] / "repos" / "DIVIDE" if modal.is_local() else Path("/proj/repos/DIVIDE")
img = (image.add_local_dir(HERE, "/proj/code/pipeline", ignore=["out", "out_*", "__pycache__", "*.npy", "*.ipynb"])
       .add_local_dir(DIVIDE_DIR, "/proj/repos/DIVIDE", ignore=[".ruff_cache", "__pycache__"]))
app = modal.App("bdc-nmi-fill", image=img)


def run(script):
    import os, subprocess
    P, V = Path("/proj"), Path("/vol")
    E = P / "data" / "embeddings_final_v4"; E.mkdir(parents=True, exist_ok=True)
    links = {E / f.name: f for f in (V / "out").iterdir()}
    links |= {E / "iliev": V / "ext_eval/iliev", E / "shubha": V / "ext_eval/shubha", E / "final_v4": V / "final_v4",
              P / "data/ext_eval": V / "ext_eval", P / "results/modal/final_v4": V / "final_v4",
              P / "results/classic": V / "nmi_out"}
    (V / "nmi_out").mkdir(exist_ok=True)
    for link, target in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists(): link.symlink_to(target)
    env = {**os.environ, "D1_OUT": str(E), "DIVIDE_CODE_ROOT": str(P / "repos/DIVIDE")}
    try:
        subprocess.run(["python", "-W", "ignore", script], cwd=P / "code/pipeline", env=env, check=True)
    finally:
        vol.commit()


@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=3 * 3600)
def d1():
    run("rq_d1_nmi.py")


@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=3 * 3600)
def divide():
    import subprocess
    subprocess.run(["pip", "install", "-q", "munkres"], check=True)  # imported by the official DIVIDE utils.py
    run("rq_divide_nmi.py")


@app.local_entrypoint()
def main():
    calls = [d1.spawn(), divide.spawn()]
    for c in calls: c.get()
    print("done: results in volume bdc-max /nmi_out")
