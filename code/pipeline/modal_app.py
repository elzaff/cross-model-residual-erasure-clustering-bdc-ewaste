"""Run feature extraction + analysis on Modal (any Modal profile, GPU L4 or A100).

One-off data upload (staged folder with items_base.csv, imgs/, models/):
  MODAL_PROFILE=<profile> modal volume create bdc-max
  MODAL_PROFILE=<profile> modal volume put bdc-max <staged_dir> /data
Run (logs stream live):
  MODAL_PROFILE=<profile> modal run modal_app.py                 # extract + rq_max
  MODAL_PROFILE=<profile> modal run modal_app.py --stage analyze # rq_max only
Fetch results:
  MODAL_PROFILE=<profile> modal volume get bdc-max /out <local_dir>
"""
import os, modal

HERE = os.path.dirname(os.path.abspath(__file__))
vol = modal.Volume.from_name("bdc-max", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.7.1", "transformers>=4.57,<5", "huggingface_hub>=0.34", "scikit-learn", "scikit-image",
                      "pandas", "pillow", "scipy", "sentencepiece", "protobuf", "timm>=1.0.19", "open_clip_torch>=2.32",
                      "imagehash", "pyarrow")
         .run_commands("pip uninstall -y opencv-python opencv-python-headless", "pip install --no-deps opencv-python-headless==4.10.0.84")
         .env({"DATA_ROOT": "/vol/data", "EXP_OUT": "/vol/out", "GIZ_ROOT": "/vol/giz", "HF_HOME": "/vol/hf"})
         .add_local_dir(HERE, "/root/exp", ignore=["out", "out_*", "__pycache__", "*.npy", "*.ipynb"]))
app = modal.App("bdc-max", image=image)

HF = [modal.Secret.from_name("huggingface-secret")]  # authenticated HF downloads (gated DINOv3, avoids 429)

def _step(script, *args):
    import subprocess, sys, threading
    print(f"=== {script} {' '.join(args)}", flush=True)
    done = threading.Event()
    def autocommit():  # survive preemption: persist partial outputs every 5 min
        while not done.wait(300): vol.commit()
    threading.Thread(target=autocommit, daemon=True).start()
    try: subprocess.run([sys.executable, script, *args], check=True, cwd="/root/exp")
    finally: done.set(); vol.commit()

WEIGHTS = ["facebook/dinov3-convnext-large-pretrain-lvd1689m", "facebook/dinov3-vith16plus-pretrain-lvd1689m",
           "facebook/dinov3-vit7b16-pretrain-lvd1689m", "timm/vit_pe_core_gigantic_patch14_448.fb",
           "timm/convnextv2_huge.fcmae_ft_in22k_in1k_384", "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"]

@app.function(cpu=4, memory=16384, volumes={"/vol": vol}, timeout=3 * 3600, secrets=HF)
def prefetch():
    """Download all weights into the volume's HF cache on a cheap CPU container (GPU then loads from cache)."""
    from huggingface_hub import snapshot_download
    for r in WEIGHTS:
        try: snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"]); print("cached", r, flush=True)
        except Exception as e: print("SKIP", r, type(e).__name__, str(e)[:200], flush=True)
    vol.commit()

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=4 * 3600, secrets=HF)
def extract_base(): _step("embed_kg.py")

@app.function(gpu="A100-80GB", cpu=8, memory=65536, volumes={"/vol": vol}, timeout=4 * 3600, secrets=HF)
def extract_extra(): _step("embed_extra.py")  # 7B model needs A100-80GB, bf16

@app.function(cpu=16, memory=65536, volumes={"/vol": vol}, timeout=6 * 3600)
def analyze(): _step("rq_max.py")  # CPU only: clustering/INLP do not need a GPU

@app.function(cpu=8, memory=32768, volumes={"/vol": vol}, timeout=2 * 3600)
def finalize(): _step("final_max.py")  # final partition, CIs, galleries

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=4 * 3600)
def extra2():  # auto style discovery (AdaIN stats) + TURTLE/TEMI baselines
    if not os.path.exists("/vol/out/emb_stylestats.npy"): _step("embed_style.py")
    _step("rq_extra.py")

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=4 * 3600)
def novel(): _step("rq_novel.py")  # CMRE, CCE, TURTLE-SI, cross-style graph, LEACE

@app.function(gpu="L4", cpu=8, memory=49152, volumes={"/vol": vol}, timeout=6 * 3600, secrets=HF)
def novel2():  # foreground-masked embeddings, then CMRE generality / combos / hubness / TURTLE-SCAN-DMoN
    if not os.path.exists("/vol/out/emb_siglip2_fg_flip.npy"):
        try: _step("embed_fg.py")
        except Exception as e: print("FG extraction failed, continuing without it:", e, flush=True)
    _step("rq_novel2.py")

META = ["facebook/webssl-dino300m-full2b-224", "facebook/webssl-dino1b-full2b-224",
        "facebook/metaclip-l14-fullcc2.5b", "facebook/metaclip-h14-fullcc2.5b"]

@app.function(cpu=4, memory=16384, volumes={"/vol": vol}, timeout=2 * 3600, secrets=HF)
def prefetch_meta():  # weights downloaded on CPU, not on a GPU container
    from huggingface_hub import snapshot_download
    for r in META:
        try: snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"]); print("cached", r, flush=True)
        except Exception as e: print("prefetch failed", r, e, flush=True)
    vol.commit()

@app.function(gpu="L4", cpu=8, memory=49152, volumes={"/vol": vol}, timeout=6 * 3600, secrets=HF)
def final2():  # MetaCLIP/Web-DINO data control, fair baselines, K choice, graph sensitivity, then final partition v3
    try: _step("embed_meta.py")
    except Exception as e: print("META extraction failed, continuing without it:", e, flush=True)
    _step("rq_final2.py")
    os.environ["FINAL_TAG"] = "_v3"; _step("final_max.py")

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=4 * 3600)
def scmax_run(): _step("rq_scmax.py")  # SCMax (AAAI-26) label-free K on CMRE / INLP / raw features

@app.function(gpu="L4", cpu=8, memory=49152, volumes={"/vol": vol}, timeout=4 * 3600)
def final3():  # final partition v3 with K = median SCMax K (13), graph sensitivity and other algorithms at that K
    os.environ["K_OVERRIDE"] = "13"; _step("rq_final2.py")
    os.environ["FINAL_TAG"] = "_v3"; _step("final_max.py")

@app.function(cpu=16, memory=65536, volumes={"/vol": vol}, timeout=6 * 3600)
def fix(): _step("rq_fix.py")  # baselines, source probes, CMRE r sweep, graph/K stability, pair diagnostics

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=4 * 3600)
def scmax_seeds():  # 6 extra SCMax seeds on the proposed features only
    os.environ.update(SCMAX_SEEDS="2,3,4,5,6,7", SCMAX_FEATS="CMRE"); _step("rq_scmax.py")

@app.function(cpu=8, memory=32768, volumes={"/vol": vol}, timeout=2 * 3600)
def final4():  # v3 features, K = median SCMax K over 9 seeds (16); v3 outputs untouched
    import json, shutil
    cfg = json.load(open("/vol/out/final_config_v3.json")); cfg["K"] = 16
    json.dump(cfg, open("/vol/out/final_config_v4.json", "w"), indent=1)
    shutil.copy("/vol/out/z_final_v3.npy", "/vol/out/z_final_v4.npy")
    os.environ["FINAL_TAG"] = "_v4"; _step("final_max.py")

# ---- held-out validation (protocol locked in RECAP 3.11): prep on CPU -> embed on L4 -> evaluation
@app.function(cpu=8, memory=32768, volumes={"/vol": vol}, timeout=4 * 3600, secrets=HF + [modal.Secret.from_name("roboflow-secret")])
def prep_new(tag): _step("prep_new.py", tag)

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=3 * 3600, secrets=HF)
def embed_new(tag): _step("embed_new.py", tag)

@app.function(cpu=16, memory=65536, volumes={"/vol": vol}, timeout=3 * 3600)
def rq_iliev(): _step("rq_iliev.py")

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=4 * 3600)
def rq_officehome(): _step("rq_officehome.py")

@app.function(cpu=16, memory=65536, volumes={"/vol": vol}, timeout=3 * 3600)
def fix2(): _step("rq_fix2.py")  # round 7: OOD scores, per-cluster Jaccard, Iliev mapping sensitivity, CMRE+INLP

@app.function(cpu=16, memory=65536, volumes={"/vol": vol}, timeout=3 * 3600)
def paper_assets(): _step("paper_assets.py")  # example sheets, t-SNE, K=16 algorithm table, K sweep, kNN-OOD per class

@app.function(cpu=1, memory=2048, timeout=12 * 3600)
def chain(tag):  # runs remotely so a detached local client can disconnect
    prep_new.remote(tag); embed_new.remote(tag); {"iliev": rq_iliev, "officehome": rq_officehome, "shubha": rq_shubha}[tag].remote()

@app.function(cpu=16, memory=65536, volumes={"/vol": vol}, timeout=3 * 3600)
def rq_shubha(): _step("rq_shubha.py")  # fresh test set for the round-7 choices

@app.function(cpu=16, memory=65536, volumes={"/vol": vol}, timeout=4 * 3600)
def r8(): _step("rq_r8.py")  # round 8: paired bootstrap, per-class, what CMRE removes, WEEE test, alpha, timing, other references, clean galleries

@app.function(cpu=8, memory=16384, volumes={"/vol": vol}, timeout=2 * 3600)
def gallery(): _step("rq_gallery.py")  # face-free cluster galleries + face audit

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=6 * 3600)
def d1(): _step("rq_d1.py")  # TURTLE / TEMI / DECMCV on the v4 features

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=3600, secrets=HF)
def bench():  # embedding throughput on L4: re-embed the Shubha crops under a separate tag (outputs unused)
    import shutil; shutil.copy("/vol/out/shubha_items.csv", "/vol/out/bench_items.csv"); _step("embed_new.py", "bench")

@app.function(cpu=16, memory=65536, volumes={"/vol": vol}, timeout=4 * 3600)
def r8b(): _step("rq_r8b.py")  # label-free reference choice + fast variants without PE-Core-G

@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": vol}, timeout=3600, secrets=HF)
def bench_dino():  # SigLIP2 and DINOv3-H+ alone on L4 (Shubha crops, tag "bench"; outputs unused)
    os.environ["BENCH_DINO_ONLY"] = "1"; _step("embed_new.py", "bench")

@app.function(cpu=16, memory=65536, volumes={"/vol": vol}, timeout=2 * 3600)
def notest(): _step("rq_notest.py")  # v4 refit without the 188 BDC test images

@app.local_entrypoint()
def main(stage: str = "extra+analyze"):
    if stage == "r8": return r8.remote()
    if stage == "notest": return notest.remote()
    if stage == "r8b": return r8b.remote()
    if stage == "bench_dino": return bench_dino.remote()
    if stage == "gallery": return gallery.remote()
    if stage == "d1": return d1.remote()
    if stage == "bench": return bench.remote()
    if stage == "fix": return fix.remote()
    if stage == "scmax_seeds": return scmax_seeds.remote()
    if stage == "final4": return final4.remote()
    if stage in ("iliev", "officehome", "shubha"): return chain.remote(stage)
    if stage == "rq_iliev": return rq_iliev.remote()
    if stage == "fix2": return fix2.remote()
    if stage == "paper": return paper_assets.remote()
    if stage == "novel2": return novel2.remote()
    if stage == "scmax": return scmax_run.remote()
    if stage == "final3": return final3.remote()
    if stage == "final2": prefetch_meta.remote(); return final2.remote()
    if stage == "finalize": return finalize.remote()
    if stage == "novel": return novel.remote()
    if stage == "extra2": return extra2.remote()
    if stage in ("all", "extract"): extract_base.remote()
    if stage in ("all", "extra", "extra+analyze", "prefetch"): prefetch.remote()
    if stage in ("all", "extra", "extra+analyze"): extract_extra.remote()
    if stage in ("all", "analyze", "extra+analyze"): analyze.remote()
