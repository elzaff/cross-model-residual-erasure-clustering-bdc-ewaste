"""Apply the authors' two-view DIVIDE training code (official commit c31e2a8) to BDC features.

The architecture, loss, sampler, learning-rate schedule, and feature extraction come
from the official repository. The input views are adapted: fused SigLIP2/PE-Core
versus DINOv3-H+. This is an application of DIVIDE, not a reproduction of its
Scene15 experiment. Scaling, training and K-means fit BDC-main only, then project
external images. K=16 is the BDC choice; SCMax-on-DIVIDE is a separate hybrid.
"""
import hashlib, os, sys, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from compare_aimv2_fusions import ROOT, cmre, fuse  # sets EXP_OUT first
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler
import rq1_metrics as metrics
from common import load_items
from rq_metrics2 import evaluate, inlp_x, partition
from rq_ext_metrics import REFS, ext_scores, load_ext
from rq_scmax import scmax

DIVIDE_CODE_ROOT = Path(os.environ.get("DIVIDE_CODE_ROOT", ROOT / "repos" / "DIVIDE"))
sys.path.insert(0, str(DIVIDE_CODE_ROOT))
from model import DIVIDE  # noqa: E402  (official code, unmodified)
from engine_train import train_one_epoch  # noqa: E402
from dataset_loader import IncompleteMultiviewDataset, IncompleteDatasetSampler  # noqa: E402
import utils as dv_utils  # noqa: E402

OUT = ROOT / "results" / "classic" / os.environ.get("DIVIDE_OUT", "divide_results.csv")
SEEDS = tuple(int(s) for s in os.environ.get("DIVIDE_SEEDS", "0,1,2").split(","))
SCMAX_SEEDS = (3407, 0, 1)
INCLUDE_SCMAX = os.environ.get("DIVIDE_INCLUDE_SCMAX", "1") == "1"
CFG = dict(batch_size=1024, momentum=0.98, temperature=0.5, blr=5.0e-4, weight_decay=0,  # config/Scene15.yaml
           drop_rate=0.2, epochs=200, warmup_epochs=20, start_rectify_epoch=100, n_views=2)  # main_train.py defaults


def divide_embed(views, main, seed):
    """Train the official DIVIDE on D_main rows; return the official extract_feature embedding for every row."""
    data = [StandardScaler().fit(v[main]).transform(v).astype("float32") for v in views]
    args = argparse.Namespace(**CFG, seed=seed, encoder_dim=[[v.shape[1], 1024, 1024, 1024, 128] for v in views])
    args.lr = args.blr * args.batch_size / 256; args.print_this_epoch = False  # main_train.main()
    dv_utils.fix_random_seeds(seed)
    ds = IncompleteMultiviewDataset(2, [d[main] for d in data], np.zeros(main.sum(), int), 0.0)
    dl = torch.utils.data.DataLoader(ds, sampler=IncompleteDatasetSampler(ds, seed=seed), batch_size=args.batch_size,
                                     num_workers=0, pin_memory=True, drop_last=True)
    model = DIVIDE(n_views=2, layer_dims=args.encoder_dim, temperature=args.temperature, n_classes=16,
                   drop_rate=args.drop_rate).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    for ep in range(args.epochs):
        train_one_epoch(model, dl, None, opt, torch.device("cuda"), ep, None, args)
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(data[0]), 1024):
            x = [torch.tensor(d[s:s + 1024]).cuda() for d in data]
            f = model.extract_feature(x, torch.ones(len(x[0]), 2, dtype=torch.bool).cuda())
            out.append(torch.nn.functional.normalize(torch.cat(f, 1), dim=-1).cpu().numpy())
    return np.vstack(out)


def kmeans_all(z, main, k):
    # Match the 10 restarts of scikit-learn's default when DIVIDE was published.
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit(z[main]).predict(z)


def build():
    flip = ROOT / "data" / "embeddings_final_v4" / "emb_dinov3hp_flip.npy"
    assert hashlib.sha256(flip.read_bytes()).hexdigest() == "f4b6f04158f1d3965460e77156669d2a281754b8000ff69299a7a1d756383268", \
        "DINOv3 flip embedding differs from the frozen final-v4 input"
    items = load_items(); main0, style0, lab0, src0, _, shared0 = metrics.setup(items); N = len(items)
    (Xi, li), (Xs, ls) = load_ext("iliev"), load_ext("shubha")
    X = {k: np.vstack([Xi[k], Xs[k][N:]]) for k in Xi}; ni, ns = len(li), len(ls)
    return dict(X=X, main=np.r_[main0, np.zeros(ni + ns, bool)], lab=np.r_[lab0, li, ls],
                style=np.r_[style0, np.full(ni + ns, style0[0])],  # proxy only read on main rows (INLP fit)
                src=np.r_[np.where(main0, "bdc", src0), np.full(ni, "iliev"), np.full(ns, "shubha")],
                shared=np.r_[shared0, np.zeros(ni + ns, bool)])


def score_all(S, y, z):
    lab, src, out = S["lab"], S["src"], {}
    pool = (lab != "") & (lab != "upload") & np.isin(src, ("bdc", "kaan", "bangla", "karan"))
    r = evaluate(y, z, lab, src, pool, S["shared"], np.random.default_rng(0), n_perm=0)
    out["BDC"] = dict(xsrc=r["xsrc"], xtv=r["xtv_all"], leak_cmi=r["leak_cmi"], cramer_v=r["cramer_v"],
                      knn_r1=r["knn_r1"], ami=r["ami_ext"])
    for e in ("iliev", "shubha"):
        out[e] = ext_scores(y, z, lab, src, np.where((lab != "") & (lab != "upload") & np.isin(src, (e,) + REFS))[0], e)
    return out


def main():
    S = build(); X, main = S["X"], S["main"]; sig, pe, dino = X["siglip2"], X["pecoreg"], X["dinov3hp"]
    cm = [cmre(sig, dino, main)[0], cmre(pe, dino, main)[0]]; rows = []
    def add(method, K, seed, y, z):
        for sname, r in score_all(S, y, z).items():
            rows.append(dict(set=sname, method=method, K=K, seed=seed, **r)); print(rows[-1], flush=True)
        pd.DataFrame(rows).round(4).to_csv(OUT, index=False)
    for ref, x in (("v4: CMRE -> Spectral", fuse(*cm, dino)), ("mentah -> Spectral", fuse(sig, pe, dino))):
        y, z = partition(x, main, 16); add(ref, 16, 0, y, z)
    # DIVIDE keeps what the two views share, so the views must both be debiased for the shared part to be the object.
    views = {"mentah": lambda: [fuse(sig, pe), dino], "CMRE": lambda: [fuse(*cm), dino],
             "CMRE-VLM": lambda: cm,  # (a) CMRE(SigLIP2) vs CMRE(PE), no DINO view
             "CMRE+INLPdino": lambda: [fuse(*cm), inlp_x(dino, main, S["style"])]}  # (b) DINO view debiased too
    for tag in os.environ.get("DIVIDE_VIEWS", "mentah,CMRE").split(","):
        ys, vv = {}, views[tag]()
        for seed in SEEDS:
            z = divide_embed(vv, main, seed)
            choices = [(16, "K=16")]
            if INCLUDE_SCMAX:
                picks = [max(scmax(z[main], s), key=lambda L: L["nnc"])["K"] for s in SCMAX_SEEDS]
                K_sc = int(np.median(picks)); print(tag, "seed", seed, "SCMax picks", picks, flush=True)
                choices.append((K_sc, "K=SCMax"))
            for K, how in choices:
                y = kmeans_all(z, main, K); ys.setdefault(how, []).append(y[main])
                add(f"{tag} -> DIVIDE, {how}", K, seed, y, z)
        for how, yy in ys.items():
            print("SEED-ARI", tag, how, round(float(np.mean([ari(yy[0], a) for a in yy[1:]])), 3), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
