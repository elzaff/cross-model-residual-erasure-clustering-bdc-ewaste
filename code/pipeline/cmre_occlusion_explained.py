"""Annotated occlusion figure from results/classic/cmre_occlusion.csv (computed on Modal by cmre_focus_modal.py).
Shows one image where CMRE shifts sensitivity toward the object and one where it does not, each map on its own colour
scale, plus the 10-image summary, so the example is not read as the average behaviour.
Writes paper_draft/fig/fig_cmre_occlusion_explained.png."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
occ = pd.read_csv(ROOT / "results/classic/cmre_occlusion.csv")
scr = pd.read_csv(ROOT / "results/classic/cmre_occlusion_screen.csv")
items = pd.read_csv(ROOT / "data/embeddings_final_v4/items.csv")
man = pd.read_csv(ROOT / "data/manifest.csv"); man = man[man.status.eq("keep") & man.source.eq("bdc")]
bdc = list(items[items.source.eq("bdc")].index)
BOX = {2671: (.05, .00, .74, 1.0), 2920: (.12, .03, .88, .98)}  # same boxes as cmre_focus_modal.py
SHOW = [(2671, "TV CRT:\nbergeser ke objek"), (2920, "Mesin cuci tabung ganda:\ntidak bergeser")]


def grid(i, method):
    d = occ[(occ.id == i) & (occ.method == method)]
    g = np.zeros((10, 10)); g[d.row, d.col] = d.cosine_change; return g


def photo(i, s=448):
    return Image.open(man.iloc[bdc.index(i)].path).convert("RGB").resize((s, s))


plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9})
fig = plt.figure(figsize=(7.4, 6.6))
gs = fig.add_gridspec(len(SHOW) + 1, 3, height_ratios=[1] * len(SHOW) + [.55], hspace=.3, wspace=.34,
                      left=.09, right=.9, top=.92, bottom=.01)
for r, (i, title) in enumerate(SHOW):
    s = scr.set_index("id").loc[i]; img = photo(i)
    ax = fig.add_subplot(gs[r, 0]); ax.imshow(img)
    x0, y0, x1, y1 = BOX[i]
    ax.add_patch(Rectangle((x0 * 448, y0 * 448), (x1 - x0) * 448, (y1 - y0) * 448, fill=False, ec="#00bcd4", lw=1.6, ls="--"))
    ax.set_ylabel(title, fontsize=9, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("kotak putus-putus = area objek", fontsize=7.5)
    if r == 0: ax.set_title("Foto asli", fontsize=10, fontweight="bold")
    for c, (m, lab) in enumerate((("raw", "Fusi mentah"), ("cmre", "CMRE"))):
        g = grid(i, m); ax = fig.add_subplot(gs[r, c + 1]); ax.imshow(img)
        up = np.asarray(Image.fromarray(g.astype(np.float32), mode="F").resize((448, 448), Image.Resampling.BICUBIC))
        h = ax.imshow(up, cmap="inferno", alpha=.6, vmin=0, vmax=g.max())  # own scale per panel: pattern, not magnitude
        cb = fig.colorbar(h, ax=ax, fraction=.046, pad=.02); cb.ax.tick_params(labelsize=6.5)
        cb.set_label("perubahan embedding", fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"objek {s[f'object_mean_{m}']:.4f} | latar {s[f'background_mean_{m}']:.4f}\n"
                      f"rasio objek/latar = {s[f'ratio_{m}']:.2f}".replace(".", ","), fontsize=7.5)
        if r == 0: ax.set_title(lab, fontsize=10, fontweight="bold")

ax = fig.add_subplot(gs[-1, :]); ax.set_axis_off()
up = int((scr.ratio_gain > 0).sum()); n = len(scr); q = int(scr.qualifies.sum())
text = (
    "Cara membaca. Foto dibagi menjadi 10 × 10 kotak. Setiap kotak ditutup abu-abu satu per satu, lalu foto di-embed\n"
    "ulang lewat pipeline yang sama. Warna terang berarti menutup kotak itu banyak mengubah embedding akhir, sehingga\n"
    "bagian itu dipakai model; warna gelap berarti hampir tidak berpengaruh. Setiap panel memakai skala warnanya sendiri,\n"
    "jadi yang dibandingkan adalah pola sebarannya; besarnya tercantum di bawah panel. Rasio objek/latar di atas 1 berarti\n"
    "objek lebih berpengaruh daripada latar.\n\n"
    f"Ringkasan {n} foto BDC. Rasio objek/latar naik setelah CMRE pada {up} foto dan turun pada {n - up} foto. Hanya {q}\n"
    "foto memenuhi kriteria yang ditetapkan sebelum perhitungan, yaitu puncak berpindah dari latar ke objek. Embedding\n"
    "CMRE berubah sekitar 10 kali lebih kecil saat sebagian foto ditutup. Contoh atas bukan perilaku rata-rata.")
ax.text(-.02, 1, text, va="top", ha="left", fontsize=8, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#bdbdbd"))
fig.suptitle("Sensitivitas oklusi sebelum dan sesudah CMRE", fontsize=12, fontweight="bold")
out = ROOT / "paper_draft/fig/fig_cmre_occlusion_explained.png"
fig.savefig(out, dpi=300, facecolor="white"); print("saved", out, f"| up {up}/{n}, qualifies {q}")
