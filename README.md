# Unsupervised Discovery Kategori Sampah Elektronik dari Citra Multi-Sumber

**Cross-Model Residual Erasure (CMRE) + SCMax + Spectral Clustering + deteksi OOD kNN**
Karya ilmiah babak semifinal **Big Data Challenge (BDC) Satria Data 2026**.

Foto e-waste datang dari sumber yang sangat beragam: katalog daring berlatar putih, kamera ponsel, sampai foto lapangan di
pengepul. Kami menemukan bahwa model fondasi visi-bahasa (SigLIP2, PE-Core, CLIP) cenderung mengelompokkan foto menurut
**cara foto diambil**, bukan menurut **bendanya**. Repositori ini berisi pipeline clustering tanpa label yang mengurangi bias
sumber foto tersebut, beserta semua hasil, gambar, dashboard, dan kode untuk mereproduksinya.

- **Input model hanya citra.** Tidak ada teks, caption, prompt, atau label kelas yang dipakai untuk membentuk kelompok.
- **Label hanya untuk evaluasi**, dihitung setelah clustering selesai. Label data uji panitia tidak pernah dibaca.

## Tim

| | |
|---|---|
| Kode kelompok / registrasi | SD2026040000100 |
| Cabang lomba | Big Data Challenge (BDC), Satria Data 2026 |
| Perguruan tinggi | Institut Teknologi Sepuluh Nopember (Program Studi Rekayasa Kecerdasan Artifisial) |
| Pembimbing | Dini Adni Navastra, S.Kom., M.Sc. (0349763664230223) |

| No | Nama | NRP | Peran |
|---|---|---|---|
| 1 | Berliana Sarlita Rahajeng | 5054241023 | Anggota |
| 2 | Fazle Mawla Wahyuhanda | 5054241020 | Ketua |
| 3 | Muhammad Fatih Al Fawwaz | 5054241007 | Anggota |

## Pipeline

```
3.792 citra BDC (3.604 latih unik + 188 citra e-waste dari data uji)
  │  dedup pHash, TTA flip (asli + cermin, dirata-rata)
  ▼
SigLIP2-So400m ─┐                     DINOv3-H+ (self-supervised, acuan)
PE-Core-G ──────┤                          │
                ▼                          ▼
   CMRE: ridge V ← S, residu R = V − S·B, hapus r = 64 arah utama residu (PCA)
                ▼
   fusi [CMRE(SigLIP2); CMRE(PE); DINOv3] → PCA-32 → L2
                ▼
   SCMax (median 9 seed → K = 16) → Spectral clustering (graf 15 tetangga)
                ▼
   16 kelompok + skor OOD kNN (jarak ke tetangga ke-10, ambang persentil 95)
```

CMRE (*Cross-Model Residual Erasure*): bagian embedding visi-bahasa yang tidak dapat diprediksi dari model *self-supervised*
diuraikan dengan PCA, lalu arah-arah utamanya diproyeksikan keluar. Arah ini berkorelasi dengan resolusi foto (|ρ| hingga 0,68)
dan latar putih (0,39). Informasi sumber **tidak** hilang (probe sumber tetap 0,96–0,99); yang berkurang adalah dominasinya
terhadap struktur ketetanggaan. Implementasi inti ada di [`code/ewaste_cmre.py`](code/ewaste_cmre.py), kurang dari 100 baris.

## Hasil utama

**xsrc** (konsistensi lintas sumber) adalah proporsi foto dari sumber kedua yang masuk ke kelompok mayoritas foto katalog
(Kaan) untuk kelas yang sama, dirata-rata per kelas. Nilai 1 berarti pengelompokan tidak terpengaruh sumber foto.

| Set data | Peran | Fusi mentah | DINOv3-H+ saja | INLP ×3 (butuh proksi) | **CMRE (usulan)** |
|---|---|---|---|---|---|
| BDC, 5 kelas (Kaan vs foto ponsel Bangladesh) | pengembangan | 0,524 | 0,758 | 0,975 | **0,975** [0,958–0,989] |
| Iliev, 10 kelas, 1.751 crop | uji terkunci (xsrc) | 0,803 | 0,755 | 0,882 | **0,885** [0,864–0,908] |
| Shubha, 3 kelas dikenal, 300 crop | uji segar (konfirmasi awal) | 0,356 | 0,889 | 0,972 | **0,956** [0,924–0,983] |
| Office-Home, 20 kelas | domain lain (tidak bias) | 0,865 | 0,761 | 0,870 | 0,864 |

- Purity BDC **0,992**, stabilitas antar-seed (ARI) **1,00**, NMI objek 0,859.
- **CMRE setara INLP tanpa merancang proksi gaya apa pun.** Uji bootstrap berpasangan (2.000 ulangan) menunjukkan selisih
  terhadap INLP tidak signifikan, sedangkan selisih terhadap fusi mentah dan DINOv3 saja signifikan (p < 0,001) pada ketiga set.
- Pada Office-Home, yang fiturnya memang tidak bias, CMRE tidak menambah kinerja tetapi juga tidak merusak.

### Deteksi kategori di luar cakupan (OOD kNN)

| Set | AUROC | Catatan |
|---|---|---|
| Iliev (set pengembangan untuk OOD) | 0,937 | naik dari 0,868 (jarak ke pusat kelompok) |
| Shubha (uji segar, 2 kelas OOD) | **0,994** [0,984–0,999] | remote dan headphone 100% ditandai |

Contoh: AC 97%, kulkas 83%, setrika/kipas/lampu LED/dispenser 100% ditandai "tidak dikenal", begitu pula 93,5% foto lampu
Bangladesh. Alarm palsu pada kelas yang dikenal: Kaan 3,0%, Bangladesh 13,1%. Foto lapangan GIZ yang ditandai (44,5%)
didominasi AC, kulkas, dan kompresor, yaitu kategori WEEE 1 yang tidak ada di data BDC.

### 16 kelompok yang ditemukan

| C | Isi | n | Kategori WEEE / B3 | | C | Isi | n | Kategori WEEE / B3 |
|---|---|---|---|---|---|---|---|---|
| 0 | Aki timbal-asam | 115 | baterai (B3) | | 8 | Mesin cuci front-load | 192 | 4 (besar) |
| 1 | Baterai pack laptop | 70 | baterai Li-ion | | 9 | Laptop | 277 | 2 (layar) |
| 2 | Turntable | 111 | 5 (kecil) | | 10 | Keyboard | 320 | 6 (TI kecil) |
| 3 | TV/monitor CRT | 174 | 2 (layar) | | 11 | Microwave | 275 | 5 (kecil) |
| 4 | Baterai ponsel/kamera | 136 | baterai Li-ion kecil | | 12 | Printer | 294 | 6 (TI kecil) |
| 5 | Pemutar CD/radio | 153 | 5 (kecil) | | 13 | Mouse | 291 | 6 (TI kecil) |
| 6 | Mesin cuci bukaan atas | 115 | 4 (besar) | | 14 | Ponsel | 366 | 6 (TI kecil) |
| 7 | TV/monitor layar datar | 263 | 2 (layar) | | 15 | PCB dan tumpukan campuran | 640 | komponen |

- 15 dari 16 kelompok berisi ≥ 90% satu label BDC.
- Kelompok **laptop (C9)** tidak tercantum pada label panitia (berkasnya berlabel "keyboard"). Temuan ini tervalidasi di Iliev:
  96,7% foto laptop masuk ke C9.
- Tingkat kasar K = 13 dipecah menjadi K = 16: 14 dari 16 kelompok mewarisi ≥ 96% citranya dari satu induk.
- Uji permutasi WEEE: kategori EU-6 crop Iliev yang dikenali cocok dengan kategori grup kelompoknya pada 79% kasus,
  dibandingkan 19% untuk penggabungan acak (p < 0,001).

### Uji ketahanan (setelah partisi final dibekukan)

| Pemeriksaan | Hasil |
|---|---|
| Metode clustering terbaru pada fitur CMRE yang sama (K = 16) | Spectral **0,975** · DECMCV 0,910 (hanya 81% citra) · TURTLE 0,790 · TEMI 0,590 |
| Dijalankan ulang **tanpa 188 citra data uji** | xsrc tetap **0,975** (Iliev 0,895, Shubha 0,950); ARI terhadap partisi final 0,93 |
| Aturan pemilihan acuan tanpa label (R² ridge) | Dua acuan teratas (DINOv3-7B, DINOv3-H+) = dua xsrc terbaik (0,983; 0,975); acuan berlabel gagal (0,384) |
| Sensitivitas r (jumlah arah dihapus) | 0,97–0,99 untuk r = 16–64; turun di luar rentang itu (r = 4: 0,449; r = 128: 0,821) |
| Sensitivitas α ridge | 0,975–0,983 untuk α = 0,01–1; turun pada α ≥ 10 |
| Versi cepat tanpa PE-Core-G | ~20 citra/detik di GPU L4 (6× lebih cepat), xsrc 0,951 / 0,860 / 0,922 |
| Waktu setelah ekstraksi fitur | sekitar 1 detik CPU untuk 3.792 citra |

## Struktur repositori

```
├── code/
│   ├── ewaste_cmre.py        implementasi inti: CMRE, fusi, clustering, OOD kNN (+ uji mandiri)
│   ├── reproduce.py          reproduksi partisi final di CPU (< 1 menit, tanpa GPU/model)
│   └── pipeline/             seluruh eksperimen (Modal): ekstraksi fitur, audit, pembanding, SCMax, validasi
├── data/build_manifest.py    menyusun daftar citra dari data mentah (data mentah tidak disertakan)
├── results/modal/            data minimal untuk reproduce.py: fitur final z_final_v4.npy, keanggotaan kelompok
└── figures/                  gambar hasil (Gambar 1–9)
```

Dashboard presentasi: https://elzaff.github.io/cross-model-residual-erasure-clustering-bdc-ewaste-dashboard/ (kode halaman: https://github.com/elzaff/cross-model-residual-erasure-clustering-bdc-ewaste-dashboard)

## Reproduksi

### 1. Cepat, di CPU (tanpa GPU dan tanpa mengunduh model)

```bash
pip install -r requirements.txt
python code/reproduce.py        # ARI 1,0000 vs partisi final; xsrc 0,975; laju OOD per sumber
python code/ewaste_cmre.py      # uji mandiri implementasi inti
```

### 2. Penuh, dari citra mentah (GPU di [Modal](https://modal.com))

1. Unduh data dari tautan pada tabel di bawah, lalu susun daftar citra dengan `data/build_manifest.py`.
2. Siapkan Modal:
   ```bash
   pip install modal
   modal setup
   modal secret create huggingface-secret HF_TOKEN=<token Anda>        # DINOv3 bersifat gated
   modal secret create roboflow-secret ROBOFLOW_API_KEY=<key Anda>     # hanya untuk Iliev dan Shubha
   modal volume create bdc-max
   modal volume put bdc-max <folder_staging> /data
   ```
3. Jalankan tahap berurutan dari `code/pipeline` (Git Bash: `export MSYS_NO_PATHCONV=1`):

| Tahap | Isi |
|---|---|
| `--stage all` | ekstraksi fitur 17 representasi (GPU) + audit bias sumber |
| `--stage final2` | pembanding adil (INLP, LEACE, fusi mentah), kontrol data MetaCLIP vs Web-DINO |
| `--stage scmax` lalu `--stage scmax_seeds` | SCMax 9 seed → median K = 16 |
| `--stage final4` | partisi final → `/out/final_v4` |
| `--stage fix`, `--stage fix2` | baseline, probe, sapuan r/K/tetangga, OOD kNN, Jaccard |
| `--stage iliev`, `--stage shubha`, `--stage officehome` | validasi eksternal (protokol dikunci sebelum unduh) |
| `--stage paper` | aset gambar paper |
| `--stage r8`, `--stage r8b`, `--stage d1`, `--stage notest`, `--stage gallery` | analisis ronde 8 |
| `--stage bench`, `--stage bench_dino` | throughput embedding di GPU L4 |

```bash
modal run --detach modal_app.py --stage all
modal volume get bdc-max /out ../../results/modal
```

## Data

Tidak ada data mentah yang didistribusikan ulang di repositori ini. Jumlah citra dihitung setelah dedup pHash terhadap BDC.

| Set | Peran | Jumlah | Lisensi | Tautan |
|---|---|---|---|---|
| BDC Satria Data 2026 | data clustering | 3.792 | aturan lomba | panitia BDC |
| Kaan, Waste Classification | acuan xsrc (katalog) | 1.794 | CDLA-Permissive-1.0 | [Kaggle](https://www.kaggle.com/datasets/kaanerkez/waste-classfication-dataset) |
| Custom Bangladeshi E-Waste | target xsrc (foto ponsel) | 480 | CC BY 4.0 | [Mendeley](https://data.mendeley.com/datasets/77383kmdnw/1) |
| Karan, Garbage Classification (baterai) | evaluasi objek | 499 | Apache 2.0 | [Kaggle](https://www.kaggle.com/datasets/karansolanki01/garbage-classification) |
| GIZ E-Waste Database | uji OOD foto lapangan | 1.000 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/GIZ/E-Waste-Database) |
| Iliev, E-Waste Dataset (Roboflow v44) | validasi terkunci | 1.751 crop, 69 kelas | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/electronic-waste-detection/e-waste-dataset-r0ojc) |
| Shubha, E-waste (Roboflow v2) | uji segar | 300 crop | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/shubha-to6ii/e-waste-1sn3k) |
| Office-Home (Product, Real World) | uji domain lain | 2.917 | riset non-komersial | [Hugging Face](https://huggingface.co/datasets/flwrlabs/office-home) |

Model: [SigLIP2-So400m](https://huggingface.co/google/siglip2-so400m-patch16-naflex),
[PE-Core-G](https://huggingface.co/timm/vit_pe_core_gigantic_patch14_448.fb),
[DINOv3-H+](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m). Semuanya dipakai tanpa fine-tuning.

## Kepatuhan aturan lomba

- Model hanya menerima citra. Menara teks model visi-bahasa tidak pernah dipanggil.
- Label nama berkas BDC dan label set eksternal hanya dipakai untuk evaluasi setelah clustering.
- 188 citra dari data uji dipilih dengan pengklasifikasi citra tim sendiri (hanya prediksi). Menjalankan ulang tanpa citra
  tersebut memberi hasil yang sama (xsrc 0,975).
- Protokol Iliev dan Shubha (pemetaan kelas dan metrik) dikunci sebelum data diunduh.

## Keterbatasan

1. BDC adalah set pengembangan, sehingga angka BDC bersifat optimistis. Bukti generalisasi berasal dari Iliev dan Shubha,
   dan Shubha masih kecil (300 crop).
2. CMRE setara INLP, bukan lebih unggul. Manfaatnya bersyarat: tidak ada kenaikan pada data yang tidak bias (Office-Home),
   dan acuannya harus model *self-supervised* berskala besar.
3. SCMax memilih K antara 13 dan 19 antar-seed; K = 16 adalah median, dan K = 19 memberi xsrc 0,794.
4. Deteksi OOD masih memakai ambang global, sehingga 13% foto ponsel dari kelas yang dikenal ikut ditandai.
5. Belum ada foto e-waste yang diambil langsung di Indonesia. Pemetaan kelompok ke kategori WEEE dilakukan penulis
   setelah melihat galeri.

## Lisensi

- Kode: MIT (lihat [`LICENSE`](LICENSE)). `code/pipeline/scmax/` adalah kode resmi SCMax (MIT).
- Citra contoh di `figures/` berasal dari set data di atas dan tetap
  mengikuti lisensi aslinya. Citra BDC adalah milik panitia BDC Satria Data 2026.
