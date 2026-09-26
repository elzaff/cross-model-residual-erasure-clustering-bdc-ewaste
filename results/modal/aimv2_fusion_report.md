# Audit fusi AIMv2 (26 September 2026)

## Protokol

Embedding AIMv2-L asli dan horizontal flip diunduh dari volume Modal `bdc-max`. Kedua matriks berukuran 7.565 × 1.024 dan memakai manifest yang identik dengan embedding v4. Embedding AIMv2 untuk Iliev (1.771 baris termasuk kontrol) dan Shubha (380 baris termasuk kontrol) diekstrak dengan model, normalisasi, dan resolusi masukan yang sama seperti audit awal. Pemeriksaan cosine pada citra kontrol mensyaratkan nilai minimum 0,98 sebelum evaluasi dilanjutkan.

Perbandingan BDC memakai 3.792 citra `main` untuk fitting PCA-32 dan Spectral clustering (`K=16`, graf 15 tetangga, seed 0–2). Citra lain diproyeksikan dan diberi kelompok dari centroid terdekat. AIMv2 memakai checkpoint `apple/aimv2-large-patch14-224` dan dibiarkan mentah agar ablasi mengukur kontribusi encoder tambahan tanpa proyeksi baru. SigLIP2 dan PE-Core diuji dalam bentuk mentah serta setelah CMRE dengan DINOv3-H+ sebagai acuan. Tiap embedding dinormalisasi L2 sebelum fusi konkatenasi.

## Hasil pada BDC

| Representasi | xsrc | NMI objek eksternal | Purity BDC | Purity eksternal |
|---|---:|---:|---:|---:|
| AIMv2-L | 0,842 | 0,801 | 0,974 | 0,952 |
| v4 dihitung ulang: CMRE(Sig) + CMRE(PE) + DINO | 0,970 | 0,855 | 0,992 | 0,982 |
| AIM + Sig mentah + DINO | 0,564 | 0,790 | 0,989 | 0,972 |
| AIM + PE mentah + DINO | 0,601 | 0,797 | 0,992 | 0,976 |
| AIM + Sig mentah + PE mentah + DINO | 0,539 | 0,793 | 0,994 | 0,973 |
| AIM + CMRE(Sig) + DINO | 0,940 | 0,823 | 0,990 | 0,975 |
| AIM + CMRE(PE) + DINO | 0,965 | 0,856 | 0,992 | 0,983 |
| **AIM + CMRE(Sig) + CMRE(PE) + DINO** | **0,987** | **0,865** | **0,994** | **0,984** |

Semua partisi memiliki ARI antarseed 1,00 pada `K=16`. Hasil v4 yang dihitung ulang secara lokal (`xsrc=0,970`) berbeda sedikit dari artefak final tersimpan (`0,975`); tabel ini memakai perhitungan lokal yang sama untuk seluruh kandidat.

## Audit reproduksi dan sumber galat

Penghitungan ulang v4 memberi xsrc BDC 0,970011, dibanding 0,975411 pada artefak final (selisih −0,005400). Setelah label cluster disejajarkan, **seluruh 3.792 citra utama memperoleh partisi yang sama** (ARI 1,000). Dari 3.773 citra di luar data utama, hanya enam penempatan centroid berubah: empat citra Bangladesh, satu Kaan, dan satu GIZ. ARI pada seluruh 7.565 citra 0,9979. Embedding PCA tersimpan mengembalikan penempatan arsip secara tepat. Pada sampel 512 citra, cosine berpasangan antara fitur PCA tersimpan dan hasil hitung ulang berkorelasi 0,9998, dengan beda absolut rerata 0,0021. Jadi selisih xsrc berasal dari sedikit perubahan penempatan citra luar saat fitur dihitung ulang; penyebab teknis perubahan fitur tersebut belum terisolasi.

Metadata konfigurasi lama menulis `r_siglip2 = r_pecore = 65`. Nilai itu adalah indeks ambang 50% varians setelah `PCA(64)`, sedangkan `components_[:65]` hanya berisi **64 arah**. Proyeksi yang benar-benar diterapkan memakai 64 arah; metadata diberi `effective_r_* = 64` tanpa mengubah catatan indeks lama.

Audit 100 seed SCMax pada v4 mengubah seed dengan data dan embedding tetap: K = 13–21; rerata xsrc 0,923 ± 0,098, rentang 0,596–0,988; ARI antarpasangan partisi 0,861 ± 0,110. Pada K = 16 yang dibekukan, seed tambahan Spectral menghasilkan ARI 1,00. Tiga seed kandidat AIMv2 juga memberi ARI 1,00 pada K = 16. Angka-angka ini mengukur variasi inisialisasi dan pilihan K; **bukan 100 pengulangan penuh** atas ekstraksi, CMRE, PCA, seleksi model, serta sampel baru. Bootstrap berpasangan 2.000 kali di Iliev/Shubha mengambil ulang citra evaluasi per kelas dengan pipeline tetap, sehingga intervalnya juga tidak mencakup variasi pembentukan representasi.

## Evaluasi eksternal kandidat terbaik

Kandidat empat model di atas dipilih dari audit BDC sebelum dinilai pada Iliev dan Shubha. Partisi v4 memakai assignment final yang dibekukan; v4 juga di-cluster ulang dalam lingkungan yang sama sebagai pemeriksaan. Kedua versi v4 menghasilkan angka eksternal yang identik. Pada kedua set, model hanya di-fit pada `main`; citra uji diproyeksikan ke centroid kelompok.

| Set | v4 xsrc | Empat model xsrc | Selisih | IK bootstrap berpasangan 95% untuk selisih | Purity keduanya |
|---|---:|---:|---:|---:|---:|
| Iliev, 10 kelas | 0,885 | 0,887 | +0,002 | [−0,012; 0,015] | 0,939 |
| Shubha, 4 kelas | 0,956 | 0,967 | +0,011 | [0,000; 0,028] | 1,000 |

Bootstrap mengambil ulang citra dalam tiap kelas sebanyak 2.000 kali dan mempertahankan rerata makro antarkelas. Interval Iliev melintasi nol; Shubha menyentuh nol. Hasil eksternal belum cukup untuk menyatakan peningkatan yang konsisten. Penambahan AIMv2 juga menambah satu encoder saat inferensi. Pemilihan `K` bebas label dengan SCMax dan audit 100 seed belum diulang untuk fitur empat model.

Shubha sekarang sudah dipakai untuk menilai varian AIMv2 ini. Jika varian diubah lagi setelah melihat hasil tersebut, Shubha tidak dapat diklaim sebagai set uji yang belum pernah dilihat untuk varian berikutnya.

Data lengkap: `aimv2_fusion_results.csv` dan `aimv2_external_results.csv`. Kode: `code/pipeline/compare_aimv2_fusions.py`, `code/pipeline/embed_aimv2_new.py`, dan `code/pipeline/compare_aimv2_external.py`. AIMv2 eksternal tersimpan di volume `bdc-max` pada `/out/iliev/emb_aimv2*.npy` dan `/out/shubha/emb_aimv2*.npy`.
