# Audit ulang 100 seed SCMax terhadap kode penulis

Seed 0–9 dijalankan pada [Modal run 1](https://modal.com/apps/apasijannn/main/ap-Cji9Qb0yzZ0FAq2f7H9ZLy) dan seed 10–99 pada [Modal run 2](https://modal.com/apps/apasijannn/main/ap-0yzkv8Yq8DI68bj4s3Qeng). Seluruhnya memakai GPU L4, fitur CMRE beku, dan pengaturan seed persis seperti demo.py penulis.

Seed 0–99; SCMax memilih K tanpa label. Spectral clustering memakai K pilihan tiap seed, fitur PCA-32 yang sama, 15 tetangga, dan random_state=0. Data eksternal dipetakan ke centroid partisi utama. Ini menguji seed SCMax pada data tetap, bukan bootstrap data atau 100 set uji independen.

- K: median 16.5; modus 17; K=16 pada 20/100 seed.
- xsrc hybrid: 0.919 ± 0.103; median 0.975; persentil 2,5–97,5 0.596–0.988; rentang 0.596–0.988.
- NMI objek eksternal hybrid: 0.853 ± 0.024; median 0.859; persentil 2,5–97,5 0.763–0.886; rentang 0.763–0.890.
- ARI hybrid antarpasangan seed (4.950 pasangan): 0.860 ± 0.110; median 0.800; persentil 2,5–97,5 0.709–1.000; rentang 0.649–1.000.
- ARI hybrid terhadap partisi final K=16: 0.874 ± 0.110; median 0.839; persentil 2,5–97,5 0.729–1.000; rentang 0.705–1.000.

| K pilihan | Seed | xsrc hybrid | NMI eksternal | ARI vs K=16 |
|---:|---:|---:|---:|---:|
| 13 | 1 | 0.978 | 0.890 | 0.879 |
| 14 | 12 | 0.972 | 0.886 | 0.948 |
| 15 | 17 | 0.975 | 0.859 | 0.985 |
| 16 | 20 | 0.975 | 0.859 | 1.000 |
| 17 | 22 | 0.988 | 0.861 | 0.800 |
| 18 | 13 | 0.794 | 0.835 | 0.762 |
| 19 | 11 | 0.794 | 0.836 | 0.731 |
| 20 | 3 | 0.596 | 0.763 | 0.729 |
| 21 | 1 | 0.596 | 0.763 | 0.705 |

Persentil 2,5–97,5% antar-seed adalah sebaran algoritmik, bukan interval kepercayaan generalisasi. Bootstrap data yang menghitung ulang pipeline diperlukan untuk menilai perubahan sampel.
