# Audit 100 seed SCMax pada fitur CMRE beku

Seed 0–7 berasal dari checkpoint sebelumnya; seed 8–9 dijalankan pada [Modal run 1](https://modal.com/apps/apasijannn/main/ap-ebKIp7sM2zYNWKmllOptcc) dan seed 10–99 pada [Modal run 2](https://modal.com/apps/apasijannn/main/ap-Q7pB7VUVqiuOkNuVZn5v7M). Seluruhnya memakai GPU L4, kode SCMax dan input yang sama.

Seed 0–99; SCMax memilih K tanpa label. Spectral clustering memakai K pilihan tiap seed, fitur PCA-32 yang sama, 15 tetangga, dan random_state=0. Data eksternal dipetakan ke centroid partisi utama. Ini menguji seed SCMax pada data tetap, bukan bootstrap data atau 100 set uji independen.

- K: median 16; modus 17; K=16 pada 20/100 seed.
- xsrc hybrid: 0.923 ± 0.098; median 0.975; persentil 2,5–97,5 0.690–0.988; rentang 0.596–0.988.
- NMI objek eksternal hybrid: 0.854 ± 0.022; median 0.859; persentil 2,5–97,5 0.797–0.886; rentang 0.763–0.890.
- ARI hybrid antarpasangan seed (4.950 pasangan): 0.861 ± 0.110; median 0.800; persentil 2,5–97,5 0.708–1.000; rentang 0.649–1.000.
- ARI hybrid terhadap partisi final K=16: 0.879 ± 0.109; median 0.913; persentil 2,5–97,5 0.730–1.000; rentang 0.705–1.000.

| K pilihan | Seed | xsrc hybrid | NMI eksternal | ARI vs K=16 |
|---:|---:|---:|---:|---:|
| 13 | 2 | 0.978 | 0.890 | 0.879 |
| 14 | 10 | 0.972 | 0.886 | 0.948 |
| 15 | 20 | 0.975 | 0.859 | 0.985 |
| 16 | 20 | 0.975 | 0.859 | 1.000 |
| 17 | 21 | 0.988 | 0.861 | 0.800 |
| 18 | 14 | 0.794 | 0.835 | 0.762 |
| 19 | 10 | 0.794 | 0.836 | 0.731 |
| 20 | 2 | 0.596 | 0.763 | 0.729 |
| 21 | 1 | 0.596 | 0.763 | 0.705 |

Persentil 2,5–97,5% antar-seed adalah sebaran algoritmik, bukan interval kepercayaan generalisasi. Bootstrap data yang menghitung ulang pipeline diperlukan untuk menilai perubahan sampel.
