# Audit kode SCMax dan DIVIDE — 27 September 2026

## Sumber dan batas kesetiaan

- **SCMax:** [repositori penulis](https://github.com/ljz441/2026-AAAI-SCMax), commit `89437a1b4aea55fcde23365323d3ba0dc35eeb50`. Tiga modul yang dipakai (`auto_encoder.py`, `neighbor_clustering.py`, `feature_optimization.py`) identik dengan sumber penulis. Loop di `rq_scmax.py` mengikuti `demo.py`: autoencoder 256 dimensi, 200 epoch MSE, 50 epoch *back-feature* per tingkat, batch 256, Adam 3e-4, penggabungan tetangga, dan skor konsensus dengan pencocokan Hungarian. Pengaturan seed kini juga mengikuti `demo.py`, termasuk `torch.cuda.manual_seed_all` dan `cudnn.deterministic=True`.
- **DIVIDE:** [repositori penulis](https://github.com/XLearning-SCU/2024-AAAI-DIVIDE), commit `c31e2a8d1ac9bc89f1ce1e5a1df4b6c00b3d2613`; checkout lokal sama dengan `origin/HEAD`. Wrapper memakai `model.DIVIDE`, `engine_train.train_one_epoch`, dataset dan sampler penulis tanpa mengubah modulnya. Arsitektur penulis memang memakai dua *view*. Di sini *view* pertama adalah fusi SigLIP2 dan PE-Core, sedangkan *view* kedua DINOv3-H+. Ini **aplikasi kode DIVIDE pada data lain**, bukan replikasi angka Scene15 dalam paper DIVIDE.
- Wrapper `code/pipeline/rq_divide.py` memerlukan checkout resmi DIVIDE pada `DIVIDE_CODE_ROOT` (default `repos/DIVIDE`) dan aset evaluasi eksternal lokal di `data/ext_eval`. Repo privat menyimpan wrapper dan hasil, tetapi aset eksternal tersebut tetap harus disediakan saat menjalankan ulang.
- Pada DIVIDE, `StandardScaler`, pelatihan, dan K-Means dipasang hanya pada 3.792 baris *main*. Baris eksternal diproyeksikan setelahnya. K=16 berasal dari protokol BDC. `n_init=10` ditulis eksplisit agar perilaku K-Means setara dengan default scikit-learn sebelum versi 1.4; versi 1.7 lokal memakai `n_init='auto'` yang hanya satu inisialisasi untuk `k-means++`.

## Masalah data yang ditemukan

Salinan lokal `emb_dinov3hp_flip.npy` berbeda dari volume Modal pada 1.895 baris eksternal: 1.893 baris kosong dan dua baris berbeda. Semua baris *main* identik. Salinan volume ber-SHA-256 `f4b6f04158f1d3965460e77156669d2a281754b8000ff69299a7a1d756383268` memulihkan `z_final_v4.npy` dengan rerata selisih absolut `1,53e-6`. Berkas lama disimpan di workspace lokal sebagai `results/classic/emb_dinov3hp_flip_local_mismatch_backup.npy`; dalam repo, versi lama tersedia di riwayat Git. Hasil percobaan sebelum perbaikan disimpan terpisah sebagai `divide_mismatched_input.csv` dan `scmax_local_mismatched_*.csv`.

## Uji ulang SCMax

Pada GPU L4 Modal, 100 seed (0–99) dijalankan ulang pada fitur CMRE beku. Hasil lengkap: [CSV per seed](../modal/scmax_results_100_source_audit.csv), [laporan sebaran](../modal/scmax_100_source_audit_report.md), dan [batch tingkat penggabungan](../modal/scmax_source_audit_batches/). K terpilih 13–21, median **16,5**, modus 17, dan K=16 pada 20/100 seed. Dibanding hasil lama, pilihan K berubah pada **12/100 seed**. Pada sembilan seed yang dipakai untuk memilih partisi final (3407, 0–7), median tetap **K=16**.

Perlu memisahkan keluaran metode: partisi **SCMax asli** memberi rerata *cross-source agreement* 0,731 ± 0,127. Jika **hanya K dari SCMax** dipakai untuk Spectral, reratanya 0,919 ± 0,103. Angka kedua adalah metode hibrida CMRE → SCMax untuk K → Spectral, bukan performa SCMax end-to-end. Seluruh simpangan baku ini adalah variasi seed pada embedding tetap, bukan *confidence interval* untuk sampel baru.

Fitur CMRE *main* yang dipakai Modal tersimpan di [matriks audit](../modal/scmax_cmre_main_source_audit.npy). Seed 3407 pada matriks yang sama memberi **K=13 di L4 Modal** dan **K=16 di RTX 3050 lokal**. Dua pengulangan lokal dengan matriks dan seed yang sama sama-sama memilih K=16. Jadi pengaturan seed deterministik tidak menjamin pilihan K identik lintas GPU/lingkungan. Perbedaan perhitungan CMRE lokal–Modal sangat kecil (maksimum `4,97e-7` per elemen), tetapi uji dengan matriks yang persis sama tetap berbeda. Hasil 100 seed harus diartikan dalam satu lingkungan komputasi yang konsisten.

## Uji ulang DIVIDE

Tiga seed (0, 1, 2), 200 epoch per seed, K=16, input dan evaluasi yang telah diperbaiki. Nilai adalah rerata ± simpangan baku *cross-source agreement* antar-seed. [CSV lengkap](divide_source_audit.csv).

| Metode | BDC | Iliev | Shubha |
|---|---:|---:|---:|
| CMRE → DIVIDE | 0,647 ± 0,011 | 0,699 ± 0,104 | 0,661 ± 0,006 |
| Fusi mentah → DIVIDE | 0,682 ± 0,101 | 0,724 ± 0,036 | 0,663 ± 0,029 |
| CMRE → Spectral, K=16 | 0,975 | 0,872 | 0,956 |

DIVIDE memakai arsitektur dan prosedur belajar penulis, tetapi konfigurasi Scene15 dipindahkan ke representasi BDC tanpa pencarian hiperparameter baru. Angka DIVIDE sebelumnya dalam `divide_results.csv`, terutama hibrida `K=SCMax` pada embedding DIVIDE, **tidak boleh dijadikan hasil terkini**. Hibrida itu bukan keluaran asli DIVIDE dan tidak diulang pada audit ini. Tiga seed hanya mengukur variasi inisialisasi, bukan galat generalisasi.

## Implikasi untuk naskah

Pernyataan “median K dari 100 seed = 16” harus diperbarui menjadi **16,5**. K=16 tetap didukung median sembilan seed yang dipakai dalam pemilihan final, tetapi 100 seed tidak menghasilkan satu median bulat tanpa aturan pembulatan tambahan. Jika mengutip DIVIDE, sebut sebagai adaptasi dua *view* pada fitur BDC. Jika mengutip xsrc 0,919, sebut sebagai hibrida SCMax–Spectral; xsrc partisi SCMax sendiri 0,731.
