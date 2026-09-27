# Uji jalur klasik dan revisi naskah CMRE

## Protokol dan hasil

Eksperimen ini **analisis pasca-hoc**. Subruang dipelajari hanya dari `D_main`; semua jalur memakai PCA-32, Spectral clustering, dan K = 16. Rank proyeksi r = 64. Pada perbandingan utama tiga model, EPO-flip mempelajari PCA dari selisih embedding foto asli dan horizontal flip untuk SigLIP2 dan PE, lalu membuang subruang itu; DINOv3 dibiarkan utuh, sama seperti pada fusi CMRE v4. EPO-flip di sini adalah adaptasi operasional EPO, bukan uji gangguan sumber secara langsung: horizontal flip tidak sama dengan perubahan dari foto katalog ke foto HP pada objek identik.

| Jalur tiga model | BDC xsrc (5 kelas) | Iliev xsrc (10 kelas terkunci) | Shubha xsrc (3 kelas) |
|---|---:|---:|---:|
| Fusi mentah | 0,524 | 0,803 | 0,356 |
| EPO-flip pada SigLIP2 + PE | 0,765 | **0,928** | 0,656 |
| CMRE v4 pada SigLIP2 + PE | **0,975** | 0,885 | **0,956** |
| EPO-flip pada ketiga model (kontrol eksploratif) | 0,598 | 0,878 | 0,378 |

Selisih EPO-flip minus CMRE pada xsrc dan IK95% bootstrap berpasangan: BDC −0,211 [−0,228; −0,194], Iliev +0,043 [+0,020; +0,067], Shubha −0,300 [−0,328; −0,267]. Bootstrap mengulang 2.000 kali foto sumber target **di dalam setiap kelas** dengan indeks yang sama untuk kedua metode; embedding, cluster, dan acuan mayoritas Kaan tetap. Ini mengukur ketidakpastian sampel target bersyarat, **bukan** ketidakpastian akibat pemilihan metode, seed clustering, atau domain baru.

Pada xsrc, penyebutnya adalah 449 foto sumber target BDC dalam 5 kelas, 345 foto Iliev dalam 10 kelas bersama, dan 180 foto Shubha dalam 3 kelas bersama. Total dataset Iliev adalah 1.751 foto dan Shubha 300 foto; jangan menulis bahwa seluruh foto tersebut masuk hitungan xsrc. `score_all()` yang lebih baru memakai batas minimal 20 foto per sel dan hanya menghitung 9 kelas Iliev karena kelas printer (15 foto) terlewat; hasilnya EPO 0,920 dan CMRE 0,872. **Jangan campur** angka 9 kelas ini dengan angka Iliev 10 kelas terkunci pada naskah (EPO 0,928; CMRE 0,885). Kolom `xsrc10_locked` di CSV mereproduksi protokol naskah.

Pada metrik lain, CMRE memiliki xtv lebih tinggi daripada EPO-flip pada BDC (0,914 vs 0,813), Iliev (0,838 vs 0,805), dan Shubha (0,968 vs 0,759). `leak_cmi` CMRE juga lebih rendah pada ketiganya (0,016 vs 0,057; 0,087 vs 0,106; 0,013 vs 0,204). Karena itu, kemenangan EPO-flip di xsrc Iliev tidak berarti EPO unggul pada semua ukuran. EPO memiliki AMI Iliev lebih tinggi (0,929 vs 0,906); laporkan tradeoff ini bila AMI dibahas.

Dalam varian **dua model** (SigLIP2 + DINOv3), EPO-flip SigLIP2 hampir menyamai CMRE pada BDC (0,935 vs 0,951), dan melampauinya pada Iliev protokol 9 kelas (0,894 vs 0,844) serta Shubha (0,956 vs 0,922). Jadi manfaat CMRE bergantung pada konfigurasi fusi. Seluruh angka dua model berada di `classic_routes.csv` dan tidak sebanding langsung dengan angka Iliev 10 kelas di tabel tiga model.

Nama baseline perlu tepat: penghapusan top-64 PC tanpa kovariat ialah **ABTT/top-PC removal**, bukan SVA/RUV penuh; standardisasi location/scale per proksi gaya bukan **ComBat** empiris Bayes; PCA residu bersyarat label adalah diagnostik yang memakai label, bukan batas atas matematis. Pada skrip dua model, semua baris Iliev/Shubha diberi proksi gaya pertama sebagai placeholder. Oleh sebab itu, angka standardisasi proksi batch di luar BDC tidak layak diklaim sebagai evaluasi generalisasi ComBat; perlu inferensi gaya eksternal yang valid sebelum menjadi baseline formal. Nama di skrip dua model telah diperbaiki tanpa mengubah hasil numeriknya.

Status data juga perlu jelas: protokol **CMRE asli** pada Iliev dikunci sebelum unduh, tetapi pemilihan dan perbandingan EPO-flip ini dilakukan setelah hasil Iliev diketahui. Sebut perbandingan EPO sebagai *analisis pasca-hoc pada data eksternal*, bukan konfirmasi prospektif tambahan. Iliev kemudian dipakai untuk memilih setelan OOD; peran tersebut berbeda dari pengujian xsrc awalnya.

## Bagian naskah yang perlu diubah

Sumber naskah aktif: `code/build_paper.py` → `paper_draft/Karya_Ilmiah_BDC2026.docx`. Markdown draft v2 bukan sumber aktif.

| Bagian | Perubahan yang disarankan |
|---|---|
| §2.2 Dasar Teori (sekitar baris 325–337) | Tambah EPO sebagai preseden proyeksi subruang gangguan berbasis pengukuran berpasangan. Bedakan pengukuran ulang karena parameter eksternal pada EPO asli dari flip foto pada kontrol ini dan residu lintas representasi pada CMRE. Kutip Roger et al. (2003). Jangan menyebut ide proyeksi ortogonal sebagai kebaruan CMRE. |
| §2.3 Metode (344–378) | Jelaskan secara eksplisit bahwa proyeksi r=64 diterapkan pada SigLIP2 dan PE, sementara DINOv3 tetap utuh dalam pembandingan utama; bedakan cara memperoleh subruang CMRE (residu terhadap SSL) dan EPO-flip (selisih asli–flip). |
| §2.4 Skenario Pengujian (380–386) | Tambah analisis pasca-hoc EPO-flip serta bootstrap berpasangan. Nyatakan protokol xsrc Iliev 10 kelas terkunci dan jumlah foto target yang masuk hitungan; jangan gabungkan dengan protokol 9 kelas yang membatasi minimal 20 per sel. |
| §3.2 Hasil (421–463) | Sisipkan tabel kecil tiga dataset berisi fusi mentah, EPO-flip, CMRE dari tabel di atas. Pertahankan Tabel 4 yang ada sebagai perbandingan BDC; jangan masukkan satu baris EPO ke Tabel 4 tanpa kolom purity/NMI yang dihitung sepadan. Tambahkan selisih dan IK berpasangan serta tradeoff xtv/leak_cmi. Lunakkan kalimat bahwa CMRE membantu setiap kali bias ada tanpa merusak kondisi tanpa bias. |
| §3.6 Studi terkait (539–552) | Tempatkan CMRE sebagai kombinasi spesifik regresi lintas representasi dan proyeksi residu pada embedding beku, dengan EPO/HEX sebagai pendahulu konsep; jelaskan kegagalan EPO-flip pada BDC/Shubha dan keberhasilannya di Iliev. |
| §3.7 Keterbatasan (554–563) | Tambah bahwa EPO-flip lebih tinggi 0,043 xsrc pada Iliev 10 kelas; pada dua model jaraknya dengan CMRE lebih kecil atau berbalik. Analisis pasca-hoc dan bootstrap bersyarat membatasi generalisasi. |
| Abstrak (214–228) dan §4 Kesimpulan (566–577) | Angka CMRE vs fusi mentah masih benar. Bila ada ruang, tambah satu klausa bahwa EPO-flip lebih tinggi di Iliev, sehingga CMRE tidak diklaim paling unggul di semua dataset/metrik. |
| Referensi | Tambah Roger et al., “EPO–PLS external parameter orthogonalisation of PLS”, *Chemometrics and Intelligent Laboratory Systems* (2003), DOI: 10.1016/S0169-7439(03)00051-0. Jika ABTT dibahas dalam naskah, tambahkan Mu & Viswanath (2018); jangan mencantumkan ComBat/SVA/RUV sebagai baseline yang telah diuji penuh. |

### Paragraf siap adaptasi untuk §3.2

> Sebagai analisis pasca-hoc, kami membandingkan CMRE dengan EPO-flip pada fusi tiga model dan setelan clustering yang sama (PCA-32, Spectral, K=16, r=64). EPO-flip memperkirakan subruang perturbasi dari selisih embedding foto asli dan horizontal flip untuk SigLIP2 dan PE, sementara DINOv3 tetap utuh. Pada xsrc, CMRE lebih tinggi pada BDC (0,975 vs 0,765) dan Shubha (0,956 vs 0,656), tetapi EPO-flip lebih tinggi pada Iliev dengan pemetaan 10 kelas terkunci (0,928 vs 0,885). Selisih EPO minus CMRE pada Iliev adalah +0,043 (IK95% bootstrap berpasangan +0,020 hingga +0,067); interval ini bersyarat pada cluster dan model yang telah dipilih. CMRE tetap memiliki xtv lebih tinggi serta `leak_cmi` lebih rendah pada ketiga dataset. Hasil ini menunjukkan bahwa keunggulan xsrc CMRE atas proyeksi sederhana bergantung pada domain, sementara karakteristik lain dari cluster dapat bergerak berbeda. Karena baseline EPO dipilih setelah hasil Iliev terlihat, pembandingan ini tidak dihitung sebagai validasi terkunci baru.

### Berkas bukti

- `code/pipeline/rq_classic_routes_3model.py`: skrip reproduksi tiga model dan bootstrap berpasangan.
- `results/classic/classic_routes_3model.csv`: seluruh metrik; `xsrc10_locked` khusus Iliev.
- `results/classic/classic_routes_3model_paired_ci.csv`: selisih xsrc dan interval bootstrap.
- `code/pipeline/rq_classic_routes.py` dan `results/classic/classic_routes.csv`: analisis dua model dengan nama baseline yang telah dikoreksi.
- `paper_draft/CMRE_landasan_literatur.md`: landasan dan batas klaim kebaruan yang telah diperbarui.
