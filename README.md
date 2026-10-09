# Deteksi Objek Basket dengan YOLO (backboard, ball, rim)

Tugas RET503 Computer Vision and Deep Learning, tahap YOLO (data, training, tuning, evaluasi).
Proyek: persepsi robot humanoid **FIRA HuroCup 2027, kategori Basketball** (Barelang FC, Politeknik Negeri Batam).
Pemilik: Muhammad Akbar Iqvi (NIM 4222401006). Ini kelanjutan dari repositori transfer learning (klasifikasi potongan, Pertemuan 3).

**Model:** YOLO26n (Ultralytics 8.4.174), 30 epoch, batch 16, seed 0, GPU Tesla T4.

## Isi repositori

```
├── README.md
├── colab/yolo_colab.ipynb        # seluruh alur (E1, E2, E3, val, benchmark, test)
├── scripts/
│   ├── build_yolo_dataset.py     # sumber_frame + metadata.csv -> dataset_v1 (split blok) + data.yaml
│   └── benchmark.py              # ms/gambar (batch 1, warm-up)
├── sumber_frame/{images,labels}  # 181 frame + label YOLO (kelas 0 backboard, 1 ball, 2 rim)
├── dataset_raw/metadata.csv      # asal split per frame (blok berurutan, anti-leakage)
├── runs/detect/E1_pretrained, E2_scratch, E3_imgsz640   # results.csv, results.png, args.yaml, confusion matrix, kurva
└── hasil/                        # test_E3.txt, val_per_kelas.txt, contoh_prediksi_test/
```
Bobot (`*.pt`) tidak di-commit. Semua bisa dibuat ulang dengan `colab/yolo_colab.ipynb`.

## Data

Sumber: kamera robot (e-con See3CAM_CU135 di kepala humanoid), **24 September 2026, lab BRAIL, cahaya netral**, di-capture tiap 0,5 detik saat robot berjalan: **181 frame dari satu rekaman**, dianotasi di Roboflow (backboard, ball, rim). Format label YOLO (`kelas cx cy w h`, ter-normalisasi).

**Split (slide 22).** Frame berurutan sangat mirip, sehingga split acak membuat test terlalu bagus. Split dibuat dari **blok frame berurutan** dengan jeda di batas blok (8 frame dibuang), sama dengan tugas klasifikasi:

| Split | Frame | Rentang nomor frame |
|---|---|---|
| train | 124 | 1530-1653 |
| val | 23 | 1658-1680 |
| test | 26 | 1685-1710 |

Augmentasi: hanya augmentasi online bawaan Ultralytics (fliplr 0,5; skala 0,5; HSV; mosaic 1,0), tanpa duplikasi gambar di disk. Gambar yang dipakai adalah ekspor Roboflow 640 x 480 (diregangkan dari 1280 x 720, tanpa letterbox); belum diekspor ulang dari frame asli.

## Eksperimen

Satu perubahan per eksperimen. Nilai val dari epoch terbaik di `results.csv`.

| Eksperimen | Perubahan | Dataset | mAP50 (val) | mAP50-95 (val) | ms/gambar (T4) | Kesimpulan |
|---|---|---|---|---|---|---|
| E1_pretrained | `yolo26n.pt`, imgsz 416 | dataset_v1 | 0,986 | 0,663 | 11,2 | Baseline |
| E2_scratch | `yolo26n.yaml`, tanpa pretrained | dataset_v1 | 0,641 | 0,393 | (arsitektur sama dengan E1, tidak diukur) | Jauh lebih rendah; bobot awal sangat membantu pada data kecil |
| E3_imgsz640 | imgsz 640 (dari E1) | dataset_v1 | 0,993 | 0,699 | 14,7 | +0,036 mAP50-95, namun ±31% lebih lambat; selisih kecil pada val 23 gambar dan 1 seed |

Waktu latih (T4, 30 epoch): E1 64 s, E2 62 s, E3 85 s. ms/gambar = praproses + inferensi + pascaproses, batch 1, setelah warm-up (`scripts/benchmark.py`). Angka mAP val per kelas ada di `hasil/val_per_kelas.txt`.

**Pemilihan model:** E3 (mAP50-95 val tertinggi), dievaluasi **sekali** di test.

## Hasil test (E3, imgsz 640, 26 gambar, 76 instance)

| Kelas | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0,940 | 0,816 | 0,908 | 0,619 |
| backboard | 0,962 | 1,000 | 0,995 | 0,816 |
| ball | 0,921 | **0,449** | 0,735 | 0,400 |
| rim | 0,935 | 1,000 | 0,995 | 0,640 |

## Analisis

- **Pretrained vs scratch.** Pada 124 gambar latih, bobot pretrained (COCO) mengungguli scratch jauh (mAP50-95 0,663 vs 0,393). Scratch belum konvergen dalam 30 epoch.
- **Kurva training** (`runs/detect/*/results.png`). Loss train dan `val/cls_loss` turun stabil, loss val tidak naik (tidak ada tanda overfitting), dan mAP mendatar sekitar epoch 15-20. `val/box_loss` bergelombang karena val hanya 23 gambar.
- **imgsz 640 (E3).** Menaikkan mAP50-95 val sebesar 0,036 dengan biaya waktu sekitar 31%. Dengan 23 gambar val dan satu seed, selisih ini belum bisa dibedakan dari noise.
- **Val vs test.** Test lebih rendah (mAP50 0,908 vs 0,993; mAP50-95 0,619 vs 0,700), hampir seluruhnya akibat kelas `ball` (recall 0,449 vs 1,000). Pada frame test yang diperiksa (1685-1691, `hasil/contoh_prediksi_test/`), bola berada di atas galon dekat kamera dan **tidak terdeteksi**, sementara backboard dan rim terdeteksi. Pada frame train yang diperiksa, bola umumnya di lantai dan jauh. Dari label, bola di test justru sekitar dua kali lebih besar daripada di train, sehingga penyebab yang lebih mungkin adalah pergeseran situasi (posisi dan latar bola) yang hanya muncul di blok test, bukan ukuran. Sampel yang diperiksa kecil, jadi ini dugaan terbaik, bukan pembuktian.
- **Tidak ada penyetelan memakai test.** Hasil test dilaporkan apa adanya.

## Keterbatasan

- Satu sesi rekam (satu hari, satu lokasi, cahaya netral); hasil belum menggambarkan kondisi lain.
- Val 23 dan test 26 gambar; metrik per kelas sangat sensitif. Frame test berurutan dan mirip, sehingga 26 bola bukan 26 kasus independen.
- Satu seed per eksperimen; selisih kecil antar eksperimen tidak dapat disimpulkan.
- Gambar latih berasal dari ekspor Roboflow 640 x 480 (peregangan aspek); pipeline robot memakai 1280 x 720, sehingga praproses di robot harus disamakan atau data diekspor ulang.
- Variasi posisi bola (di galon, dekat, jauh, di udara) belum tercakup oleh data latih.
- ms/gambar diukur di Tesla T4, **bukan Jetson Xavier NX**; belum membuktikan budget latensi robot. Pengukuran TensorRT FP16 di Jetson belum dilakukan.

## Rencana berikutnya

1. Rekam sesi baru dengan variasi posisi bola, cahaya, dan jarak; gunakan sesi terpisah sebagai test.
2. Pertimbangkan `dataset_v2` dengan split yang menyebar tiap fase rekaman; bila dipakai, laporkan sebagai eksperimen terpisah karena dipilih setelah melihat hasil test.
3. Ukur latensi di Jetson (TensorRT FP16) dan bandingkan dengan budget di dokumen desain Pertemuan 3.

## Cara menjalankan ulang

Buka `colab/yolo_colab.ipynb` di Google Colab dengan runtime T4 GPU, atau secara lokal:

```bash
pip install -r requirements.txt
python scripts/build_yolo_dataset.py
yolo detect train data=dataset_v1/data.yaml model=yolo26n.pt epochs=30 imgsz=416 batch=16 seed=0 name=E1_pretrained
```
