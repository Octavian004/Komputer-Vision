# Project: Document Scanner (Modul 2)

## Deskripsi
Proyek ini adalah implementasi sistem **Document Scanner** sederhana yang memenuhi kriteria **Soal 1** dari Modul 2.

Fitur utama:
- Deteksi kontur dokumen otomatis (perspektif, warping)
- Koreksi distorsi lensa (jika tersedia data kalibrasi)
- Gamma correction adaptif untuk memperbaiki pencahayaan
- Batch processing untuk seluruh folder gambar
- Hasil di-export dalam ukuran A4 (300 dpi) dan thumbnail preview

## Konsep yang diintegrasikan (minimal 10 dari 20)
1. Transformasi perspektif (warp perspective)
2. Deteksi kontur / edge detection
3. Koreksi distorsi lensa (undistort)
4. Gamma correction (intensitas)
5. Resize / interpolate (cv2.resize)
6. Crop / ROI
7. Morphological operations (opening/closing) untuk cleaning
8. Image pyramid (opsional: untuk deteksi multi-skala)
9. Remapping (cv2.remap) via undistort
10. Pengaturan file input/output batch

## Struktur Project
```
project_document_scanner/
  ├─ configs/
  │    └─ calib_camera.json (opsional)
  ├─ input/             # taruh foto dokumen di sini (jpg/png)
  ├─ output/            # akan terisi hasil scan
  ├─ scanner.py         # entrypoint utama
  ├─ utils.py           # helper fungsi (deteksi dokumen, gamma, dll)
  └─ README.md
```

## Cara Pakai
1. Siapkan folder `input/` dan isi dengan foto dokumen.
2. (Opsional) Letakkan file kalibrasi kamera (JSON) di `configs/calib_camera.json`.
3. Jalankan:

```bash
python scanner.py
```

> Jika deteksi kontur gagal, skrip akan tetap menghasilkan output dengan menggunakan seluruh gambar sebagai fallback.

> Jika ingin menggunakan folder khusus, tambahkan argumen:
>
> ```bash
> python scanner.py --input my_input --output my_output
> ```

Opsi tambahan:
- `--calib configs/calib_camera.json` : koreksi distorsi
- `--dpi 300` : output A4 di 300 dpi (default)
- `--no-gamma` : matikan auto gamma correction

## Contoh Output
- `output/scan_0001.png` (A4 300dpi)
- `output/scan_0001_thumb.png` (thumbnail 512px)
- `output/report.csv` (ringkasan proses)

---

## Catatan
Pastikan OpenCV (`opencv-python`) dan NumPy (`numpy`) terpasang.

```bash
pip install -r requirements.txt
```
