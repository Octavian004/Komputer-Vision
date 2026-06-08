"""
==========================================================================
PERCOBAAN 19: PIPELINE FEATURE MATCHING LENGKAP DENGAN EVALUASI
==========================================================================
Program ini membangun pipeline feature matching yang dapat dikonfigurasi:
memilih detektor, deskriptor, matcher, dan verifier. Pipeline diuji
dengan 3 kombinasi (SIFT+FLANN, ORB+BF, AKAZE+BF) pada 5 pasangan
gambar uji. Evaluasi mencakup jumlah matches, inlier ratio, waktu
pemrosesan, precision, recall, dan F1-score.

Konsep yang dipelajari:
- Pipeline feature matching: deteksi → deskripsi → matching → verifikasi
- Konfigurasi pipeline: memilih komponen terbaik untuk tugas tertentu
- Evaluasi kuantitatif: precision, recall, F1-score
- Inlier ratio: proporsi matches yang konsisten secara geometris
- Trade-off kecepatan vs akurasi pada setiap kombinasi
- Rekomendasi kombinasi terbaik berdasarkan skenario penggunaan

Fungsi utama yang dipelajari:
- cv2.SIFT_create()        : Detektor/deskriptor SIFT (akurat)
- cv2.ORB_create()          : Detektor/deskriptor ORB (cepat, binary)
- cv2.AKAZE_create()        : Detektor/deskriptor AKAZE (robust)
- cv2.FlannBasedMatcher()   : FLANN matcher (efisien untuk float)
- cv2.BFMatcher()           : Brute-Force matcher
- cv2.findHomography()      : Verifikasi geometris dengan RANSAC

Hasil: Visualisasi matching per pipeline dan tabel evaluasi komprehensif
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan fitur
import cv2

# Mengimpor NumPy untuk operasi array dan matriks numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan
import time

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan judul percobaan
print("=" * 60)
print("PERCOBAAN 19: PIPELINE FEATURE MATCHING DENGAN EVALUASI")
print("=" * 60)

# ============================================================
# 1. Mendefinisikan Pipeline Feature Matching
# ============================================================

# Menampilkan header bagian
print("\n--- 1. Mendefinisikan Pipeline ---")


# Mendefinisikan fungsi pipeline feature matching yang dapat dikonfigurasi
def pipeline_feature_matching(img1, img2, config):
    """
    Pipeline feature matching lengkap yang dapat dikonfigurasi.
    
    Parameter:
    - img1, img2: gambar input (BGR)
    - config: dictionary berisi konfigurasi pipeline
    
    Return: dictionary berisi hasil matching dan evaluasi
    """
    # Menyiapkan dictionary untuk menyimpan hasil
    hasil = {}

    # Mengkonversi gambar ke grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # --- Tahap 1: Deteksi dan Deskripsi ---
    # Mengukur waktu deteksi
    t_start_det = time.time()

    # Membuat detektor sesuai konfigurasi
    if config['detektor'] == 'SIFT':
        detector = cv2.SIFT_create()
    elif config['detektor'] == 'ORB':
        detector = cv2.ORB_create(nfeatures=1000)
    elif config['detektor'] == 'AKAZE':
        detector = cv2.AKAZE_create()
    else:
        detector = cv2.SIFT_create()

    # Mendeteksi keypoint dan deskriptor pada kedua gambar
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    # Menghitung waktu deteksi
    t_deteksi = time.time() - t_start_det

    # Menyimpan jumlah keypoint
    hasil['kp1'] = len(kp1)
    hasil['kp2'] = len(kp2)
    hasil['waktu_deteksi'] = t_deteksi

    # Memeriksa apakah deskriptor valid
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        hasil['good_matches'] = 0
        hasil['inlier_ratio'] = 0.0
        hasil['waktu_matching'] = 0.0
        hasil['waktu_total'] = t_deteksi
        hasil['matches_obj'] = []
        hasil['kp1_obj'] = kp1
        hasil['kp2_obj'] = kp2
        hasil['true_positive'] = 0
        hasil['false_positive'] = 0
        hasil['precision'] = 0.0
        hasil['recall'] = 0.0
        hasil['f1'] = 0.0
        return hasil

    # --- Tahap 2: Matching ---
    # Mengukur waktu matching
    t_start_match = time.time()

    # Membuat matcher sesuai konfigurasi
    if config['matcher'] == 'FLANN':
        # Menentukan parameter FLANN berdasarkan tipe deskriptor
        if config['detektor'] == 'SIFT':
            index_params = dict(algorithm=1, trees=5)
        else:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # Menentukan tipe norm berdasarkan detektor
        if config['detektor'] == 'SIFT':
            norm = cv2.NORM_L2
        else:
            norm = cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm, crossCheck=False)

    # Melakukan KNN matching
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        hasil['good_matches'] = 0
        hasil['inlier_ratio'] = 0.0
        hasil['waktu_matching'] = 0.0
        hasil['waktu_total'] = t_deteksi
        hasil['matches_obj'] = []
        hasil['kp1_obj'] = kp1
        hasil['kp2_obj'] = kp2
        hasil['true_positive'] = 0
        hasil['false_positive'] = 0
        hasil['precision'] = 0.0
        hasil['recall'] = 0.0
        hasil['f1'] = 0.0
        return hasil

    # Menerapkan ratio test Lowe
    good_matches = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n_m = m_pair
            if m.distance < config.get('ratio', 0.75) * n_m.distance:
                good_matches.append(m)

    # Menghitung waktu matching
    t_matching = time.time() - t_start_match

    # Menyimpan hasil matching
    hasil['good_matches'] = len(good_matches)
    hasil['waktu_matching'] = t_matching
    hasil['matches_obj'] = good_matches
    hasil['kp1_obj'] = kp1
    hasil['kp2_obj'] = kp2

    # --- Tahap 3: Verifikasi Geometris (RANSAC) ---
    # Menginisialisasi variabel verifikasi
    inliers = 0
    true_positive = 0
    false_positive = 0

    # Memeriksa apakah cukup matches untuk verifikasi
    if len(good_matches) >= 4:
        # Mengekstrak titik-titik korespondensi
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Mengestimasi homography dengan RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Menghitung inlier jika homography berhasil
        if mask is not None:
            inliers = int(mask.sum())
            true_positive = inliers
            false_positive = len(good_matches) - inliers
        else:
            true_positive = 0
            false_positive = len(good_matches)
    else:
        true_positive = 0
        false_positive = len(good_matches)

    # Menghitung inlier ratio
    inlier_ratio = inliers / max(len(good_matches), 1)

    # Menghitung precision, recall, dan F1
    precision = true_positive / max(true_positive + false_positive, 1)
    estimated_total_correct = max(min(hasil['kp1'], hasil['kp2']) * 0.3, true_positive)
    recall = true_positive / max(estimated_total_correct, 1)
    recall = min(recall, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    # Menyimpan hasil verifikasi
    hasil['inlier_ratio'] = inlier_ratio
    hasil['true_positive'] = true_positive
    hasil['false_positive'] = false_positive
    hasil['precision'] = precision
    hasil['recall'] = recall
    hasil['f1'] = f1
    hasil['waktu_total'] = t_deteksi + t_matching

    # Mengembalikan hasil pipeline
    return hasil


# Menampilkan informasi pipeline
print("[INFO] Fungsi pipeline_feature_matching() siap digunakan")

# ============================================================
# 2. Mendefinisikan Konfigurasi Pipeline
# ============================================================

# Menampilkan header bagian
print("\n--- 2. Konfigurasi Pipeline ---")

# Mendefinisikan 3 konfigurasi pipeline
konfigurasi_pipeline = {
    'SIFT+FLANN': {
        'detektor': 'SIFT',
        'matcher': 'FLANN',
        'ratio': 0.75,
        'warna': '#4472C4',
    },
    'ORB+BF': {
        'detektor': 'ORB',
        'matcher': 'BF',
        'ratio': 0.75,
        'warna': '#ED7D31',
    },
    'AKAZE+BF': {
        'detektor': 'AKAZE',
        'matcher': 'BF',
        'ratio': 0.75,
        'warna': '#70AD47',
    },
}

# Menampilkan konfigurasi yang akan diuji
for nama, config in konfigurasi_pipeline.items():
    print(f"  {nama}: detektor={config['detektor']}, matcher={config['matcher']}, ratio={config['ratio']}")

# ============================================================
# 3. Mendefinisikan Pasangan Gambar Uji
# ============================================================

# Menampilkan header bagian
print("\n--- 3. Pasangan Gambar Uji ---")

# Mendefinisikan 5 pasangan gambar uji dengan berbagai tingkat kesulitan
pasangan_uji = [
    ('scene_left.jpg', 'scene_right.jpg', 'Overlapping Scene'),
    ('objek_buku.jpg', 'scene_buku.jpg', 'Objek dalam Scene'),
    ('buku_rot0.jpg', 'buku_rot30.jpg', 'Rotasi 30°'),
    ('buku_scale100.jpg', 'buku_scale150.jpg', 'Skala 1.5x'),
    ('pano_left.jpg', 'pano_center.jpg', 'Panorama Berurutan'),
]

# Memuat semua pasangan gambar
pasangan_data = []
for file_a, file_b, label in pasangan_uji:
    # Membaca gambar A
    img_a = cv2.imread(os.path.join(IMAGE_DIR, file_a))

    # Membaca gambar B
    img_b = cv2.imread(os.path.join(IMAGE_DIR, file_b))

    # Memeriksa apakah kedua gambar berhasil dimuat
    if img_a is not None and img_b is not None:
        pasangan_data.append((img_a, img_b, label, file_a, file_b))
        print(f"  [OK] {label}: {file_a} ({img_a.shape}) vs {file_b} ({img_b.shape})")
    else:
        print(f"  [SKIP] {label}: gambar tidak ditemukan")

# Menampilkan jumlah pasangan uji yang berhasil dimuat
print(f"\n[INFO] Total pasangan uji: {len(pasangan_data)}")

# ============================================================
# 4. Menjalankan Pipeline untuk Setiap Kombinasi
# ============================================================

# Menampilkan header bagian
print("\n--- 4. Menjalankan Pipeline ---")

# Menyiapkan dictionary untuk menyimpan semua hasil
semua_hasil = {}

# Melakukan iterasi untuk setiap konfigurasi pipeline
for nama_pipeline, config in konfigurasi_pipeline.items():
    # Menampilkan header pipeline
    print(f"\n  === Pipeline: {nama_pipeline} ===")

    # Menyiapkan list untuk hasil pipeline ini
    semua_hasil[nama_pipeline] = []

    # Melakukan iterasi untuk setiap pasangan uji
    for img_a, img_b, label, file_a, file_b in pasangan_data:
        # Menjalankan pipeline
        hasil = pipeline_feature_matching(img_a, img_b, config)

        # Menambahkan label dan info file
        hasil['label'] = label
        hasil['file_a'] = file_a
        hasil['file_b'] = file_b

        # Menyimpan hasil
        semua_hasil[nama_pipeline].append(hasil)

        # Menampilkan hasil
        print(f"    {label}: matches={hasil['good_matches']}, "
              f"inlier_ratio={hasil['inlier_ratio']:.3f}, "
              f"precision={hasil['precision']:.3f}, "
              f"waktu={hasil['waktu_total']:.3f}s")

# ============================================================
# 5. Visualisasi Matching per Pipeline
# ============================================================

# Menampilkan header bagian
print("\n--- 5. Visualisasi Matching ---")

# --- SIFT + FLANN ---
# Membuat figure untuk hasil SIFT+FLANN
fig, axes = plt.subplots(len(pasangan_data), 1, figsize=(16, 5 * len(pasangan_data)))

# Menghandle kasus pasangan tunggal
if len(pasangan_data) == 1:
    axes = [axes]

# Menampilkan hasil matching SIFT+FLANN
for idx, (hasil, (img_a, img_b, label, file_a, file_b)) in enumerate(
        zip(semua_hasil['SIFT+FLANN'], pasangan_data)):
    # Menggambar matches
    img_matches = cv2.drawMatches(
        img_a, hasil['kp1_obj'], img_b, hasil['kp2_obj'],
        hasil['matches_obj'][:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    axes[idx].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f'{label}: {hasil["good_matches"]} matches, '
                        f'inlier={hasil["inlier_ratio"]:.2f}, '
                        f'P={hasil["precision"]:.2f}, R={hasil["recall"]:.2f}', fontsize=11)
    axes[idx].axis('off')

# Menambahkan judul
fig.suptitle('Pipeline SIFT + FLANN', fontsize=14, fontweight='bold')
plt.tight_layout()

# Menyimpan visualisasi SIFT
output_sift = os.path.join(OUTPUT_DIR, "19_pipeline_sift.png")
plt.savefig(output_sift, dpi=100, bbox_inches='tight')
plt.show()
plt.close()
print(f"[SAVED] {output_sift}")

# --- ORB + BF ---
# Membuat figure untuk hasil ORB+BF
fig, axes = plt.subplots(len(pasangan_data), 1, figsize=(16, 5 * len(pasangan_data)))

# Menghandle kasus pasangan tunggal
if len(pasangan_data) == 1:
    axes = [axes]

# Menampilkan hasil matching ORB+BF
for idx, (hasil, (img_a, img_b, label, file_a, file_b)) in enumerate(
        zip(semua_hasil['ORB+BF'], pasangan_data)):
    img_matches = cv2.drawMatches(
        img_a, hasil['kp1_obj'], img_b, hasil['kp2_obj'],
        hasil['matches_obj'][:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    axes[idx].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f'{label}: {hasil["good_matches"]} matches, '
                        f'inlier={hasil["inlier_ratio"]:.2f}, '
                        f'P={hasil["precision"]:.2f}, R={hasil["recall"]:.2f}', fontsize=11)
    axes[idx].axis('off')

# Menambahkan judul
fig.suptitle('Pipeline ORB + BF', fontsize=14, fontweight='bold')
plt.tight_layout()

# Menyimpan visualisasi ORB
output_orb = os.path.join(OUTPUT_DIR, "19_pipeline_orb.png")
plt.savefig(output_orb, dpi=100, bbox_inches='tight')
plt.show()
plt.close()
print(f"[SAVED] {output_orb}")

# --- AKAZE + BF ---
# Membuat figure untuk hasil AKAZE+BF
fig, axes = plt.subplots(len(pasangan_data), 1, figsize=(16, 5 * len(pasangan_data)))

# Menghandle kasus pasangan tunggal
if len(pasangan_data) == 1:
    axes = [axes]

# Menampilkan hasil matching AKAZE+BF
for idx, (hasil, (img_a, img_b, label, file_a, file_b)) in enumerate(
        zip(semua_hasil['AKAZE+BF'], pasangan_data)):
    img_matches = cv2.drawMatches(
        img_a, hasil['kp1_obj'], img_b, hasil['kp2_obj'],
        hasil['matches_obj'][:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    axes[idx].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f'{label}: {hasil["good_matches"]} matches, '
                        f'inlier={hasil["inlier_ratio"]:.2f}, '
                        f'P={hasil["precision"]:.2f}, R={hasil["recall"]:.2f}', fontsize=11)
    axes[idx].axis('off')

# Menambahkan judul
fig.suptitle('Pipeline AKAZE + BF', fontsize=14, fontweight='bold')
plt.tight_layout()

# Menyimpan visualisasi AKAZE
output_akaze = os.path.join(OUTPUT_DIR, "19_pipeline_akaze.png")
plt.savefig(output_akaze, dpi=100, bbox_inches='tight')
plt.show()
plt.close()
print(f"[SAVED] {output_akaze}")

# ============================================================
# 6. Tabel Evaluasi Komprehensif
# ============================================================

# Menampilkan header bagian
print("\n--- 6. Tabel Evaluasi Komprehensif ---")

# Menyiapkan data tabel
# Header: Pipeline | Pasangan | Matches | Inlier % | TP | FP | Precision | Recall | F1 | Waktu
header_tabel = ['Pipeline', 'Pasangan', 'Matches', 'Inlier%', 'TP', 'FP',
                'Precision', 'Recall', 'F1', 'Waktu(s)']

# Menyiapkan data baris tabel
baris_tabel = []

# Melakukan iterasi untuk mengumpulkan semua data
for nama_pipeline in konfigurasi_pipeline:
    for hasil in semua_hasil[nama_pipeline]:
        baris_tabel.append([
            nama_pipeline,
            hasil['label'][:18],
            str(hasil['good_matches']),
            f"{hasil['inlier_ratio']:.2f}",
            str(hasil['true_positive']),
            str(hasil['false_positive']),
            f"{hasil['precision']:.3f}",
            f"{hasil['recall']:.3f}",
            f"{hasil['f1']:.3f}",
            f"{hasil['waktu_total']:.3f}",
        ])

# Membuat figure untuk tabel evaluasi (perlu ukuran besar)
fig, ax = plt.subplots(figsize=(22, max(8, len(baris_tabel) * 0.45 + 2)))

# Menyembunyikan axes
ax.axis('off')

# Membuat tabel
tabel = ax.table(cellText=baris_tabel, colLabels=header_tabel,
                 loc='center', cellLoc='center')

# Mengatur ukuran font
tabel.auto_set_font_size(False)
tabel.set_fontsize(8)

# Mengatur tinggi baris
tabel.scale(1, 1.6)

# Mewarnai header
for j in range(len(header_tabel)):
    tabel[0, j].set_facecolor('#4472C4')
    tabel[0, j].set_text_props(color='white', fontweight='bold')

# Menentukan warna per pipeline
warna_pipeline = {
    'SIFT+FLANN': '#D6E4F0',
    'ORB+BF': '#FDE8D0',
    'AKAZE+BF': '#E2EFDA',
}

# Mewarnai baris berdasarkan pipeline
for i, baris in enumerate(baris_tabel):
    warna = warna_pipeline.get(baris[0], '#FFFFFF')
    for j in range(len(header_tabel)):
        tabel[i + 1, j].set_facecolor(warna)

# Menambahkan judul
ax.set_title('Tabel Evaluasi Pipeline Feature Matching\n'
             '(Perbandingan SIFT+FLANN, ORB+BF, AKAZE+BF)',
             fontsize=14, fontweight='bold', pad=20)

# Menghitung ringkasan rata-rata per pipeline
print("\nRingkasan rata-rata per pipeline:")
for nama_pipeline in konfigurasi_pipeline:
    hasil_list = semua_hasil[nama_pipeline]
    rata_matches = np.mean([h['good_matches'] for h in hasil_list])
    rata_inlier = np.mean([h['inlier_ratio'] for h in hasil_list])
    rata_precision = np.mean([h['precision'] for h in hasil_list])
    rata_recall = np.mean([h['recall'] for h in hasil_list])
    rata_f1 = np.mean([h['f1'] for h in hasil_list])
    rata_waktu = np.mean([h['waktu_total'] for h in hasil_list])
    print(f"  {nama_pipeline}: matches={rata_matches:.1f}, inlier={rata_inlier:.3f}, "
          f"P={rata_precision:.3f}, R={rata_recall:.3f}, F1={rata_f1:.3f}, "
          f"waktu={rata_waktu:.3f}s")

# Menambahkan ringkasan di bawah tabel
ringkasan_text = []
for nama_pipeline in konfigurasi_pipeline:
    hasil_list = semua_hasil[nama_pipeline]
    rata_f1 = np.mean([h['f1'] for h in hasil_list])
    rata_waktu = np.mean([h['waktu_total'] for h in hasil_list])
    ringkasan_text.append(f"{nama_pipeline}: F1={rata_f1:.3f}, Waktu={rata_waktu:.3f}s")

# Menampilkan ringkasan di figure
ax.text(0.5, -0.02, ' | '.join(ringkasan_text),
        transform=ax.transAxes, ha='center', fontsize=10, style='italic')

# Mengatur layout
plt.tight_layout()

# Menyimpan tabel evaluasi
output_eval = os.path.join(OUTPUT_DIR, "19_evaluation_table.png")
plt.savefig(output_eval, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"[SAVED] {output_eval}")

# ============================================================
# 7. Rekomendasi Kombinasi Terbaik
# ============================================================

# Menampilkan header bagian
print("\n--- 7. Rekomendasi ---")

# Mencari pipeline dengan F1 tertinggi
f1_per_pipeline = {}
for nama in konfigurasi_pipeline:
    f1_per_pipeline[nama] = np.mean([h['f1'] for h in semua_hasil[nama]])

# Mencari pipeline dengan F1 tertinggi
best_f1 = max(f1_per_pipeline, key=f1_per_pipeline.get)
print(f"[REKOMENDASI] Akurasi terbaik (F1 tertinggi): {best_f1} (F1={f1_per_pipeline[best_f1]:.3f})")

# Mencari pipeline tercepat
waktu_per_pipeline = {}
for nama in konfigurasi_pipeline:
    waktu_per_pipeline[nama] = np.mean([h['waktu_total'] for h in semua_hasil[nama]])

# Mencari pipeline tercepat
best_speed = min(waktu_per_pipeline, key=waktu_per_pipeline.get)
print(f"[REKOMENDASI] Kecepatan terbaik: {best_speed} ({waktu_per_pipeline[best_speed]:.3f}s)")

# Menampilkan rekomendasi per skenario
print("\nRekomendasi per skenario:")
print(f"  - Real-time application : ORB+BF (cepat, efisien memori)")
print(f"  - High accuracy needed  : SIFT+FLANN (akurat, lebih lambat)")
print(f"  - Balanced performance  : AKAZE+BF (keseimbangan kecepatan & akurasi)")

# ============================================================
# 8. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 19: PIPELINE FEATURE MATCHING DENGAN EVALUASI")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan pipeline
print("1. Pipeline feature matching terdiri dari 3 tahap utama:")
print("   deteksi/deskripsi → matching → verifikasi geometris.")

# Menampilkan hasil perbandingan
print("2. SIFT+FLANN: akurasi tinggi, waktu pemrosesan lebih lama,")
print("   cocok untuk aplikasi offline yang memerlukan presisi.")

# Menampilkan tentang ORB
print("3. ORB+BF: tercepat, akurasi lebih rendah pada kasus sulit,")
print("   cocok untuk aplikasi real-time dengan resource terbatas.")

# Menampilkan tentang AKAZE
print("4. AKAZE+BF: keseimbangan antara kecepatan dan akurasi,")
print("   cocok untuk sebagian besar aplikasi umum.")

# Menampilkan tentang evaluasi
print("5. Evaluasi kuantitatif (precision, recall, F1) penting")
print("   untuk membandingkan secara objektif antar metode.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 19_pipeline_sift.png")
print("  - 19_pipeline_orb.png")
print("  - 19_pipeline_akaze.png")
print("  - 19_evaluation_table.png")

# Menampilkan garis penutup
print("=" * 60)
