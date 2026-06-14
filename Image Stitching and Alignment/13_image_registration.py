"""
==========================================================================
PERCOBAAN 13: IMAGE REGISTRATION (PENYELARASAN GAMBAR)
==========================================================================
Program ini mengimplementasikan berbagai metode image registration:
feature-based, dan ECC (Enhanced Correlation Coefficient)-based.
Image registration adalah proses menyelaraskan dua gambar agar
piksel-piksel yang berkorespondensi berada pada posisi yang sama.

Konsep yang dipelajari:
- Image registration: konsep dan aplikasi
- Feature-based registration (SIFT → homography → warp)
- ECC (Enhanced Correlation Coefficient) registration
- Mode transformasi: translation, Euclidean, affine, homography
- Evaluasi kualitas alignment (pixel difference, overlay)
- Checkerboard visualization untuk visual inspection

Fungsi utama yang dipelajari:
- cv2.findTransformECC() : Estimasi transformasi ECC (iteratif)
- cv2.MOTION_TRANSLATION : Mode registrasi translasi (2 DOF)
- cv2.MOTION_EUCLIDEAN   : Mode registrasi rigid (3 DOF: rotasi+translasi)
- cv2.MOTION_AFFINE      : Mode registrasi affine (6 DOF)
- cv2.MOTION_HOMOGRAPHY  : Mode registrasi perspektif (8 DOF)
- cv2.warpAffine()       : Warping dengan transformasi affine
- cv2.warpPerspective()  : Warping dengan transformasi perspektif
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan computer vision
import cv2

# Mengimpor library NumPy untuk operasi array dan matriks
import numpy as np

# Mengimpor library os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi dan grid perbandingan
import matplotlib.pyplot as plt

# Mengimpor modul time untuk mengukur waktu eksekusi
import time

# ============================================================
# SETUP PATH DIREKTORI
# ============================================================

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HEADER PROGRAM
# ============================================================
print("=" * 65)
print("PERCOBAAN 13: IMAGE REGISTRATION (PENYELARASAN GAMBAR)")
print("=" * 65)


# ============================================================
# FUNGSI HELPER
# ============================================================

def feature_based_registration(img_src, img_ref, label=""):
    """
    Melakukan image registration berbasis fitur.
    Pipeline: SIFT → FLANN → Lowe's ratio test → RANSAC → warp.

    Parameter:
    - img_src : Gambar yang akan diselaraskan (source)
    - img_ref : Gambar referensi (target alignment)
    - label   : Label untuk logging

    Returns:
    - aligned : Gambar source yang sudah diselaraskan ke referensi
    - H       : Matriks homography 3x3
    - waktu   : Waktu eksekusi (detik)
    """
    waktu_start = time.time()

    # Mengkonversi ke grayscale untuk deteksi fitur
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # Membuat detektor SIFT dan mendeteksi fitur
    sift = cv2.SIFT_create()
    kp_src, desc_src = sift.detectAndCompute(gray_src, None)
    kp_ref, desc_ref = sift.detectAndCompute(gray_ref, None)

    # Validasi deskriptor
    if desc_src is None or desc_ref is None or len(desc_src) < 4 or len(desc_ref) < 4:
        if label:
            print(f"    {label}: Tidak cukup fitur")
        return img_src.copy(), np.eye(3, dtype=np.float64), 0

    # Mencocokkan fitur menggunakan FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_src, desc_ref, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        if label:
            print(f"    {label}: Hanya {len(good)} matches")
        return img_src.copy(), np.eye(3, dtype=np.float64), 0

    # Mengekstrak titik korespondensi
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Mengestimasi homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Melakukan warping gambar source ke koordinat referensi
    h_ref, w_ref = img_ref.shape[:2]
    aligned = cv2.warpPerspective(img_src, H, (w_ref, h_ref))

    waktu = time.time() - waktu_start

    if label:
        inliers = int(mask.ravel().sum()) if mask is not None else 0
        print(f"    {label}: {len(good)} matches, {inliers} inliers, {waktu*1000:.1f}ms")

    return aligned, H, waktu


def ecc_registration(img_src, img_ref, mode, label=""):
    """
    Melakukan image registration menggunakan ECC (Enhanced Correlation
    Coefficient). ECC adalah metode intensitas-based yang secara iteratif
    mencari transformasi optimal.

    Parameter:
    - img_src : Gambar yang akan diselaraskan
    - img_ref : Gambar referensi
    - mode    : Mode transformasi (MOTION_TRANSLATION, dll.)
    - label   : Label logging

    Returns:
    - aligned  : Gambar yang sudah diselaraskan
    - warp_mat : Matriks transformasi
    - waktu    : Waktu eksekusi
    - cc       : Koefisien korelasi akhir
    """
    waktu_start = time.time()

    # Mengkonversi ke grayscale (ECC bekerja dengan intensitas)
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # Menginisialisasi matriks warp sesuai mode transformasi
    if mode == cv2.MOTION_HOMOGRAPHY:
        # Untuk homography: matriks 3x3 (identitas sebagai initial guess)
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        # Untuk translation/Euclidean/affine: matriks 2x3
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Mendefinisikan kriteria terminasi ECC
    # maxCount=1000 iterasi, epsilon=1e-6 (konvergensi)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)

    try:
        # Menjalankan ECC untuk menemukan transformasi optimal
        # cc = Enhanced Correlation Coefficient (nilai tertinggi = alignment terbaik)
        cc, warp_matrix = cv2.findTransformECC(
            gray_ref,       # Template (referensi)
            gray_src,       # Input (yang akan ditransformasi)
            warp_matrix,    # Inisialisasi transformasi
            mode,           # Mode transformasi
            criteria        # Kriteria terminasi
        )

        # Melakukan warping berdasarkan mode
        h_ref, w_ref = img_ref.shape[:2]
        if mode == cv2.MOTION_HOMOGRAPHY:
            # warpPerspective untuk transformasi homography
            aligned = cv2.warpPerspective(
                img_src, warp_matrix, (w_ref, h_ref),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        else:
            # warpAffine untuk translasi/Euclidean/affine
            aligned = cv2.warpAffine(
                img_src, warp_matrix, (w_ref, h_ref),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )

        waktu = time.time() - waktu_start

        if label:
            print(f"    {label}: ECC={cc:.6f}, waktu={waktu*1000:.1f}ms")

        return aligned, warp_matrix, waktu, cc

    except cv2.error as e:
        waktu = time.time() - waktu_start
        if label:
            print(f"    {label}: ECC gagal konvergen ({e})")
        return img_src.copy(), warp_matrix, waktu, 0.0


def hitung_alignment_error(img_aligned, img_ref):
    """
    Menghitung error alignment sebagai perbedaan piksel antara
    gambar yang sudah diselaraskan dan gambar referensi.

    Parameter:
    - img_aligned : Gambar yang sudah diselaraskan
    - img_ref     : Gambar referensi

    Returns:
    - mean_err : Rata-rata perbedaan intensitas per piksel
    - max_err  : Perbedaan maksimum
    - pct_good : Persentase piksel dengan perbedaan < 10
    """
    # Mengkonversi ke grayscale
    gray_a = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_r = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Membuat mask untuk area yang valid (bukan hitam) di kedua gambar
    mask_a = gray_a > 5
    mask_r = gray_r > 5
    mask_valid = mask_a & mask_r

    if np.sum(mask_valid) == 0:
        return float('inf'), float('inf'), 0.0

    # Menghitung perbedaan absolut hanya pada area valid
    diff = np.abs(gray_a - gray_r)
    diff_valid = diff[mask_valid]

    mean_err = np.mean(diff_valid)
    max_err = np.max(diff_valid)
    pct_good = np.sum(diff_valid < 10) / len(diff_valid) * 100

    return mean_err, max_err, pct_good


def buat_checkerboard_overlay(img1, img2, block_size=50):
    """
    Membuat visualisasi checkerboard: menampilkan blok-blok bergantian
    dari dua gambar. Ini cara efektif untuk memeriksa kualitas alignment.

    Parameter:
    - img1       : Gambar pertama
    - img2       : Gambar kedua
    - block_size : Ukuran blok checkerboard (piksel)

    Returns:
    - checker    : Gambar checkerboard overlay
    """
    # Memastikan kedua gambar memiliki ukuran yang sama
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_crop = img1[:h, :w].copy()
    img2_crop = img2[:h, :w].copy()

    # Membuat mask checkerboard
    checker = img1_crop.copy()
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            # Blok genap: dari gambar 1, blok ganjil: dari gambar 2
            block_row = y // block_size
            block_col = x // block_size
            if (block_row + block_col) % 2 == 1:
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                checker[y:y_end, x:x_end] = img2_crop[y:y_end, x:x_end]

    return checker


def buat_overlay_blend(img1, img2, alpha=0.5):
    """
    Membuat overlay campuran (blend) dari dua gambar.
    Area yang tumpang tindih akan terlihat transparan.

    Parameter:
    - img1  : Gambar pertama
    - img2  : Gambar kedua
    - alpha : Bobot gambar pertama (0..1)

    Returns:
    - blended : Gambar overlay
    """
    # Memastikan ukuran sama
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_crop = img1[:h, :w]
    img2_crop = img2[:h, :w]

    # Melakukan alpha blending
    blended = cv2.addWeighted(img1_crop, alpha, img2_crop, 1 - alpha, 0)

    return blended


# ============================================================
# LANGKAH 1: Memuat Gambar untuk Registrasi
# ============================================================
print("\n[LANGKAH 1] Memuat gambar alignment test...")

# Memuat gambar referensi (original)
img_original = cv2.imread(os.path.join(IMAGE_DIR, "alignment_original.jpg"))

# Memuat gambar yang ditranslasi
img_translated = cv2.imread(os.path.join(IMAGE_DIR, "alignment_translated.jpg"))

# Memuat gambar yang dirotasi
img_rotated = cv2.imread(os.path.join(IMAGE_DIR, "alignment_rotated.jpg"))

# Memuat gambar yang ditransformasi perspektif
img_perspective = cv2.imread(os.path.join(IMAGE_DIR, "alignment_perspective.jpg"))

# Memvalidasi gambar
gambar_valid = True
for nama, img in [("original", img_original), ("translated", img_translated),
                   ("rotated", img_rotated), ("perspective", img_perspective)]:
    if img is None:
        print(f"  [ERROR] {nama} tidak ditemukan!")
        gambar_valid = False
    else:
        print(f"  {nama:>12}: {img.shape[1]}x{img.shape[0]}")

if not gambar_valid:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()


# ============================================================
# LANGKAH 2: Method 1 - Feature-Based Registration
# ============================================================
print("\n[LANGKAH 2] Method 1: Feature-based registration (SIFT → homography)...")

# Menerapkan pada gambar yang ditranslasi
aligned_feat_trans, H_feat_trans, t_feat_trans = feature_based_registration(
    img_translated, img_original, "Feature-based (translated)"
)

# Menerapkan pada gambar yang dirotasi
aligned_feat_rot, H_feat_rot, t_feat_rot = feature_based_registration(
    img_rotated, img_original, "Feature-based (rotated)"
)

# Menerapkan pada gambar perspektif
aligned_feat_persp, H_feat_persp, t_feat_persp = feature_based_registration(
    img_perspective, img_original, "Feature-based (perspective)"
)

# Menyimpan hasil feature-based registration
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_feat_aligned_translated.jpg"), aligned_feat_trans)
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_feat_aligned_rotated.jpg"), aligned_feat_rot)
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_feat_aligned_perspective.jpg"), aligned_feat_persp)
print("  [OK] Hasil feature-based registration disimpan.")


# ============================================================
# LANGKAH 3: Method 2 - ECC Translation Registration
# ============================================================
print("\n[LANGKAH 3] Method 2: ECC Translation registration...")

# MOTION_TRANSLATION: hanya translasi (2 DOF: tx, ty)
aligned_ecc_trans, mat_ecc_trans, t_ecc_trans, cc_ecc_trans = ecc_registration(
    img_translated, img_original, cv2.MOTION_TRANSLATION, "ECC Translation"
)

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_ecc_translation.jpg"), aligned_ecc_trans)

# Menampilkan matriks transformasi
print(f"  Matriks Translation:")
print(f"    tx = {mat_ecc_trans[0, 2]:.4f}, ty = {mat_ecc_trans[1, 2]:.4f}")


# ============================================================
# LANGKAH 4: Method 3 - ECC Euclidean Registration
# ============================================================
print("\n[LANGKAH 4] Method 3: ECC Euclidean registration...")

# MOTION_EUCLIDEAN: rotasi + translasi (3 DOF: theta, tx, ty)
aligned_ecc_eucl, mat_ecc_eucl, t_ecc_eucl, cc_ecc_eucl = ecc_registration(
    img_rotated, img_original, cv2.MOTION_EUCLIDEAN, "ECC Euclidean (rotated)"
)

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_ecc_euclidean.jpg"), aligned_ecc_eucl)

# Menampilkan parameter transformasi
theta = np.arctan2(mat_ecc_eucl[1, 0], mat_ecc_eucl[0, 0])
print(f"  Matriks Euclidean:")
print(f"    Rotasi = {np.degrees(theta):.4f}°")
print(f"    tx = {mat_ecc_eucl[0, 2]:.4f}, ty = {mat_ecc_eucl[1, 2]:.4f}")


# ============================================================
# LANGKAH 5: Method 4 - ECC Affine Registration
# ============================================================
print("\n[LANGKAH 5] Method 4: ECC Affine registration...")

# MOTION_AFFINE: transformasi affine (6 DOF)
aligned_ecc_aff_trans, mat_ecc_aff_t, t_ecc_aff_t, cc_ecc_aff_t = ecc_registration(
    img_translated, img_original, cv2.MOTION_AFFINE, "ECC Affine (translated)"
)

aligned_ecc_aff_rot, mat_ecc_aff_r, t_ecc_aff_r, cc_ecc_aff_r = ecc_registration(
    img_rotated, img_original, cv2.MOTION_AFFINE, "ECC Affine (rotated)"
)

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_ecc_affine_translated.jpg"), aligned_ecc_aff_trans)
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_ecc_affine_rotated.jpg"), aligned_ecc_aff_rot)


# ============================================================
# LANGKAH 6: Method 5 - ECC Homography Registration
# ============================================================
print("\n[LANGKAH 6] Method 5: ECC Homography registration...")

# MOTION_HOMOGRAPHY: perspektif penuh (8 DOF)
aligned_ecc_homo, mat_ecc_homo, t_ecc_homo, cc_ecc_homo = ecc_registration(
    img_perspective, img_original, cv2.MOTION_HOMOGRAPHY, "ECC Homography (perspective)"
)

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_ecc_homography.jpg"), aligned_ecc_homo)


# ============================================================
# LANGKAH 7: Perbandingan Semua Metode (Overlay Original + Aligned)
# ============================================================
print("\n[LANGKAH 7] Membuat overlay perbandingan sebelum/sesudah alignment...")

# Overlay sebelum alignment (source vs reference)
overlay_before_trans = buat_overlay_blend(img_translated, img_original, 0.5)
overlay_before_rot = buat_overlay_blend(img_rotated, img_original, 0.5)
overlay_before_persp = buat_overlay_blend(img_perspective, img_original, 0.5)

# Overlay sesudah alignment (aligned vs reference)
overlay_feat_trans = buat_overlay_blend(aligned_feat_trans, img_original, 0.5)
overlay_ecc_trans = buat_overlay_blend(aligned_ecc_trans, img_original, 0.5)
overlay_feat_rot = buat_overlay_blend(aligned_feat_rot, img_original, 0.5)
overlay_ecc_eucl = buat_overlay_blend(aligned_ecc_eucl, img_original, 0.5)
overlay_ecc_homo = buat_overlay_blend(aligned_ecc_homo, img_original, 0.5)

# Menyimpan overlay
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_overlay_before_translated.jpg"), overlay_before_trans)
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_overlay_feat_translated.jpg"), overlay_feat_trans)
cv2.imwrite(os.path.join(OUTPUT_DIR, "13_overlay_ecc_translated.jpg"), overlay_ecc_trans)
print("  [OK] Overlay perbandingan disimpan.")


# ============================================================
# LANGKAH 8: Menghitung Alignment Error untuk Semua Metode
# ============================================================
print("\n[LANGKAH 8] Menghitung alignment error (piksel) per metode...")

# Dictionary untuk menyimpan semua hasil
all_results = {}

# --- Gambar Translated ---
# Sebelum alignment
err_before_t = hitung_alignment_error(img_translated, img_original)
all_results["Trans - Before"] = {"mean": err_before_t[0], "max": err_before_t[1],
                                  "pct_good": err_before_t[2], "waktu": 0, "cc": 0}

# Feature-based
err_feat_t = hitung_alignment_error(aligned_feat_trans, img_original)
all_results["Trans - Feature"] = {"mean": err_feat_t[0], "max": err_feat_t[1],
                                   "pct_good": err_feat_t[2], "waktu": t_feat_trans * 1000,
                                   "cc": 0}

# ECC Translation
err_ecc_t = hitung_alignment_error(aligned_ecc_trans, img_original)
all_results["Trans - ECC Trans"] = {"mean": err_ecc_t[0], "max": err_ecc_t[1],
                                     "pct_good": err_ecc_t[2], "waktu": t_ecc_trans * 1000,
                                     "cc": cc_ecc_trans}

# ECC Affine
err_ecc_at = hitung_alignment_error(aligned_ecc_aff_trans, img_original)
all_results["Trans - ECC Affine"] = {"mean": err_ecc_at[0], "max": err_ecc_at[1],
                                      "pct_good": err_ecc_at[2], "waktu": t_ecc_aff_t * 1000,
                                      "cc": cc_ecc_aff_t}

# --- Gambar Rotated ---
err_before_r = hitung_alignment_error(img_rotated, img_original)
all_results["Rot - Before"] = {"mean": err_before_r[0], "max": err_before_r[1],
                                "pct_good": err_before_r[2], "waktu": 0, "cc": 0}

err_feat_r = hitung_alignment_error(aligned_feat_rot, img_original)
all_results["Rot - Feature"] = {"mean": err_feat_r[0], "max": err_feat_r[1],
                                 "pct_good": err_feat_r[2], "waktu": t_feat_rot * 1000,
                                 "cc": 0}

err_ecc_er = hitung_alignment_error(aligned_ecc_eucl, img_original)
all_results["Rot - ECC Eucl"] = {"mean": err_ecc_er[0], "max": err_ecc_er[1],
                                  "pct_good": err_ecc_er[2], "waktu": t_ecc_eucl * 1000,
                                  "cc": cc_ecc_eucl}

err_ecc_ar = hitung_alignment_error(aligned_ecc_aff_rot, img_original)
all_results["Rot - ECC Affine"] = {"mean": err_ecc_ar[0], "max": err_ecc_ar[1],
                                    "pct_good": err_ecc_ar[2], "waktu": t_ecc_aff_r * 1000,
                                    "cc": cc_ecc_aff_r}

# --- Gambar Perspective ---
err_before_p = hitung_alignment_error(img_perspective, img_original)
all_results["Persp - Before"] = {"mean": err_before_p[0], "max": err_before_p[1],
                                  "pct_good": err_before_p[2], "waktu": 0, "cc": 0}

err_feat_p = hitung_alignment_error(aligned_feat_persp, img_original)
all_results["Persp - Feature"] = {"mean": err_feat_p[0], "max": err_feat_p[1],
                                   "pct_good": err_feat_p[2], "waktu": t_feat_persp * 1000,
                                   "cc": 0}

err_ecc_h = hitung_alignment_error(aligned_ecc_homo, img_original)
all_results["Persp - ECC Homo"] = {"mean": err_ecc_h[0], "max": err_ecc_h[1],
                                    "pct_good": err_ecc_h[2], "waktu": t_ecc_homo * 1000,
                                    "cc": cc_ecc_homo}

# Menampilkan tabel perbandingan
print(f"\n  {'Metode':<22} | {'Mean Err':>10} | {'Max Err':>10} | {'%<10px':>8} | {'Waktu':>8} | {'ECC':>8}")
print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
for nama, data in all_results.items():
    print(f"  {nama:<22} | {data['mean']:>8.2f}px | {data['max']:>8.1f}px | "
          f"{data['pct_good']:>7.1f}% | {data['waktu']:>6.1f}ms | "
          f"{data['cc']:>8.4f}" if data['cc'] > 0 else
          f"  {nama:<22} | {data['mean']:>8.2f}px | {data['max']:>8.1f}px | "
          f"{data['pct_good']:>7.1f}% | {data['waktu']:>6.1f}ms | {'N/A':>8}")


# ============================================================
# LANGKAH 9: Test pada Gambar Rotated (Semua Metode)
# ============================================================
print("\n[LANGKAH 9] Test semua metode ECC pada gambar rotated...")

# Mencoba semua mode ECC pada gambar yang dirotasi
mode_names = {
    cv2.MOTION_TRANSLATION: "Translation",
    cv2.MOTION_EUCLIDEAN: "Euclidean",
    cv2.MOTION_AFFINE: "Affine",
    cv2.MOTION_HOMOGRAPHY: "Homography"
}

ecc_rot_results = {}
for mode, mode_nama in mode_names.items():
    aligned_r, mat_r, t_r, cc_r = ecc_registration(
        img_rotated, img_original, mode, f"ECC {mode_nama} (rot)"
    )
    err_r = hitung_alignment_error(aligned_r, img_original)
    ecc_rot_results[mode_nama] = {
        "aligned": aligned_r,
        "mean_err": err_r[0],
        "pct_good": err_r[2],
        "waktu": t_r * 1000,
        "cc": cc_r
    }

# Menampilkan perbandingan mode ECC pada gambar rotated
print(f"\n  Perbandingan ECC Modes pada Gambar Rotated:")
print(f"  {'Mode':<15} | {'Mean Err':>10} | {'%<10px':>8} | {'ECC':>10} | {'Waktu':>8}")
print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")
for mode_nama, data in ecc_rot_results.items():
    print(f"  {mode_nama:<15} | {data['mean_err']:>8.2f}px | {data['pct_good']:>7.1f}% | "
          f"{data['cc']:>10.6f} | {data['waktu']:>6.1f}ms")


# ============================================================
# LANGKAH 10: Test pada Gambar Perspective (Semua Metode)
# ============================================================
print("\n[LANGKAH 10] Test semua metode ECC pada gambar perspective...")

ecc_persp_results = {}
for mode, mode_nama in mode_names.items():
    aligned_p, mat_p, t_p, cc_p = ecc_registration(
        img_perspective, img_original, mode, f"ECC {mode_nama} (persp)"
    )
    err_p = hitung_alignment_error(aligned_p, img_original)
    ecc_persp_results[mode_nama] = {
        "aligned": aligned_p,
        "mean_err": err_p[0],
        "pct_good": err_p[2],
        "waktu": t_p * 1000,
        "cc": cc_p
    }

# Menampilkan perbandingan
print(f"\n  Perbandingan ECC Modes pada Gambar Perspective:")
print(f"  {'Mode':<15} | {'Mean Err':>10} | {'%<10px':>8} | {'ECC':>10} | {'Waktu':>8}")
print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")
for mode_nama, data in ecc_persp_results.items():
    print(f"  {mode_nama:<15} | {data['mean_err']:>8.2f}px | {data['pct_good']:>7.1f}% | "
          f"{data['cc']:>10.6f} | {data['waktu']:>6.1f}ms")


# ============================================================
# LANGKAH 11: Membuat Visualisasi Overlay (Before/After, Checkerboard)
# ============================================================
print("\n[LANGKAH 11] Membuat visualisasi overlay dan checkerboard...")

# --- Grid 1: Translated image registration ---
fig1, axes1 = plt.subplots(2, 4, figsize=(22, 10))

# Baris 1 judul: Before dan After
axes1[0, 0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title("Referensi\n(Original)", fontsize=10)
axes1[0, 0].axis("off")

axes1[0, 1].imshow(cv2.cvtColor(overlay_before_trans, cv2.COLOR_BGR2RGB))
axes1[0, 1].set_title(f"Before Alignment\nmean={err_before_t[0]:.2f}px", fontsize=10)
axes1[0, 1].axis("off")

axes1[0, 2].imshow(cv2.cvtColor(overlay_feat_trans, cv2.COLOR_BGR2RGB))
axes1[0, 2].set_title(f"Feature-Based\nmean={err_feat_t[0]:.2f}px", fontsize=10)
axes1[0, 2].axis("off")

axes1[0, 3].imshow(cv2.cvtColor(overlay_ecc_trans, cv2.COLOR_BGR2RGB))
axes1[0, 3].set_title(f"ECC Translation\nmean={err_ecc_t[0]:.2f}px", fontsize=10)
axes1[0, 3].axis("off")

# Baris 2: Checkerboard
checker_before = buat_checkerboard_overlay(img_translated, img_original)
checker_feat = buat_checkerboard_overlay(aligned_feat_trans, img_original)
checker_ecc_t = buat_checkerboard_overlay(aligned_ecc_trans, img_original)
checker_ecc_af = buat_checkerboard_overlay(aligned_ecc_aff_trans, img_original)

axes1[1, 0].imshow(cv2.cvtColor(checker_before, cv2.COLOR_BGR2RGB))
axes1[1, 0].set_title("Checkerboard: Before", fontsize=10)
axes1[1, 0].axis("off")

axes1[1, 1].imshow(cv2.cvtColor(checker_feat, cv2.COLOR_BGR2RGB))
axes1[1, 1].set_title("Checkerboard: Feature", fontsize=10)
axes1[1, 1].axis("off")

axes1[1, 2].imshow(cv2.cvtColor(checker_ecc_t, cv2.COLOR_BGR2RGB))
axes1[1, 2].set_title("Checkerboard: ECC Trans", fontsize=10)
axes1[1, 2].axis("off")

axes1[1, 3].imshow(cv2.cvtColor(checker_ecc_af, cv2.COLOR_BGR2RGB))
axes1[1, 3].set_title("Checkerboard: ECC Affine", fontsize=10)
axes1[1, 3].axis("off")

plt.suptitle("Percobaan 13: Image Registration - Gambar Translated\n"
             "(Overlay blend dan Checkerboard)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "13_grid_registration_translated.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid registration translated disimpan.")
plt.close()

# --- Grid 2: Rotated image registration ---
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

overlay_feat_r = buat_overlay_blend(aligned_feat_rot, img_original, 0.5)

axes2[0, 0].imshow(cv2.cvtColor(overlay_before_rot, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title(f"Before\nmean={err_before_r[0]:.2f}px", fontsize=10)
axes2[0, 0].axis("off")

axes2[0, 1].imshow(cv2.cvtColor(overlay_feat_r, cv2.COLOR_BGR2RGB))
axes2[0, 1].set_title(f"Feature-Based\nmean={err_feat_r[0]:.2f}px", fontsize=10)
axes2[0, 1].axis("off")

axes2[0, 2].imshow(cv2.cvtColor(buat_overlay_blend(aligned_ecc_eucl, img_original), cv2.COLOR_BGR2RGB))
axes2[0, 2].set_title(f"ECC Euclidean\nmean={err_ecc_er[0]:.2f}px", fontsize=10)
axes2[0, 2].axis("off")

# Checkerboards untuk rotated
checker_rot_before = buat_checkerboard_overlay(img_rotated, img_original)
checker_rot_feat = buat_checkerboard_overlay(aligned_feat_rot, img_original)
checker_rot_eucl = buat_checkerboard_overlay(aligned_ecc_eucl, img_original)

axes2[1, 0].imshow(cv2.cvtColor(checker_rot_before, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("Checkerboard: Before", fontsize=10)
axes2[1, 0].axis("off")

axes2[1, 1].imshow(cv2.cvtColor(checker_rot_feat, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title("Checkerboard: Feature", fontsize=10)
axes2[1, 1].axis("off")

axes2[1, 2].imshow(cv2.cvtColor(checker_rot_eucl, cv2.COLOR_BGR2RGB))
axes2[1, 2].set_title("Checkerboard: ECC Euclidean", fontsize=10)
axes2[1, 2].axis("off")

plt.suptitle("Percobaan 13: Image Registration - Gambar Rotated",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "13_grid_registration_rotated.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid registration rotated disimpan.")
plt.close()

# --- Grid 3: Perspective + Perbandingan ECC modes ---
fig3, axes3 = plt.subplots(2, 4, figsize=(22, 10))

axes3[0, 0].imshow(cv2.cvtColor(overlay_before_persp, cv2.COLOR_BGR2RGB))
axes3[0, 0].set_title("Before Alignment\n(Perspective)", fontsize=10)
axes3[0, 0].axis("off")

for i, (mode_nama, data) in enumerate(ecc_persp_results.items()):
    if i < 3:
        overlay_p = buat_overlay_blend(data["aligned"], img_original, 0.5)
        axes3[0, i + 1].imshow(cv2.cvtColor(overlay_p, cv2.COLOR_BGR2RGB))
        axes3[0, i + 1].set_title(f"ECC {mode_nama}\nmean={data['mean_err']:.2f}px", fontsize=10)
        axes3[0, i + 1].axis("off")

# Baris bawah: ECC Homography + Feature-based + difference maps
if "Homography" in ecc_persp_results:
    om = buat_overlay_blend(ecc_persp_results["Homography"]["aligned"], img_original, 0.5)
    axes3[1, 0].imshow(cv2.cvtColor(om, cv2.COLOR_BGR2RGB))
    axes3[1, 0].set_title(f"ECC Homography\nmean={ecc_persp_results['Homography']['mean_err']:.2f}px",
                          fontsize=10)
else:
    axes3[1, 0].axis("off")
axes3[1, 0].axis("off")

# Feature-based result
overlay_feat_p = buat_overlay_blend(aligned_feat_persp, img_original, 0.5)
axes3[1, 1].imshow(cv2.cvtColor(overlay_feat_p, cv2.COLOR_BGR2RGB))
axes3[1, 1].set_title(f"Feature-Based\nmean={err_feat_p[0]:.2f}px", fontsize=10)
axes3[1, 1].axis("off")

# Difference map (before vs after best)
diff_before = cv2.absdiff(img_perspective, img_original)
axes3[1, 2].imshow(cv2.cvtColor(diff_before, cv2.COLOR_BGR2RGB))
axes3[1, 2].set_title("Perbedaan: Before", fontsize=10)
axes3[1, 2].axis("off")

diff_after = cv2.absdiff(aligned_feat_persp, img_original)
axes3[1, 3].imshow(cv2.cvtColor(diff_after, cv2.COLOR_BGR2RGB))
axes3[1, 3].set_title("Perbedaan: After (Feature)", fontsize=10)
axes3[1, 3].axis("off")

plt.suptitle("Percobaan 13: Image Registration - Gambar Perspective",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "13_grid_registration_perspective.png"),
            dpi=150, bbox_inches="tight")
print("  [OK] Grid registration perspective disimpan.")
plt.show()
plt.close()


# ============================================================
# LANGKAH 12: Tabel Ringkasan dan Kesimpulan
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 13: IMAGE REGISTRATION")
print("=" * 65)

# Tabel ringkasan semua hasil
print(f"\n  Tabel Perbandingan Lengkap:")
print(f"  {'Metode':<22} | {'Mean Err':>10} | {'%<10px':>8} | {'Waktu':>8}")
print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
for nama, data in all_results.items():
    print(f"  {nama:<22} | {data['mean']:>8.2f}px | {data['pct_good']:>7.1f}% | "
          f"{data['waktu']:>6.1f}ms")

# Daftar file output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("13_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

# Kesimpulan
print("\n  Kesimpulan:")
print("    - Feature-based baik untuk displacement besar (bisa wrap perspektif)")
print("    - ECC Translation optimal untuk gambar yang hanya bergeser (cepat)")
print("    - ECC Euclidean menangani rotasi + translasi dengan baik")
print("    - ECC Affine fleksibel untuk berbagai deformasi kecil")
print("    - ECC Homography paling general, tapi bisa gagal konvergen")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.findTransformECC() → Registrasi ECC (iteratif)")
print("    cv2.MOTION_TRANSLATION → Mode translasi (2 DOF)")
print("    cv2.MOTION_EUCLIDEAN   → Mode rigid (3 DOF)")
print("    cv2.MOTION_AFFINE      → Mode affine (6 DOF)")
print("    cv2.MOTION_HOMOGRAPHY  → Mode perspektif (8 DOF)")
print("    cv2.warpAffine()       → Warping affine")
print("    cv2.warpPerspective()  → Warping perspektif")
print("=" * 65)
