"""
==========================================================================
PERCOBAAN 11: PANORAMA CROPPING DAN AUTO-CROP
==========================================================================
Program ini mengimplementasikan teknik auto-cropping untuk menghapus
area hitam/invalid dari hasil panorama stitching. Setelah warping
perspektif, panorama biasanya memiliki area hitam di tepi yang perlu
dihilangkan agar hasil akhir terlihat rapi.

Konsep yang dipelajari:
- Auto-cropping panorama hasil stitching
- Deteksi area valid (non-hitam) menggunakan thresholding
- Contour-based cropping untuk menemukan area terbesar
- Operasi morfologi untuk membersihkan mask sebelum crop
- Maximum inscribed rectangle (persegi terbesar tanpa hitam)
- Perbandingan berbagai metode cropping

Fungsi utama yang dipelajari:
- cv2.threshold()     : Threshold untuk menemukan area valid (non-hitam)
- cv2.findContours()  : Menemukan kontur area valid
- cv2.boundingRect()  : Mendapatkan bounding rectangle dari kontur
- cv2.morphologyEx()  : Operasi morfologi untuk membersihkan mask
- cv2.bitwise_and()   : Masking area valid
- cv2.erode()         : Erosi untuk memperkecil area mask
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
print("PERCOBAAN 11: PANORAMA CROPPING DAN AUTO-CROP")
print("=" * 65)


# ============================================================
# FUNGSI HELPER: Homography dan Stitching Sederhana
# ============================================================

def hitung_homography(img_src, img_dst, label=""):
    """
    Menghitung homography dari img_src ke img_dst.
    Pipeline: SIFT -> FLANN -> ratio test -> RANSAC.

    Parameter:
    - img_src : Gambar sumber
    - img_dst : Gambar tujuan
    - label   : Label untuk logging

    Returns:
    - H        : Matriks homography 3x3
    - n_inlier : Jumlah inlier RANSAC
    """
    # Mengkonversi kedua gambar ke grayscale untuk deteksi fitur
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    # Membuat detektor SIFT
    sift = cv2.SIFT_create()

    # Mendeteksi keypoints dan menghitung deskriptor
    kp_src, desc_src = sift.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = sift.detectAndCompute(gray_dst, None)

    # Memvalidasi deskriptor
    if (desc_src is None or desc_dst is None or
            len(desc_src) < 4 or len(desc_dst) < 4):
        if label:
            print(f"    {label}: Tidak cukup fitur terdeteksi")
        return np.eye(3, dtype=np.float64), 0

    # Mengonfigurasi FLANN matcher untuk pencarian tetangga terdekat
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Melakukan k-NN matching (k=2) untuk Lowe's ratio test
    matches = flann.knnMatch(desc_src, desc_dst, k=2)

    # Menerapkan Lowe's ratio test (threshold=0.75)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Membutuhkan minimal 10 kecocokan baik untuk homography
    if len(good) < 10:
        if label:
            print(f"    {label}: Hanya {len(good)} kecocokan (kurang dari 10)")
        return np.eye(3, dtype=np.float64), 0

    # Mengekstrak titik korespondensi dari kecocokan yang baik
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Mengestimasi homography menggunakan RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Menghitung jumlah inlier
    n_inlier = int(mask.ravel().sum()) if mask is not None else 0

    if label:
        print(f"    {label}: {len(good)} matches, {n_inlier} inliers")

    return H, n_inlier


def stitch_dua_gambar(img_left, img_right, label=""):
    """
    Melakukan stitching sederhana dua gambar (tanpa blending).
    Menghasilkan raw panorama dengan area hitam di tepinya.

    Parameter:
    - img_left  : Gambar kiri
    - img_right : Gambar kanan (referensi)
    - label     : Label logging

    Returns:
    - panorama  : Hasil stitching (bisa memiliki area hitam)
    """
    # Menghitung homography dari gambar kiri ke gambar kanan
    H, n_inlier = hitung_homography(img_left, img_right, label)

    # Mendapatkan dimensi kedua gambar
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    # Menghitung batas canvas dari sudut gambar kiri yang ditransformasi
    corners_left = np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]]).reshape(-1, 1, 2)
    corners_right = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)

    # Mentransformasi sudut gambar kiri menggunakan homography
    corners_left_t = cv2.perspectiveTransform(corners_left, H)

    # Menggabungkan semua sudut untuk menentukan batas canvas
    all_corners = np.concatenate([corners_left_t, corners_right], axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # Memastikan koordinat negatif ditangani
    x_min = min(x_min, 0)
    y_min = min(y_min, 0)

    # Menghitung ukuran canvas dan matriks translasi
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

    # Melakukan warping gambar kiri ke canvas
    warped = cv2.warpPerspective(img_left, T @ H, (canvas_w, canvas_h))

    # Menempatkan gambar kanan pada canvas
    ox, oy = -x_min, -y_min
    y_end = min(oy + h_r, canvas_h)
    x_end = min(ox + w_r, canvas_w)
    warped[oy:y_end, ox:x_end] = img_right[:y_end - oy, :x_end - ox]

    return warped


# ============================================================
# LANGKAH 1: Membuat Panorama Mentah dengan Area Hitam
# ============================================================
print("\n[LANGKAH 1] Membuat panorama mentah dari gambar outdoor...")

# Memuat gambar panorama outdoor (3 bagian yang tumpang tindih)
img_out1 = cv2.imread(os.path.join(IMAGE_DIR, "panorama_outdoor_1.jpg"))
img_out2 = cv2.imread(os.path.join(IMAGE_DIR, "panorama_outdoor_2.jpg"))
img_out3 = cv2.imread(os.path.join(IMAGE_DIR, "panorama_outdoor_3.jpg"))

# Memvalidasi gambar berhasil dimuat
if img_out1 is None or img_out2 is None or img_out3 is None:
    print("[ERROR] Gambar panorama outdoor tidak ditemukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan dimensi gambar
print(f"  Outdoor 1: {img_out1.shape[1]}x{img_out1.shape[0]}")
print(f"  Outdoor 2: {img_out2.shape[1]}x{img_out2.shape[0]}")
print(f"  Outdoor 3: {img_out3.shape[1]}x{img_out3.shape[0]}")

# Melakukan stitching bertahap: (1+2) lalu (+3)
print("\n  Stitching outdoor 1 + 2...")
pano_step1 = stitch_dua_gambar(img_out1, img_out2, "outdoor 1→2")

print("  Stitching (1+2) + 3...")
panorama_outdoor = stitch_dua_gambar(pano_step1, img_out3, "outdoor (1+2)→3")

# Menyimpan panorama mentah (raw) yang masih memiliki area hitam
cv2.imwrite(os.path.join(OUTPUT_DIR, "11_panorama_raw_outdoor.jpg"), panorama_outdoor)
print(f"  [OK] Panorama raw outdoor: {panorama_outdoor.shape[1]}x{panorama_outdoor.shape[0]}")

# Memuat sekaligus membuat panorama indoor
print("\n  Membuat panorama mentah dari gambar indoor...")
img_in1 = cv2.imread(os.path.join(IMAGE_DIR, "panorama_indoor_1.jpg"))
img_in2 = cv2.imread(os.path.join(IMAGE_DIR, "panorama_indoor_2.jpg"))
img_in3 = cv2.imread(os.path.join(IMAGE_DIR, "panorama_indoor_3.jpg"))
img_in4 = cv2.imread(os.path.join(IMAGE_DIR, "panorama_indoor_4.jpg"))

panorama_indoor = None
if img_in1 is not None and img_in2 is not None:
    try:
        # Stitching bertahap indoor
        pano_in = stitch_dua_gambar(img_in1, img_in2, "indoor 1→2")
        if img_in3 is not None:
            pano_in = stitch_dua_gambar(pano_in, img_in3, "indoor +3")
        if img_in4 is not None:
            pano_in = stitch_dua_gambar(pano_in, img_in4, "indoor +4")
        panorama_indoor = pano_in
        cv2.imwrite(os.path.join(OUTPUT_DIR, "11_panorama_raw_indoor.jpg"), panorama_indoor)
        print(f"  [OK] Panorama raw indoor: {panorama_indoor.shape[1]}x{panorama_indoor.shape[0]}")
    except Exception as e:
        print(f"  [WARN] Gagal membuat panorama indoor: {e}")
        panorama_indoor = None
else:
    print("  [WARN] Gambar indoor tidak lengkap, skip indoor panorama.")


# ============================================================
# LANGKAH 2: Menampilkan Panorama Raw dengan Area Hitam
# ============================================================
print("\n[LANGKAH 2] Menganalisis area hitam pada panorama raw...")

def analisis_area_hitam(panorama, nama=""):
    """
    Menganalisis seberapa banyak area hitam (invalid) pada panorama.

    Parameter:
    - panorama : Gambar panorama dengan area hitam
    - nama     : Label untuk printing

    Returns:
    - pct_hitam : Persentase area hitam
    """
    # Mengkonversi ke grayscale untuk deteksi area hitam
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    # Menghitung piksel hitam (intensitas < 5 dianggap hitam)
    hitam = np.sum(gray < 5)
    total = gray.size
    pct_hitam = hitam / total * 100

    # Menampilkan statistik
    print(f"  {nama}:")
    print(f"    Total piksel     : {total:,}")
    print(f"    Piksel hitam     : {hitam:,} ({pct_hitam:.1f}%)")
    print(f"    Piksel valid     : {total - hitam:,} ({100 - pct_hitam:.1f}%)")

    return pct_hitam

# Menganalisis area hitam pada panorama outdoor
pct_outdoor = analisis_area_hitam(panorama_outdoor, "Panorama Outdoor")

# Menganalisis area hitam pada panorama indoor jika tersedia
pct_indoor = 0.0
if panorama_indoor is not None:
    pct_indoor = analisis_area_hitam(panorama_indoor, "Panorama Indoor")


# ============================================================
# LANGKAH 3: Method 1 - Simple Crop (Non-Black Rows/Columns)
# ============================================================
print("\n[LANGKAH 3] Method 1: Simple crop (cari baris/kolom non-hitam)...")

def simple_crop(panorama, threshold=5):
    """
    Metode cropping paling sederhana: mencari baris dan kolom pertama/terakhir
    yang seluruhnya bukan hitam, lalu crop ke area tersebut.

    Parameter:
    - panorama  : Gambar panorama dengan area hitam
    - threshold : Nilai minimum untuk dianggap bukan hitam

    Returns:
    - cropped   : Gambar yang sudah di-crop
    - bbox      : (x, y, w, h) bounding box yang digunakan
    """
    # Mengkonversi ke grayscale untuk analisis
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    # Mencari baris yang memiliki setidaknya satu piksel non-hitam
    # np.any mengecek apakah ada piksel > threshold di setiap baris
    rows_valid = np.any(gray > threshold, axis=1)

    # Mencari kolom yang memiliki setidaknya satu piksel non-hitam
    cols_valid = np.any(gray > threshold, axis=0)

    # Mendapatkan indeks baris/kolom pertama dan terakhir yang valid
    y_min, y_max = np.where(rows_valid)[0][[0, -1]]
    x_min, x_max = np.where(cols_valid)[0][[0, -1]]

    # Melakukan crop berdasarkan bounding box tersebut
    cropped = panorama[y_min:y_max + 1, x_min:x_max + 1].copy()

    return cropped, (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


# Mengukur waktu eksekusi
waktu_start = time.time()

# Menerapkan simple crop pada panorama outdoor
crop_simple, bbox_simple = simple_crop(panorama_outdoor)
waktu_simple = time.time() - waktu_start

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "11_crop_simple_outdoor.jpg"), crop_simple)
print(f"  Hasil crop: {crop_simple.shape[1]}x{crop_simple.shape[0]}")
print(f"  Bounding box: x={bbox_simple[0]}, y={bbox_simple[1]}, "
      f"w={bbox_simple[2]}, h={bbox_simple[3]}")
print(f"  Waktu: {waktu_simple*1000:.2f} ms")

# Menghitung persentase area yang hilang karena crop
area_original = panorama_outdoor.shape[0] * panorama_outdoor.shape[1]
area_cropped = crop_simple.shape[0] * crop_simple.shape[1]
pct_lost_simple = (1 - area_cropped / area_original) * 100
print(f"  Area hilang: {pct_lost_simple:.1f}%")


# ============================================================
# LANGKAH 4: Method 2 - Contour-Based Crop
# ============================================================
print("\n[LANGKAH 4] Method 2: Contour-based crop (threshold + findContours)...")

def contour_crop(panorama, threshold=5):
    """
    Metode cropping berbasis kontur: threshold gambar → cari kontur terbesar
    → boundingRect dari kontur tersebut → crop.
    Lebih akurat dari simple crop karena mempertimbangkan bentuk area valid.

    Parameter:
    - panorama  : Gambar panorama
    - threshold : Nilai threshold untuk binarisasi

    Returns:
    - cropped   : Gambar yang sudah di-crop
    - bbox      : (x, y, w, h) bounding box
    """
    # Mengkonversi ke grayscale
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    # Melakukan thresholding untuk membuat mask biner (hitam = 0, lainnya = 255)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Mencari semua kontur pada mask biner
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Jika tidak ada kontur ditemukan, kembalikan gambar asli
    if len(contours) == 0:
        h, w = panorama.shape[:2]
        return panorama.copy(), (0, 0, w, h)

    # Mencari kontur terbesar (area valid terbesar)
    kontur_terbesar = max(contours, key=cv2.contourArea)

    # Mendapatkan bounding rectangle dari kontur terbesar
    x, y, w, h = cv2.boundingRect(kontur_terbesar)

    # Melakukan crop berdasarkan bounding rectangle
    cropped = panorama[y:y + h, x:x + w].copy()

    return cropped, (x, y, w, h)


# Mengukur waktu eksekusi
waktu_start = time.time()

# Menerapkan contour-based crop
crop_contour, bbox_contour = contour_crop(panorama_outdoor)
waktu_contour = time.time() - waktu_start

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "11_crop_contour_outdoor.jpg"), crop_contour)
print(f"  Hasil crop: {crop_contour.shape[1]}x{crop_contour.shape[0]}")
print(f"  Bounding box: x={bbox_contour[0]}, y={bbox_contour[1]}, "
      f"w={bbox_contour[2]}, h={bbox_contour[3]}")
print(f"  Waktu: {waktu_contour*1000:.2f} ms")

# Menghitung area yang hilang
area_contour = crop_contour.shape[0] * crop_contour.shape[1]
pct_lost_contour = (1 - area_contour / area_original) * 100
print(f"  Area hilang: {pct_lost_contour:.1f}%")


# ============================================================
# LANGKAH 5: Method 3 - Morphological Crop
# ============================================================
print("\n[LANGKAH 5] Method 3: Morphological crop (opening + closing + crop)...")

def morphological_crop(panorama, threshold=5, kernel_size=15):
    """
    Metode cropping menggunakan operasi morfologi untuk membersihkan
    noise pada mask sebelum cropping. Morphological opening menghilangkan
    noise kecil, closing mengisi lubang kecil.

    Parameter:
    - panorama    : Gambar panorama
    - threshold   : Nilai threshold binarisasi
    - kernel_size : Ukuran kernel morfologi

    Returns:
    - cropped     : Gambar yang sudah di-crop
    - bbox        : (x, y, w, h) bounding box
    - mask_clean  : Mask setelah operasi morfologi (untuk visualisasi)
    """
    # Mengkonversi ke grayscale
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    # Melakukan thresholding untuk membuat mask biner
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Membuat kernel untuk operasi morfologi (persegi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Menerapkan morphological closing: mengisi lubang-lubang kecil di mask
    # Closing = dilatasi → erosi
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Menerapkan morphological opening: menghilangkan noise kecil
    # Opening = erosi → dilatasi
    mask_clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Mencari kontur pada mask yang sudah dibersihkan
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Jika tidak ada kontur, kembalikan gambar asli
    if len(contours) == 0:
        h, w = panorama.shape[:2]
        return panorama.copy(), (0, 0, w, h), mask_clean

    # Mencari kontur terbesar
    kontur_terbesar = max(contours, key=cv2.contourArea)

    # Mendapatkan bounding rectangle
    x, y, w, h = cv2.boundingRect(kontur_terbesar)

    # Melakukan crop
    cropped = panorama[y:y + h, x:x + w].copy()

    return cropped, (x, y, w, h), mask_clean


# Mengukur waktu eksekusi
waktu_start = time.time()

# Menerapkan morphological crop
crop_morph, bbox_morph, mask_morph = morphological_crop(panorama_outdoor)
waktu_morph = time.time() - waktu_start

# Menyimpan hasil dan mask
cv2.imwrite(os.path.join(OUTPUT_DIR, "11_crop_morph_outdoor.jpg"), crop_morph)
cv2.imwrite(os.path.join(OUTPUT_DIR, "11_mask_morph_outdoor.jpg"), mask_morph)
print(f"  Hasil crop: {crop_morph.shape[1]}x{crop_morph.shape[0]}")
print(f"  Bounding box: x={bbox_morph[0]}, y={bbox_morph[1]}, "
      f"w={bbox_morph[2]}, h={bbox_morph[3]}")
print(f"  Waktu: {waktu_morph*1000:.2f} ms")

# Menghitung area yang hilang
area_morph = crop_morph.shape[0] * crop_morph.shape[1]
pct_lost_morph = (1 - area_morph / area_original) * 100
print(f"  Area hilang: {pct_lost_morph:.1f}%")


# ============================================================
# LANGKAH 6: Method 4 - Maximum Inscribed Rectangle
# ============================================================
print("\n[LANGKAH 6] Method 4: Maximum inscribed rectangle...")

def max_inscribed_rect(panorama, threshold=5, step=5):
    """
    Mencari persegi panjang terbesar yang seluruhnya berada di area valid
    (tanpa piksel hitam). Menggunakan pendekatan iteratif: erosi bertahap
    hingga ditemukan persegi yang sepenuhnya valid.

    Parameter:
    - panorama  : Gambar panorama
    - threshold : Nilai threshold untuk binarisasi
    - step      : Langkah erosi per iterasi (piksel)

    Returns:
    - cropped   : Gambar yang sudah di-crop (tanpa area hitam sama sekali)
    - bbox      : (x, y, w, h) bounding box
    """
    # Mengkonversi ke grayscale dan buat mask biner
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Pendekatan iteratif: erosi mask bertahap, lalu cek bounding rect
    # Semakin banyak erosi, semakin kecil area tapi semakin bersih dari hitam
    h_img, w_img = mask.shape[:2]
    best_bbox = (0, 0, w_img, h_img)
    best_area = 0

    # Membuat kernel erosi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (step, step))

    # Menyalin mask untuk erosi iteratif
    current_mask = mask.copy()

    for iteration in range(100):
        # Mencari kontur pada mask saat ini
        contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            break

        # Mendapatkan kontur terbesar
        kontur = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(kontur)

        # Mengecek apakah area di dalam bounding rect sepenuhnya valid
        roi_mask = mask[y:y + h, x:x + w]
        pct_valid = np.sum(roi_mask > 0) / (w * h) * 100

        # Jika area >99% valid, ini adalah kandidat yang baik
        if pct_valid > 99.0:
            area = w * h
            if area > best_area:
                best_area = area
                best_bbox = (x, y, w, h)
            break  # Sudah menemukan rectangle yang valid
        else:
            # Erosi mask untuk mengurangi area hitam di tepi
            current_mask = cv2.erode(current_mask, kernel)

    # Melakukan crop berdasarkan bounding box terbaik
    x, y, w, h = best_bbox
    cropped = panorama[y:y + h, x:x + w].copy()

    return cropped, best_bbox


# Mengukur waktu eksekusi
waktu_start = time.time()

# Menerapkan maximum inscribed rectangle
crop_maxrect, bbox_maxrect = max_inscribed_rect(panorama_outdoor)
waktu_maxrect = time.time() - waktu_start

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "11_crop_maxrect_outdoor.jpg"), crop_maxrect)
print(f"  Hasil crop: {crop_maxrect.shape[1]}x{crop_maxrect.shape[0]}")
print(f"  Bounding box: x={bbox_maxrect[0]}, y={bbox_maxrect[1]}, "
      f"w={bbox_maxrect[2]}, h={bbox_maxrect[3]}")
print(f"  Waktu: {waktu_maxrect*1000:.2f} ms")

# Menghitung area yang hilang
area_maxrect = crop_maxrect.shape[0] * crop_maxrect.shape[1]
pct_lost_maxrect = (1 - area_maxrect / area_original) * 100
print(f"  Area hilang: {pct_lost_maxrect:.1f}%")

# Memverifikasi tidak ada area hitam
gray_check = cv2.cvtColor(crop_maxrect, cv2.COLOR_BGR2GRAY)
pct_hitam_maxrect = np.sum(gray_check < 5) / gray_check.size * 100
print(f"  Piksel hitam tersisa: {pct_hitam_maxrect:.2f}%")


# ============================================================
# LANGKAH 7: Perbandingan Semua Metode Cropping
# ============================================================
print("\n[LANGKAH 7] Membuat tabel perbandingan semua metode cropping...")

# Menyusun data perbandingan untuk setiap metode
metode_data = {
    "Raw (Tanpa Crop)": {
        "gambar": panorama_outdoor,
        "bbox": (0, 0, panorama_outdoor.shape[1], panorama_outdoor.shape[0]),
        "waktu_ms": 0,
        "area": area_original,
        "pct_lost": 0.0
    },
    "Simple Crop": {
        "gambar": crop_simple,
        "bbox": bbox_simple,
        "waktu_ms": waktu_simple * 1000,
        "area": area_cropped,
        "pct_lost": pct_lost_simple
    },
    "Contour Crop": {
        "gambar": crop_contour,
        "bbox": bbox_contour,
        "waktu_ms": waktu_contour * 1000,
        "area": area_contour,
        "pct_lost": pct_lost_contour
    },
    "Morphological Crop": {
        "gambar": crop_morph,
        "bbox": bbox_morph,
        "waktu_ms": waktu_morph * 1000,
        "area": area_morph,
        "pct_lost": pct_lost_morph
    },
    "Max Inscribed Rect": {
        "gambar": crop_maxrect,
        "bbox": bbox_maxrect,
        "waktu_ms": waktu_maxrect * 1000,
        "area": area_maxrect,
        "pct_lost": pct_lost_maxrect
    }
}

# Menampilkan tabel perbandingan
print(f"\n  {'Metode':<22} | {'Ukuran':>12} | {'Area Hilang':>11} | {'Waktu (ms)':>10}")
print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*11}-+-{'-'*10}")
for nama, data in metode_data.items():
    g = data["gambar"]
    ukuran = f"{g.shape[1]}x{g.shape[0]}"
    print(f"  {nama:<22} | {ukuran:>12} | {data['pct_lost']:>10.1f}% | {data['waktu_ms']:>10.2f}")


# ============================================================
# LANGKAH 8: Menghitung Persentase Piksel Hitam Tersisa
# ============================================================
print("\n[LANGKAH 8] Menghitung piksel hitam tersisa per metode...")

for nama, data in metode_data.items():
    # Mengkonversi ke grayscale
    g = cv2.cvtColor(data["gambar"], cv2.COLOR_BGR2GRAY)

    # Menghitung persentase piksel hitam
    pct_hitam_sisa = np.sum(g < 5) / g.size * 100

    # Menampilkan hasil
    print(f"  {nama:<22}: {pct_hitam_sisa:.2f}% piksel hitam tersisa")


# ============================================================
# LANGKAH 9: Menerapkan Cropping pada Panorama Indoor
# ============================================================
print("\n[LANGKAH 9] Menerapkan semua metode crop pada panorama indoor...")

crop_indoor_results = {}
if panorama_indoor is not None:
    try:
        # Method 1: Simple crop pada indoor
        crop_in_simple, bbox_in_simple = simple_crop(panorama_indoor)
        crop_indoor_results["Simple"] = crop_in_simple
        print(f"  Simple crop indoor  : {crop_in_simple.shape[1]}x{crop_in_simple.shape[0]}")

        # Method 2: Contour crop pada indoor
        crop_in_contour, bbox_in_contour = contour_crop(panorama_indoor)
        crop_indoor_results["Contour"] = crop_in_contour
        print(f"  Contour crop indoor : {crop_in_contour.shape[1]}x{crop_in_contour.shape[0]}")

        # Method 3: Morphological crop pada indoor
        crop_in_morph, bbox_in_morph, _ = morphological_crop(panorama_indoor)
        crop_indoor_results["Morph"] = crop_in_morph
        print(f"  Morph crop indoor   : {crop_in_morph.shape[1]}x{crop_in_morph.shape[0]}")

        # Method 4: Max inscribed rect pada indoor
        crop_in_maxrect, bbox_in_maxrect = max_inscribed_rect(panorama_indoor)
        crop_indoor_results["MaxRect"] = crop_in_maxrect
        print(f"  MaxRect crop indoor : {crop_in_maxrect.shape[1]}x{crop_in_maxrect.shape[0]}")

        # Menyimpan hasil indoor
        for key, img_crop in crop_indoor_results.items():
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"11_crop_{key.lower()}_indoor.jpg"), img_crop)
        print("  [OK] Semua hasil crop indoor disimpan.")

    except Exception as e:
        print(f"  [WARN] Error pada crop indoor: {e}")
else:
    print("  [SKIP] Panorama indoor tidak tersedia.")


# ============================================================
# LANGKAH 10: Penanganan Edge Cases
# ============================================================
print("\n[LANGKAH 10] Menguji edge cases (area hitam besar/minimal)...")

# Edge Case 1: Panorama dengan area hitam sangat besar (simulasi)
print("\n  Edge Case 1: Panorama dengan 60%+ area hitam...")
h_ec, w_ec = 400, 800
pano_banyak_hitam = np.zeros((h_ec, w_ec, 3), dtype=np.uint8)

# Membuat area valid kecil di bagian tengah atas (simulasi warping ekstrem)
valid_area = panorama_outdoor[:min(200, panorama_outdoor.shape[0]),
                              :min(400, panorama_outdoor.shape[1])]
vh, vw = valid_area.shape[:2]

# Menempatkan area valid di posisi offset
ox_ec, oy_ec = 200, 50
pano_banyak_hitam[oy_ec:oy_ec + vh, ox_ec:ox_ec + vw] = valid_area

# Menerapkan setiap metode crop
try:
    ec1_simple, _ = simple_crop(pano_banyak_hitam)
    ec1_contour, _ = contour_crop(pano_banyak_hitam)
    ec1_morph, _, _ = morphological_crop(pano_banyak_hitam)
    ec1_maxrect, _ = max_inscribed_rect(pano_banyak_hitam)

    print(f"    Simple  : {ec1_simple.shape[1]}x{ec1_simple.shape[0]}")
    print(f"    Contour : {ec1_contour.shape[1]}x{ec1_contour.shape[0]}")
    print(f"    Morph   : {ec1_morph.shape[1]}x{ec1_morph.shape[0]}")
    print(f"    MaxRect : {ec1_maxrect.shape[1]}x{ec1_maxrect.shape[0]}")
except Exception as e:
    print(f"    [WARN] Error pada edge case 1: {e}")

# Edge Case 2: Panorama yang hampir tidak ada hitamnya
print("\n  Edge Case 2: Panorama dengan sedikit area hitam...")
pano_sedikit_hitam = panorama_outdoor.copy()

# Hanya menambahkan segitiga hitam kecil di sudut kiri atas
pts_hitam = np.array([[0, 0], [50, 0], [0, 50]], np.int32)
cv2.fillPoly(pano_sedikit_hitam, [pts_hitam], (0, 0, 0))

try:
    ec2_simple, _ = simple_crop(pano_sedikit_hitam)
    ec2_maxrect, _ = max_inscribed_rect(pano_sedikit_hitam)
    print(f"    Simple  : {ec2_simple.shape[1]}x{ec2_simple.shape[0]}")
    print(f"    MaxRect : {ec2_maxrect.shape[1]}x{ec2_maxrect.shape[0]}")
except Exception as e:
    print(f"    [WARN] Error pada edge case 2: {e}")


# ============================================================
# LANGKAH 11: Membuat Grid Visualisasi Perbandingan
# ============================================================
print("\n[LANGKAH 11] Membuat grid visualisasi perbandingan semua metode...")

# Membuat figure utama: 2 baris x 3 kolom
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Daftar gambar dan judul untuk ditampilkan
items = [
    ("Raw Panorama\n(dengan area hitam)", panorama_outdoor),
    ("Method 1: Simple Crop", crop_simple),
    ("Method 2: Contour Crop", crop_contour),
    ("Method 3: Morphological Crop", crop_morph),
    ("Method 4: Max Inscribed Rect", crop_maxrect),
    ("Morphological Mask", None)  # Placeholder untuk mask
]

for idx, (title, img) in enumerate(items):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    if idx == 5:
        # Menampilkan mask morfologi sebagai heatmap
        ax.imshow(mask_morph, cmap='gray')
        ax.set_title("Morphological Mask", fontsize=11)
    else:
        # Menampilkan gambar (konversi BGR ke RGB untuk matplotlib)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Menambahkan informasi dimensi
        h_i, w_i = img.shape[:2]
        ax.set_title(f"{title}\n{w_i}x{h_i}", fontsize=10)

    ax.axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 11: Perbandingan Metode Auto-Crop Panorama\n"
             "(Outdoor Panorama)",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid perbandingan
plt.savefig(os.path.join(OUTPUT_DIR, "11_grid_perbandingan_crop.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid perbandingan crop outdoor disimpan.")
plt.close()

# --- Grid kedua: Perbandingan statistik (bar chart) ---
print("  Membuat grafik perbandingan area dan waktu...")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Menyiapkan data untuk bar chart
metode_nama = ["Simple", "Contour", "Morph", "MaxRect"]
area_hilang = [pct_lost_simple, pct_lost_contour, pct_lost_morph, pct_lost_maxrect]
waktu_list = [waktu_simple * 1000, waktu_contour * 1000,
              waktu_morph * 1000, waktu_maxrect * 1000]

# Warna untuk setiap metode
warna = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

# Bar chart 1: Persentase area yang hilang
bars1 = ax1.bar(metode_nama, area_hilang, color=warna, edgecolor='black', linewidth=0.5)
ax1.set_ylabel("Area Hilang (%)")
ax1.set_title("Perbandingan Area yang Hilang\n(lebih kecil = lebih baik)")

# Menambahkan label nilai pada setiap bar
for bar, val in zip(bars1, area_hilang):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontsize=9)

# Bar chart 2: Waktu eksekusi
bars2 = ax2.bar(metode_nama, waktu_list, color=warna, edgecolor='black', linewidth=0.5)
ax2.set_ylabel("Waktu (ms)")
ax2.set_title("Perbandingan Waktu Eksekusi\n(lebih kecil = lebih cepat)")

# Menambahkan label waktu
for bar, val in zip(bars2, waktu_list):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f'{val:.2f}', ha='center', fontsize=9)

plt.suptitle("Percobaan 11: Statistik Metode Cropping",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "11_grid_statistik_crop.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid statistik crop disimpan.")
plt.close()

# --- Grid ketiga: Indoor panorama (jika tersedia) ---
if panorama_indoor is not None and len(crop_indoor_results) > 0:
    n_indoor = len(crop_indoor_results) + 1  # +1 untuk raw
    fig3, axes3 = plt.subplots(1, min(n_indoor, 5), figsize=(20, 4))

    if n_indoor == 1:
        axes3 = [axes3]

    # Menampilkan raw indoor
    axes3[0].imshow(cv2.cvtColor(panorama_indoor, cv2.COLOR_BGR2RGB))
    axes3[0].set_title(f"Raw Indoor\n{panorama_indoor.shape[1]}x{panorama_indoor.shape[0]}")
    axes3[0].axis("off")

    # Menampilkan hasil crop masing-masing metode
    for i, (key, img_crop) in enumerate(crop_indoor_results.items()):
        if i + 1 < len(axes3):
            axes3[i + 1].imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
            axes3[i + 1].set_title(f"{key}\n{img_crop.shape[1]}x{img_crop.shape[0]}")
            axes3[i + 1].axis("off")

    plt.suptitle("Auto-Crop pada Panorama Indoor", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_grid_crop_indoor.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Grid crop indoor disimpan.")
    plt.close()


# ============================================================
# LANGKAH 12: Ringkasan dan Kesimpulan
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 11: PANORAMA CROPPING DAN AUTO-CROP")
print("=" * 65)

# Menampilkan tabel ringkasan akhir
print("\n  Tabel Ringkasan Metode Cropping:")
print(f"  {'Metode':<22} | {'Ukuran Hasil':>12} | {'Crop %':>8} | {'Waktu':>8} | {'Sisa Hitam':>10}")
print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

for nama, data in metode_data.items():
    if nama == "Raw (Tanpa Crop)":
        continue
    g = data["gambar"]
    ukuran = f"{g.shape[1]}x{g.shape[0]}"
    gray_cek = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    sisa_hitam = np.sum(gray_cek < 5) / gray_cek.size * 100
    print(f"  {nama:<22} | {ukuran:>12} | {data['pct_lost']:>7.1f}% | "
          f"{data['waktu_ms']:>6.2f}ms | {sisa_hitam:>9.2f}%")

# Menampilkan daftar file output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("11_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

# Kesimpulan metode terbaik
print("\n  Kesimpulan:")
print("    - Simple Crop     : Paling cepat, tapi mungkin menyisakan hitam di sudut")
print("    - Contour Crop    : Bagus untuk area valid yang menyambung")
print("    - Morph Crop      : Robust terhadap noise, membersihkan mask")
print("    - Max Inscribed   : Paling bersih (0% hitam), tapi area hilang lebih banyak")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.threshold()     → Binarisasi gambar (deteksi hitam/non-hitam)")
print("    cv2.findContours()  → Menemukan kontur area valid")
print("    cv2.boundingRect()  → Bounding rectangle dari kontur")
print("    cv2.morphologyEx()  → Operasi morfologi (opening, closing)")
print("    cv2.erode()         → Erosi mask untuk inscribed rectangle")
print("    cv2.bitwise_and()   → Masking area valid")
print("=" * 65)
