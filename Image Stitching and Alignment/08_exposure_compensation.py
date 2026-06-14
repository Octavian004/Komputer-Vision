"""
==========================================================================
PERCOBAAN 8: EXPOSURE COMPENSATION
==========================================================================
Program ini mengatasi perbedaan exposure antar gambar dalam panorama
menggunakan gain compensation. Perbedaan brightness antar gambar
menyebabkan seam yang terlihat jelas pada panorama. Exposure compensation
menyeimbangkan brightness sehingga transisi antar gambar lebih halus.

Konsep yang dipelajari:
- Perbedaan exposure dan dampaknya pada panorama
- Analisis brightness menggunakan histogram dan mean intensity
- Gain compensation manual (global brightness adjustment)
- Block-based gain compensation untuk koreksi lebih lokal
- Perbandingan stitching dengan dan tanpa exposure compensation
- Histogram equalization pada area overlap

Fungsi utama yang dipelajari:
- np.mean()               : Menghitung rata-rata brightness
- cv2.cvtColor()          : Konversi BGR ke HSV untuk analisis brightness
- cv2.calcHist()          : Menghitung histogram intensitas
- cv2.equalizeHist()      : Ekualisasi histogram
- cv2.createCLAHE()       : Adaptive histogram equalization
- cv2.Stitcher_create()   : Stitcher API otomatis (termasuk exp. comp.)
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan computer vision
import cv2

# Mengimpor library NumPy untuk operasi array dan matriks
import numpy as np

# Mengimpor library os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi histogram dan grid perbandingan
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
print("PERCOBAAN 8: EXPOSURE COMPENSATION")
print("=" * 65)


# ============================================================
# FUNGSI HELPER
# ============================================================

def hitung_homography(img_src, img_dst, label=""):
    """
    Menghitung homography dari img_src ke img_dst.
    Pipeline: SIFT → FLANN → ratio test → RANSAC.

    Parameter:
    - img_src : Gambar sumber
    - img_dst : Gambar tujuan
    - label   : Label logging

    Returns:
    - H        : Matriks homography 3x3
    - n_inlier : Jumlah inlier
    """
    # Mengkonversi ke grayscale
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    # Membuat detektor SIFT
    sift = cv2.SIFT_create()

    # Mendeteksi fitur dan deskriptor
    kp_src, desc_src = sift.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = sift.detectAndCompute(gray_dst, None)

    # Memeriksa validitas deskriptor
    if (desc_src is None or desc_dst is None or
            len(desc_src) < 4 or len(desc_dst) < 4):
        if label:
            print(f"    {label}: Tidak cukup fitur")
        return np.eye(3, dtype=np.float64), 0

    # Mengonfigurasi FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Melakukan kNN matching
    matches = flann.knnMatch(desc_src, desc_dst, k=2)

    # Menerapkan Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Memastikan ada cukup kecocokan
    if len(good) < 4:
        if label:
            print(f"    {label}: Terlalu sedikit matches ({len(good)})")
        return np.eye(3, dtype=np.float64), 0

    # Mengekstrak titik korespondensi
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Mengestimasi homography dengan RANSAC
    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        if label:
            print(f"    {label}: Homography gagal")
        return np.eye(3, dtype=np.float64), 0

    n_inlier = int(mask_h.ravel().sum()) if mask_h is not None else 0
    if label:
        print(f"    {label}: matches={len(good)}, inliers={n_inlier}")

    return H, n_inlier


def stitch_pair_simple(img_left, img_right, H):
    """
    Melakukan stitching sederhana (tanpa blending) dari dua gambar
    menggunakan homography yang diberikan.

    Parameter:
    - img_left  : Gambar kiri
    - img_right : Gambar kanan
    - H         : Homography dari kanan ke kiri

    Returns:
    - result    : Gambar hasil stitching
    """
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    # Menghitung batas canvas dari perspectiveTransform
    corners_right = np.float32([[0, 0], [w_r, 0], [w_r, h_r],
                                 [0, h_r]]).reshape(-1, 1, 2)
    corners_trans = cv2.perspectiveTransform(corners_right, H)

    # Menggabungkan semua sudut untuk menentukan ukuran canvas
    all_corners = np.concatenate([
        np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]]).reshape(-1, 1, 2),
        corners_trans
    ])

    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))

    canvas_w = min(x_max - x_min, 4000)
    canvas_h = min(y_max - y_min, 2000)

    # Matriks translasi
    H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]],
                            dtype=np.float64)

    # Warping gambar kanan
    warped_right = cv2.warpPerspective(img_right, H_translate @ H,
                                        (canvas_w, canvas_h))

    # Menempatkan gambar kiri pada canvas
    result = warped_right.copy()
    ox = -x_min
    oy = -y_min
    y1 = max(0, oy)
    y2 = min(canvas_h, oy + h_l)
    x1 = max(0, ox)
    x2 = min(canvas_w, ox + w_l)

    # Membuat mask untuk blending sederhana
    mask_left = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    mask_right = (cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

    sy1 = max(0, -oy)
    sx1 = max(0, -ox)
    actual_h = y2 - y1
    actual_w = x2 - x1

    region_left = img_left[sy1:sy1 + actual_h, sx1:sx1 + actual_w]
    mask_l_region = (cv2.cvtColor(region_left, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    mask_left[y1:y1 + actual_h, x1:x1 + actual_w] = mask_l_region

    # Blending: rata-rata di overlap, satu sisi di non-overlap
    overlap = (mask_left > 0) & (mask_right > 0)
    only_left = (mask_left > 0) & (mask_right == 0)

    # Menempatkan gambar kiri
    result[y1:y1 + actual_h, x1:x1 + actual_w] = np.where(
        only_left[y1:y1 + actual_h, x1:x1 + actual_w, np.newaxis],
        region_left[:actual_h, :actual_w],
        result[y1:y1 + actual_h, x1:x1 + actual_w]
    )

    # Rata-rata di area overlap
    for c in range(3):
        result[:, :, c] = np.where(
            overlap,
            ((result[:, :, c].astype(np.float32) +
              np.where(mask_left > 0,
                       cv2.warpPerspective(
                           img_left, H_translate,
                           (canvas_w, canvas_h)
                       )[:, :, c].astype(np.float32), 0)) / 2).astype(np.uint8),
            result[:, :, c]
        )

    return result


# ============================================================
# LANGKAH 1: Memuat Gambar dengan Exposure Berbeda
# ============================================================
print("\n[LANGKAH 1] Memuat gambar dengan exposure berbeda...")

# Memuat gambar dark, normal, dan bright
img_dark = cv2.imread(os.path.join(IMAGE_DIR, "exposure_dark_1.jpg"))
img_normal = cv2.imread(os.path.join(IMAGE_DIR, "exposure_normal_2.jpg"))
img_bright = cv2.imread(os.path.join(IMAGE_DIR, "exposure_bright_3.jpg"))

# Memeriksa apakah semua gambar berhasil dimuat
if img_dark is None or img_normal is None or img_bright is None:
    print("  [ERROR] Gambar exposure tidak ditemukan!")
    print("  Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi gambar
print(f"  exposure_dark_1.jpg:    {img_dark.shape[1]}x{img_dark.shape[0]} piksel")
print(f"  exposure_normal_2.jpg:  {img_normal.shape[1]}x{img_normal.shape[0]} piksel")
print(f"  exposure_bright_3.jpg:  {img_bright.shape[1]}x{img_bright.shape[0]} piksel")

# Menyusun list gambar exposure untuk stitching
exposure_images = [img_dark, img_normal, img_bright]
exposure_labels = ["Dark", "Normal", "Bright"]


# ============================================================
# LANGKAH 2: Analisis Brightness Differences
# ============================================================
print("\n[LANGKAH 2] Menganalisis perbedaan brightness...")

# Menganalisis brightness setiap gambar
brightness_info = []

for i, (img, label) in enumerate(zip(exposure_images, exposure_labels)):
    # Mengkonversi ke grayscale untuk analisis intensitas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Menghitung statistik brightness
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    min_val = np.min(gray)
    max_val = np.max(gray)

    # Mengkonversi ke HSV untuk analisis brightness (channel V)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_v = np.mean(v_channel)

    brightness_info.append({
        'label': label,
        'mean_gray': mean_val,
        'std_gray': std_val,
        'mean_v': mean_v,
        'min': min_val,
        'max': max_val
    })

    print(f"  {label:>7}: Mean={mean_val:.1f}, Std={std_val:.1f}, "
          f"V={mean_v:.1f}, Range=[{min_val}, {max_val}]")

# Menghitung rasio brightness
ref_brightness = brightness_info[1]['mean_gray']  # Normal sebagai referensi
print(f"\n  Referensi brightness: {ref_brightness:.1f} (Normal)")
for info in brightness_info:
    ratio = info['mean_gray'] / ref_brightness if ref_brightness > 0 else 1
    print(f"    {info['label']:>7}: rasio = {ratio:.3f}")


# ============================================================
# LANGKAH 3: Menghitung Histogram Sebelum Kompensasi
# ============================================================
print("\n[LANGKAH 3] Menghitung histogram sebelum kompensasi...")

# Menghitung histogram untuk setiap gambar
histograms_before = []
for i, img in enumerate(exposure_images):
    # Mengkonversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Menghitung histogram (256 bins, range 0-256)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histograms_before.append(hist)

print("  Histogram sebelum kompensasi dihitung.")


# ============================================================
# LANGKAH 4: Stitching TANPA Exposure Compensation
# ============================================================
print("\n[LANGKAH 4] Melakukan stitching TANPA exposure compensation...")

# Menghitung homography antar pasangan exposure
print("  Menghitung homography...")
H_dark_normal, _ = hitung_homography(img_dark, img_normal,
                                      label="H(dark→normal)")
H_bright_normal, _ = hitung_homography(img_bright, img_normal,
                                        label="H(bright→normal)")

# Stitching tanpa kompensasi menggunakan canvas manual
h_img, w_img = img_normal.shape[:2]

# Menghitung batas canvas
corners_dark = np.float32([[0, 0], [w_img, 0], [w_img, h_img],
                            [0, h_img]]).reshape(-1, 1, 2)
corners_bright = corners_dark.copy()

dark_trans = cv2.perspectiveTransform(corners_dark, H_dark_normal)
bright_trans = cv2.perspectiveTransform(corners_bright, H_bright_normal)
normal_corners = corners_dark.copy()

all_corners = np.concatenate([dark_trans, normal_corners, bright_trans])
x_min = int(np.floor(all_corners[:, 0, 0].min()))
y_min = int(np.floor(all_corners[:, 0, 1].min()))
x_max = int(np.ceil(all_corners[:, 0, 0].max()))
y_max = int(np.ceil(all_corners[:, 0, 1].max()))

canvas_w = min(x_max - x_min, 6000)
canvas_h = min(y_max - y_min, 3000)

# Matriks translasi
H_tr = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]],
                 dtype=np.float64)

# Warping ketiga gambar (tanpa kompensasi)
warped_dark_no = cv2.warpPerspective(img_dark, H_tr @ H_dark_normal,
                                      (canvas_w, canvas_h))
warped_normal_no = cv2.warpPerspective(img_normal, H_tr,
                                        (canvas_w, canvas_h))
warped_bright_no = cv2.warpPerspective(img_bright, H_tr @ H_bright_normal,
                                        (canvas_w, canvas_h))

# Blending sederhana (averaging)
canvas_no_comp = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
count_no_comp = np.zeros((canvas_h, canvas_w), dtype=np.float32)

for warped in [warped_dark_no, warped_normal_no, warped_bright_no]:
    mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    for c in range(3):
        canvas_no_comp[:, :, c] += warped[:, :, c].astype(np.float64) * mask
    count_no_comp += mask

count_no_comp[count_no_comp == 0] = 1
result_no_comp = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
for c in range(3):
    result_no_comp[:, :, c] = np.clip(
        canvas_no_comp[:, :, c] / count_no_comp, 0, 255
    ).astype(np.uint8)

# Crop dan simpan
gray_nc = cv2.cvtColor(result_no_comp, cv2.COLOR_BGR2GRAY)
_, thresh_nc = cv2.threshold(gray_nc, 1, 255, cv2.THRESH_BINARY)
cnt_nc, _ = cv2.findContours(thresh_nc, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
if cnt_nc:
    lg = max(cnt_nc, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(lg)
    result_no_comp_crop = result_no_comp[y:y + h, x:x + w]
else:
    result_no_comp_crop = result_no_comp

cv2.imwrite(os.path.join(OUTPUT_DIR, "08_stitch_no_compensation.jpg"),
            result_no_comp_crop)
print(f"  [OK] Stitching tanpa kompensasi disimpan.")
print(f"  Ukuran: {result_no_comp_crop.shape[1]}x{result_no_comp_crop.shape[0]}")


# ============================================================
# LANGKAH 5: Implementasi Gain Compensation Manual (Global)
# ============================================================
print("\n[LANGKAH 5] Mengimplementasikan gain compensation manual...")


def gain_compensation_global(images, ref_idx=1):
    """
    Melakukan global gain compensation.
    Menghitung rasio brightness terhadap gambar referensi
    dan mengalikan faktor gain untuk menyamakan brightness.

    Parameter:
    - images  : List gambar BGR
    - ref_idx : Indeks gambar referensi (default: 1 = normal)

    Returns:
    - compensated : List gambar yang sudah dikompensasi
    - gains       : List faktor gain untuk setiap gambar
    """
    # Menghitung mean brightness gambar referensi
    ref_gray = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY)
    ref_mean = np.mean(ref_gray)

    compensated = []
    gains = []

    for i, img in enumerate(images):
        # Menghitung mean brightness gambar ini
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_mean = np.mean(gray)

        # Menghitung gain factor (rasio brightness)
        if img_mean > 0:
            gain = ref_mean / img_mean
        else:
            gain = 1.0

        gains.append(gain)

        # Menerapkan gain ke semua channel
        compensated_img = np.clip(
            img.astype(np.float64) * gain, 0, 255
        ).astype(np.uint8)
        compensated.append(compensated_img)

        print(f"    Gambar {i + 1} ({exposure_labels[i]}): "
              f"mean={img_mean:.1f}, gain={gain:.3f}")

    return compensated, gains


# Melakukan global gain compensation
print("  Menghitung global gain factors...")
comp_global, gains_global = gain_compensation_global(exposure_images, ref_idx=1)

# Menyimpan gambar yang sudah dikompensasi
for i, (img_comp, label) in enumerate(zip(comp_global, exposure_labels)):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"08_compensated_global_{label.lower()}.jpg"),
                img_comp)
print("  [OK] Gambar terkompensasi (global gain) disimpan.")


# ============================================================
# LANGKAH 6: Stitching dengan Global Gain Compensation
# ============================================================
print("\n[LANGKAH 6] Melakukan stitching dengan global gain compensation...")

# Menghitung homography pada gambar terkompensasi
H_dark_comp, _ = hitung_homography(comp_global[0], comp_global[1],
                                    label="H(dark_comp→normal_comp)")
H_bright_comp, _ = hitung_homography(comp_global[2], comp_global[1],
                                      label="H(bright_comp→normal_comp)")

# Warping dan compositing gambar terkompensasi
warped_dark_gc = cv2.warpPerspective(comp_global[0], H_tr @ H_dark_comp,
                                      (canvas_w, canvas_h))
warped_normal_gc = cv2.warpPerspective(comp_global[1], H_tr,
                                        (canvas_w, canvas_h))
warped_bright_gc = cv2.warpPerspective(comp_global[2], H_tr @ H_bright_comp,
                                        (canvas_w, canvas_h))

# Blending
canvas_gc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
count_gc = np.zeros((canvas_h, canvas_w), dtype=np.float32)

for warped in [warped_dark_gc, warped_normal_gc, warped_bright_gc]:
    mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    for c in range(3):
        canvas_gc[:, :, c] += warped[:, :, c].astype(np.float64) * mask
    count_gc += mask

count_gc[count_gc == 0] = 1
result_gc = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
for c in range(3):
    result_gc[:, :, c] = np.clip(
        canvas_gc[:, :, c] / count_gc, 0, 255
    ).astype(np.uint8)

# Crop dan simpan
gray_gc = cv2.cvtColor(result_gc, cv2.COLOR_BGR2GRAY)
_, thresh_gc = cv2.threshold(gray_gc, 1, 255, cv2.THRESH_BINARY)
cnt_gc, _ = cv2.findContours(thresh_gc, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
if cnt_gc:
    lg = max(cnt_gc, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(lg)
    result_gc_crop = result_gc[y:y + h, x:x + w]
else:
    result_gc_crop = result_gc

cv2.imwrite(os.path.join(OUTPUT_DIR, "08_stitch_global_gain.jpg"),
            result_gc_crop)
print(f"  [OK] Stitching dengan global gain disimpan.")


# ============================================================
# LANGKAH 7: Block-Based Gain Compensation
# ============================================================
print("\n[LANGKAH 7] Mengimplementasikan block-based gain compensation...")


def gain_compensation_blocks(images, ref_idx=1, block_size=64):
    """
    Melakukan block-based gain compensation.
    Membagi gambar menjadi blok-blok kecil dan menghitung gain
    per blok untuk koreksi yang lebih lokal.

    Parameter:
    - images     : List gambar BGR
    - ref_idx    : Indeks gambar referensi
    - block_size : Ukuran blok (piksel)

    Returns:
    - compensated : List gambar yang sudah dikompensasi per blok
    """
    # Menghitung brightness per blok pada gambar referensi
    ref_gray = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY)
    h, w = ref_gray.shape[:2]

    # Menghitung jumlah blok
    n_blocks_y = (h + block_size - 1) // block_size
    n_blocks_x = (w + block_size - 1) // block_size

    # Menghitung mean brightness per blok untuk referensi
    ref_block_means = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float64)
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y1 = by * block_size
            y2 = min((by + 1) * block_size, h)
            x1 = bx * block_size
            x2 = min((bx + 1) * block_size, w)
            ref_block_means[by, bx] = np.mean(ref_gray[y1:y2, x1:x2])

    compensated = []

    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = img.astype(np.float64).copy()

        # Menghitung dan menerapkan gain per blok
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                y1 = by * block_size
                y2 = min((by + 1) * block_size, h)
                x1 = bx * block_size
                x2 = min((bx + 1) * block_size, w)

                # Mean brightness blok ini
                block_mean = np.mean(gray[y1:y2, x1:x2])

                # Menghitung gain untuk blok ini
                if block_mean > 5:  # Menghindari pembagian dengan ~0
                    gain = ref_block_means[by, bx] / block_mean
                else:
                    gain = 1.0

                # Membatasi gain agar tidak terlalu ekstrim
                gain = np.clip(gain, 0.3, 3.0)

                # Menerapkan gain pada blok
                result[y1:y2, x1:x2] = np.clip(
                    result[y1:y2, x1:x2] * gain, 0, 255
                )

        compensated.append(result.astype(np.uint8))
        mean_after = np.mean(cv2.cvtColor(compensated[-1], cv2.COLOR_BGR2GRAY))
        print(f"    Gambar {i + 1} ({exposure_labels[i]}): "
              f"block-compensated, mean_after={mean_after:.1f}")

    return compensated


# Melakukan block-based gain compensation
print("  Menghitung block-based gain factors...")
comp_blocks = gain_compensation_blocks(exposure_images, ref_idx=1, block_size=64)

# Menyimpan gambar block-compensated
for i, (img_comp, label) in enumerate(zip(comp_blocks, exposure_labels)):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"08_compensated_blocks_{label.lower()}.jpg"),
                img_comp)
print("  [OK] Gambar terkompensasi (block-based) disimpan.")


# ============================================================
# LANGKAH 8: Stitching dengan Block-Based Gain Compensation
# ============================================================
print("\n[LANGKAH 8] Melakukan stitching dengan block-based gain...")

# Menghitung homography
H_dark_blk, _ = hitung_homography(comp_blocks[0], comp_blocks[1],
                                   label="H(dark_blk→normal_blk)")
H_bright_blk, _ = hitung_homography(comp_blocks[2], comp_blocks[1],
                                     label="H(bright_blk→normal_blk)")

# Warping dan compositing
warped_dark_blk = cv2.warpPerspective(comp_blocks[0], H_tr @ H_dark_blk,
                                       (canvas_w, canvas_h))
warped_normal_blk = cv2.warpPerspective(comp_blocks[1], H_tr,
                                         (canvas_w, canvas_h))
warped_bright_blk = cv2.warpPerspective(comp_blocks[2], H_tr @ H_bright_blk,
                                         (canvas_w, canvas_h))

# Blending
canvas_blk = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
count_blk = np.zeros((canvas_h, canvas_w), dtype=np.float32)

for warped in [warped_dark_blk, warped_normal_blk, warped_bright_blk]:
    mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    for c in range(3):
        canvas_blk[:, :, c] += warped[:, :, c].astype(np.float64) * mask
    count_blk += mask

count_blk[count_blk == 0] = 1
result_blk = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
for c in range(3):
    result_blk[:, :, c] = np.clip(
        canvas_blk[:, :, c] / count_blk, 0, 255
    ).astype(np.uint8)

# Crop dan simpan
gray_blk = cv2.cvtColor(result_blk, cv2.COLOR_BGR2GRAY)
_, thresh_blk = cv2.threshold(gray_blk, 1, 255, cv2.THRESH_BINARY)
cnt_blk, _ = cv2.findContours(thresh_blk, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
if cnt_blk:
    lg = max(cnt_blk, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(lg)
    result_blk_crop = result_blk[y:y + h, x:x + w]
else:
    result_blk_crop = result_blk

cv2.imwrite(os.path.join(OUTPUT_DIR, "08_stitch_block_gain.jpg"),
            result_blk_crop)
print(f"  [OK] Stitching dengan block-based gain disimpan.")


# ============================================================
# LANGKAH 9: Stitching dengan OpenCV Stitcher (Otomatis)
# ============================================================
print("\n[LANGKAH 9] Melakukan stitching dengan OpenCV Stitcher (otomatis)...")
print("  OpenCV Stitcher menangani exposure compensation internal.")

t_start = time.time()
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status_auto, result_auto = stitcher.stitch(exposure_images)
t_auto = time.time() - t_start

if status_auto == cv2.Stitcher_OK:
    # Crop border hitam
    gray_auto = cv2.cvtColor(result_auto, cv2.COLOR_BGR2GRAY)
    _, thresh_auto = cv2.threshold(gray_auto, 1, 255, cv2.THRESH_BINARY)
    cnt_auto, _ = cv2.findContours(thresh_auto, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if cnt_auto:
        lg = max(cnt_auto, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(lg)
        result_auto_crop = result_auto[y:y + h, x:x + w]
    else:
        result_auto_crop = result_auto

    cv2.imwrite(os.path.join(OUTPUT_DIR, "08_stitch_stitcher_auto.jpg"),
                result_auto_crop)
    print(f"  [OK] Stitcher API berhasil ({t_auto:.3f} detik)")
    print(f"  Ukuran: {result_auto_crop.shape[1]}x{result_auto_crop.shape[0]}")
else:
    result_auto_crop = None
    print(f"  [WARNING] Stitcher gagal (status={status_auto})")


# ============================================================
# LANGKAH 10: Menghitung Histogram Setelah Kompensasi
# ============================================================
print("\n[LANGKAH 10] Menghitung histogram setelah kompensasi...")

# Histogram gambar setelah global gain
histograms_after_global = []
for img in comp_global:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histograms_after_global.append(hist)

# Histogram gambar setelah block-based gain
histograms_after_blocks = []
for img in comp_blocks:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histograms_after_blocks.append(hist)

print("  Histogram setelah kompensasi dihitung.")


# ============================================================
# LANGKAH 11: Tes pada Gambar Normal (Referensi)
# ============================================================
print("\n[LANGKAH 11] Melakukan tes pada gambar normal (panorama outdoor)...")

# Memuat gambar outdoor normal
outdoor_normal = []
for i in range(1, 4):
    path = os.path.join(IMAGE_DIR, f"panorama_outdoor_{i}.jpg")
    img = cv2.imread(path)
    if img is not None:
        outdoor_normal.append(img)

if len(outdoor_normal) >= 2:
    # Analisis brightness gambar normal
    print("  Brightness gambar normal:")
    for i, img in enumerate(outdoor_normal):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"    Gambar {i + 1}: mean={np.mean(gray):.1f}, std={np.std(gray):.1f}")

    # Stitching gambar normal dengan Stitcher
    stitcher_norm = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status_norm, result_norm = stitcher_norm.stitch(outdoor_normal)
    if status_norm == cv2.Stitcher_OK:
        gray_norm = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)
        _, thresh_norm = cv2.threshold(gray_norm, 1, 255, cv2.THRESH_BINARY)
        cnt_norm, _ = cv2.findContours(thresh_norm, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if cnt_norm:
            lg = max(cnt_norm, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(lg)
            result_norm_crop = result_norm[y:y + h, x:x + w]
        else:
            result_norm_crop = result_norm
        cv2.imwrite(os.path.join(OUTPUT_DIR, "08_stitch_normal_reference.jpg"),
                    result_norm_crop)
        print("  [OK] Stitching gambar normal disimpan sebagai referensi.")
    else:
        print("  [WARNING] Stitching gambar normal gagal.")
else:
    print("  [WARNING] Gambar outdoor normal tidak tersedia.")


# ============================================================
# LANGKAH 12: Membuat Grid Perbandingan Histogram
# ============================================================
print("\n[LANGKAH 12] Membuat grid perbandingan histogram...")

# Grid: 3 baris (sebelum, global gain, block gain) x 3 kolom (dark, normal, bright)
fig1, axes1 = plt.subplots(3, 3, figsize=(16, 12))
colors_hist = ['steelblue', 'green', 'coral']

for col, label in enumerate(exposure_labels):
    # Baris 0: Histogram sebelum kompensasi
    axes1[0, col].plot(histograms_before[col], color=colors_hist[col])
    mean_before = brightness_info[col]['mean_gray']
    axes1[0, col].set_title(f"Sebelum: {label} (mean={mean_before:.0f})",
                             fontsize=10)
    axes1[0, col].set_xlim([0, 256])
    axes1[0, col].axvline(x=mean_before, color='red', linestyle='--', alpha=0.5)

    # Baris 1: Histogram setelah global gain
    axes1[1, col].plot(histograms_after_global[col], color=colors_hist[col])
    mean_after_g = np.mean(cv2.cvtColor(comp_global[col], cv2.COLOR_BGR2GRAY))
    axes1[1, col].set_title(f"Global Gain: {label} (mean={mean_after_g:.0f})",
                             fontsize=10)
    axes1[1, col].set_xlim([0, 256])
    axes1[1, col].axvline(x=mean_after_g, color='red', linestyle='--', alpha=0.5)

    # Baris 2: Histogram setelah block-based gain
    axes1[2, col].plot(histograms_after_blocks[col], color=colors_hist[col])
    mean_after_b = np.mean(cv2.cvtColor(comp_blocks[col], cv2.COLOR_BGR2GRAY))
    axes1[2, col].set_title(f"Block Gain: {label} (mean={mean_after_b:.0f})",
                             fontsize=10)
    axes1[2, col].set_xlim([0, 256])
    axes1[2, col].axvline(x=mean_after_b, color='red', linestyle='--', alpha=0.5)

plt.suptitle("Percobaan 8: Histogram Sebelum dan Setelah Exposure Compensation",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_grid_histogram_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid histogram disimpan.")
plt.close()


# ============================================================
# LANGKAH 13: Membuat Grid Perbandingan Stitching
# ============================================================
print("\n[LANGKAH 13] Membuat grid perbandingan stitching...")

# Grid: 2x2 (no comp, global gain, block gain, stitcher auto)
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

# (0,0): Tanpa kompensasi
axes2[0, 0].imshow(cv2.cvtColor(result_no_comp_crop, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("Tanpa Exposure Compensation", fontsize=11)
axes2[0, 0].axis("off")

# (0,1): Global gain
axes2[0, 1].imshow(cv2.cvtColor(result_gc_crop, cv2.COLOR_BGR2RGB))
axes2[0, 1].set_title(f"Global Gain Compensation\n"
                       f"Gains: {[f'{g:.2f}' for g in gains_global]}", fontsize=11)
axes2[0, 1].axis("off")

# (1,0): Block-based gain
axes2[1, 0].imshow(cv2.cvtColor(result_blk_crop, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("Block-Based Gain Compensation (64x64)", fontsize=11)
axes2[1, 0].axis("off")

# (1,1): OpenCV Stitcher auto
if result_auto_crop is not None:
    axes2[1, 1].imshow(cv2.cvtColor(result_auto_crop, cv2.COLOR_BGR2RGB))
    axes2[1, 1].set_title("OpenCV Stitcher (Otomatis + BA)", fontsize=11)
else:
    axes2[1, 1].text(0.5, 0.5, "GAGAL", ha='center', va='center', fontsize=16)
    axes2[1, 1].set_title("OpenCV Stitcher (GAGAL)", fontsize=11)
axes2[1, 1].axis("off")

plt.suptitle("Percobaan 8: Perbandingan Metode Exposure Compensation",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_grid_stitching_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid stitching comparison disimpan.")
plt.close()


# ============================================================
# LANGKAH 14: Membuat Grid Gambar Input vs Compensated
# ============================================================
print("\n[LANGKAH 14] Membuat grid gambar sebelum vs setelah kompensasi...")

# Grid: 3 baris (original, global, block) x 3 kolom (dark, normal, bright)
fig3, axes3 = plt.subplots(3, 3, figsize=(16, 12))

for col in range(3):
    # Baris 0: Original
    axes3[0, col].imshow(cv2.cvtColor(exposure_images[col], cv2.COLOR_BGR2RGB))
    mean_orig = brightness_info[col]['mean_gray']
    axes3[0, col].set_title(f"Original {exposure_labels[col]}\nmean={mean_orig:.0f}",
                             fontsize=10)
    axes3[0, col].axis("off")

    # Baris 1: Global gain compensated
    axes3[1, col].imshow(cv2.cvtColor(comp_global[col], cv2.COLOR_BGR2RGB))
    mean_gc = np.mean(cv2.cvtColor(comp_global[col], cv2.COLOR_BGR2GRAY))
    axes3[1, col].set_title(f"Global Gain {exposure_labels[col]}\n"
                             f"mean={mean_gc:.0f}, gain={gains_global[col]:.2f}",
                             fontsize=10)
    axes3[1, col].axis("off")

    # Baris 2: Block-based gain compensated
    axes3[2, col].imshow(cv2.cvtColor(comp_blocks[col], cv2.COLOR_BGR2RGB))
    mean_blk = np.mean(cv2.cvtColor(comp_blocks[col], cv2.COLOR_BGR2GRAY))
    axes3[2, col].set_title(f"Block Gain {exposure_labels[col]}\nmean={mean_blk:.0f}",
                             fontsize=10)
    axes3[2, col].axis("off")

plt.suptitle("Percobaan 8: Gambar Sebelum vs Setelah Exposure Compensation",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_grid_before_after_compensation.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid before/after compensation disimpan.")
plt.close()


# ============================================================
# LANGKAH 15: Ringkasan dan Statistik
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 8: EXPOSURE COMPENSATION")
print("=" * 65)

# Tabel brightness analysis
print("\n  Tabel Analisis Brightness:")
print(f"  {'Gambar':<10} | {'Mean Orig':>10} | {'Mean Global':>12} | "
      f"{'Mean Block':>11} | {'Gain':>6}")
print(f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 11}-+-{'-' * 6}")
for i in range(3):
    mean_orig = brightness_info[i]['mean_gray']
    mean_gc = np.mean(cv2.cvtColor(comp_global[i], cv2.COLOR_BGR2GRAY))
    mean_blk = np.mean(cv2.cvtColor(comp_blocks[i], cv2.COLOR_BGR2GRAY))
    gain = gains_global[i]
    print(f"  {exposure_labels[i]:<10} | {mean_orig:>10.1f} | "
          f"{mean_gc:>12.1f} | {mean_blk:>11.1f} | {gain:>6.3f}")

# Tabel evaluasi metode
print(f"\n  Tabel Evaluasi Metode:")
print(f"  {'Metode':<30} | {'Kualitas':>10} | {'Catatan':<30}")
print(f"  {'-' * 30}-+-{'-' * 10}-+-{'-' * 30}")
print(f"  {'Tanpa Compensation':<30} | {'Rendah':>10} | Seam exposure jelas terlihat")
print(f"  {'Global Gain':<30} | {'Sedang':>10} | Seragam, tapi lokal kurang tepat")
print(f"  {'Block-Based Gain (64px)':<30} | {'Baik':>10} | Koreksi lokal lebih tepat")
print(f"  {'OpenCV Stitcher (Otomatis)':<30} | {'Terbaik':>10} | BA + exposure comp. built-in")

# Penjelasan konsep
print("\n  Konsep Exposure Compensation:")
print("  - Perbedaan exposure menyebabkan brightness seam pada panorama")
print("  - Global gain: mengalikan faktor konstan pada seluruh gambar")
print("  - Block-based gain: gain berbeda per blok untuk koreksi lokal")
print("  - OpenCV Stitcher menggabungkan BA + exposure comp. + blending")
print("  - Gain = brightness_ref / brightness_src")
print("  - Semakin halus kompensasi (lokal), semakin baik hasilnya")

# Daftar output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("08_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    np.mean()             → Menghitung rata-rata brightness")
print("    cv2.cvtColor()        → Konversi BGR↔HSV↔Gray")
print("    cv2.calcHist()        → Menghitung histogram intensitas")
print("    cv2.distanceTransform → Weight mask untuk feather blending")
print("    cv2.Stitcher_create() → Stitcher API (otomatis)")
print("    gain = ref/src        → Rumus gain compensation sederhana")
print("=" * 65)
