"""
==========================================================================
PERCOBAAN 3: TEKNIK BLENDING - PERBANDINGAN
==========================================================================
Program ini membandingkan berbagai teknik blending yang digunakan saat
menggabungkan gambar dalam image stitching. Blending yang baik menghasilkan
transisi mulus pada area overlap (seam) tanpa artefak visual.

Konsep yang dipelajari:
- Mengapa blending diperlukan dalam image stitching
- Perbedaan berbagai metode blending
- Konsep Laplacian pyramid untuk multi-band blending
- Analisis visual kualitas seam pada area overlap

Teknik blending yang dibandingkan:
1. No Blending       - Direct overlay (gambar kanan menutupi kiri)
2. Average Blending  - Rata-rata 50-50 di area overlap
3. Feather Blending  - Gradien linear alpha di area overlap
4. Multi-band Blend  - Laplacian pyramid blending (paling halus)

Fungsi utama yang dipelajari:
- cv2.warpPerspective()  : Warping gambar ke perspektif target
- cv2.GaussianBlur()     : Membuat kernel Gaussian untuk feather
- cv2.pyrDown()          : Downsampling gambar (Gaussian pyramid)
- cv2.pyrUp()            : Upsampling gambar (rekonstruksi pyramid)
- np.where()             : Seleksi piksel berdasarkan kondisi (mask)
- np.linspace()          : Membuat gradien linear untuk feather blend
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan warping
import cv2

# Mengimpor library NumPy untuk operasi array, matriks, dan mask
import numpy as np

# Mengimpor library os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi grid perbandingan
import matplotlib.pyplot as plt

# Mengimpor modul time untuk mengukur waktu eksekusi setiap metode
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
print("PERCOBAAN 3: TEKNIK BLENDING - PERBANDINGAN")
print("=" * 65)

# ============================================================
# LANGKAH 1: Memuat Gambar dan Menghitung Homography
# ============================================================
print("\n[LANGKAH 1] Memuat gambar dan menghitung homography...")

# Membaca gambar pasangan untuk stitching
img_left = cv2.imread(os.path.join(IMAGE_DIR, "pair_left.jpg"))
img_right = cv2.imread(os.path.join(IMAGE_DIR, "pair_right.jpg"))

# Memastikan gambar berhasil dimuat
if img_left is None or img_right is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi ukuran gambar
print(f"  Gambar kiri  : {img_left.shape[1]}x{img_left.shape[0]}")
print(f"  Gambar kanan : {img_right.shape[1]}x{img_right.shape[0]}")

# Mengkonversi ke grayscale untuk deteksi fitur
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Membuat detektor SIFT untuk mendeteksi fitur
sift = cv2.SIFT_create()

# Mendeteksi keypoints dan deskriptor pada kedua gambar
kp_left, desc_left = sift.detectAndCompute(gray_left, None)
kp_right, desc_right = sift.detectAndCompute(gray_right, None)

# Menampilkan jumlah fitur yang terdeteksi
print(f"  Fitur kiri   : {len(kp_left)}")
print(f"  Fitur kanan  : {len(kp_right)}")

# Mengonfigurasi FLANN matcher untuk pencocokan fitur
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Melakukan pencocokan k-nearest neighbors (k=2)
matches_knn = flann.knnMatch(desc_left, desc_right, k=2)

# Menerapkan Lowe's ratio test untuk memfilter kecocokan
good_matches = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"  Good matches : {len(good_matches)}")

# Mengekstrak titik-titik korespondensi untuk estimasi homography
src_pts = np.float32([kp_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Mengestimasi homography dari gambar kiri ke gambar kanan
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print(f"  Inliers      : {mask.ravel().sum()}")

# ============================================================
# LANGKAH 2: Menyiapkan Canvas dan Warping
# ============================================================
print("\n[LANGKAH 2] Menyiapkan canvas dan melakukan warping...")

# Mendapatkan dimensi gambar
h_left, w_left = img_left.shape[:2]
h_right, w_right = img_right.shape[:2]

# Menghitung posisi sudut gambar kiri setelah transformasi
corners_left = np.float32([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]]).reshape(-1, 1, 2)
corners_left_t = cv2.perspectiveTransform(corners_left, H)

# Menghitung posisi sudut gambar kanan
corners_right = np.float32([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]]).reshape(-1, 1, 2)

# Menggabungkan semua sudut untuk menentukan batas canvas
all_corners = np.concatenate([corners_left_t, corners_right], axis=0)
x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
x_min = min(x_min, 0)
y_min = min(y_min, 0)

# Menghitung ukuran canvas
canvas_w = x_max - x_min
canvas_h = y_max - y_min

# Membuat matriks translasi untuk area positif
T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

# Menghitung homography gabungan (translasi + homography)
H_final = T @ H

# Melakukan warping gambar kiri ke canvas
warped_left = cv2.warpPerspective(img_left, H_final, (canvas_w, canvas_h))

# Menampilkan info canvas
print(f"  Ukuran canvas: {canvas_w} x {canvas_h}")

# Menghitung offset untuk gambar kanan
offset_x = -x_min
offset_y = -y_min

# ============================================================
# LANGKAH 3: Membuat Mask untuk Overlap Detection
# ============================================================
print("\n[LANGKAH 3] Membuat mask area overlap...")

# Mask gambar kiri: area yang terisi setelah warping (non-hitam)
mask_left = (cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

# Mask gambar kanan: area yang ditempati gambar kanan pada canvas
mask_right = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
y_end = min(offset_y + h_right, canvas_h)
x_end = min(offset_x + w_right, canvas_w)
mask_right[offset_y:y_end, offset_x:x_end] = 1

# Mask overlap: area irisan kedua gambar
mask_overlap = (mask_left & mask_right).astype(np.uint8)

# Mask hanya kiri (tanpa overlap)
mask_left_only = (mask_left & ~mask_right).astype(np.uint8)

# Mask hanya kanan (tanpa overlap)
mask_right_only = (~mask_left & mask_right).astype(np.uint8)

# Menampilkan statistik mask
overlap_count = np.sum(mask_overlap)
print(f"  Piksel hanya kiri  : {np.sum(mask_left_only)}")
print(f"  Piksel hanya kanan : {np.sum(mask_right_only)}")
print(f"  Piksel overlap     : {overlap_count}")

# Menyiapkan gambar kanan pada canvas
canvas_right = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
canvas_right[offset_y:y_end, offset_x:x_end] = img_right[:y_end - offset_y, :x_end - offset_x]

# ============================================================
# LANGKAH 4: Metode 1 - No Blending (Direct Overlay)
# ============================================================
print("\n[LANGKAH 4] Metode 1: No Blending (Direct Overlay)...")

# Mengukur waktu eksekusi
waktu_mulai = time.time()

# Membuat canvas dengan gambar kiri yang sudah di-warp
result_no_blend = warped_left.copy()

# Menimpa area gambar kanan langsung (overwrite) tanpa blending apapun
# Piksel gambar kanan langsung menggantikan piksel gambar kiri di area overlap
result_no_blend[offset_y:y_end, offset_x:x_end] = img_right[:y_end - offset_y, :x_end - offset_x]

# Menghitung waktu
waktu_no_blend = time.time() - waktu_mulai

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "03_no_blending.jpg"), result_no_blend)
print(f"  Waktu: {waktu_no_blend*1000:.2f} ms")
print("  [OK] Hasil no blending disimpan.")

# ============================================================
# LANGKAH 5: Metode 2 - Average Blending (50-50)
# ============================================================
print("\n[LANGKAH 5] Metode 2: Average Blending (50-50)...")

# Mengukur waktu eksekusi
waktu_mulai = time.time()

# Memulai dari salinan gambar kiri yang sudah di-warp
result_average = warped_left.copy()

# Menempatkan gambar kanan di area non-overlap
# Area hanya kanan: salin langsung gambar kanan
for c in range(3):
    result_average[:, :, c] = np.where(
        mask_right_only == 1,
        canvas_right[:, :, c],
        result_average[:, :, c]
    )

# Area overlap: rata-rata 50% gambar kiri + 50% gambar kanan
# Ini menciptakan transisi yang lebih halus dibanding direct overlay
for c in range(3):
    result_average[:, :, c] = np.where(
        mask_overlap == 1,
        (warped_left[:, :, c].astype(np.float32) * 0.5 +
         canvas_right[:, :, c].astype(np.float32) * 0.5).astype(np.uint8),
        result_average[:, :, c]
    )

# Menghitung waktu
waktu_average = time.time() - waktu_mulai

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "03_average_blending.jpg"), result_average)
print(f"  Waktu: {waktu_average*1000:.2f} ms")
print("  [OK] Hasil average blending disimpan.")

# ============================================================
# LANGKAH 6: Metode 3 - Feather Blending (Linear Gradient)
# ============================================================
print("\n[LANGKAH 6] Metode 3: Feather Blending (Gradien Linear)...")

# Mengukur waktu eksekusi
waktu_mulai = time.time()

# Membuat alpha map untuk feather blending
# Alpha berubah secara linear dari 1 (100% kiri) ke 0 (100% kanan) di area overlap
alpha_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

# Menggunakan distance transform untuk membuat gradien yang halus
# Distance transform menghitung jarak setiap piksel ke tepi terdekat
dist_left = cv2.distanceTransform(mask_left, cv2.DIST_L2, 5)
dist_right = cv2.distanceTransform(mask_right, cv2.DIST_L2, 5)

# Menghitung alpha sebagai rasio jarak
# Di area overlap, piksel yang lebih dekat ke "pusat" gambar kiri
# mendapat bobot lebih tinggi untuk gambar kiri, dan sebaliknya
total_dist = dist_left + dist_right

# Menghindari pembagian dengan nol
total_dist[total_dist == 0] = 1

# Alpha untuk gambar kiri: semakin dekat ke tepi kiri → semakin tinggi
alpha_left = dist_left / total_dist

# Memastikan area non-overlap memiliki alpha yang benar
alpha_left[mask_left_only == 1] = 1.0   # Area hanya kiri: 100% kiri
alpha_left[mask_right_only == 1] = 0.0  # Area hanya kanan: 0% kiri

# Membuat hasil feather blending
result_feather = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
for c in range(3):
    # Blending: alpha * kiri + (1-alpha) * kanan
    blended = (alpha_left * warped_left[:, :, c].astype(np.float32) +
               (1 - alpha_left) * canvas_right[:, :, c].astype(np.float32))
    result_feather[:, :, c] = np.clip(blended, 0, 255).astype(np.uint8)

# Menghitung waktu
waktu_feather = time.time() - waktu_mulai

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "03_feather_blending.jpg"), result_feather)
print(f"  Waktu: {waktu_feather*1000:.2f} ms")
print("  [OK] Hasil feather blending disimpan.")

# Menyimpan visualisasi alpha map
alpha_vis = (alpha_left * 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "03_feather_alpha_map.jpg"), alpha_vis)
print("  [OK] Visualisasi alpha map disimpan.")


# ============================================================
# LANGKAH 7: Metode 4 - Multi-Band Blending (Laplacian Pyramid)
# ============================================================
print("\n[LANGKAH 7] Metode 4: Multi-Band Blending (Laplacian Pyramid)...")


def bangun_gaussian_pyramid(img, levels):
    """
    Membangun Gaussian pyramid dari sebuah gambar.
    Gaussian pyramid dibuat dengan menerapkan Gaussian blur + downscale
    secara berulang. Level 0 = gambar asli, level tertinggi = paling kecil.

    Parameter:
    - img    : Gambar input (BGR atau grayscale)
    - levels : Jumlah level pyramid
    """
    # Level 0 adalah gambar asli
    pyramid = [img.copy()]

    # Membuat level-level berikutnya dengan downsampling
    for i in range(levels):
        # pyrDown melakukan Gaussian blur lalu mengecilkan ukuran 2x
        img = cv2.pyrDown(img)
        pyramid.append(img)

    return pyramid


def bangun_laplacian_pyramid(img, levels):
    """
    Membangun Laplacian pyramid dari sebuah gambar.
    Laplacian pyramid menyimpan detail (frekuensi tinggi) pada setiap level.
    L_i = G_i - expand(G_{i+1})

    Parameter:
    - img    : Gambar input
    - levels : Jumlah level pyramid
    """
    # Membangun Gaussian pyramid terlebih dahulu
    gaussian_pyr = bangun_gaussian_pyramid(img, levels)
    laplacian_pyr = []

    # Menghitung Laplacian pada setiap level kecuali yang terakhir
    for i in range(levels):
        # Upsampling level berikutnya ke ukuran level saat ini
        expanded = cv2.pyrUp(gaussian_pyr[i + 1])

        # Menyesuaikan ukuran jika terjadi perbedaan kecil akibat pembulatan
        h, w = gaussian_pyr[i].shape[:2]
        expanded = cv2.resize(expanded, (w, h))

        # Laplacian = gambar saat ini - upsampled dari level berikutnya
        # Ini menyimpan detail (frekuensi tinggi) pada setiap level
        laplacian = cv2.subtract(gaussian_pyr[i], expanded)
        laplacian_pyr.append(laplacian)

    # Level terakhir = gambar Gaussian terkecil (base/low frequency)
    laplacian_pyr.append(gaussian_pyr[levels])

    return laplacian_pyr


def rekonstruksi_dari_laplacian(laplacian_pyr):
    """
    Merekonstruksi gambar dari Laplacian pyramid.
    Proses: mulai dari level terkecil, upscale, lalu tambahkan detail.

    Parameter:
    - laplacian_pyr : List Laplacian pyramid (level terkecil di akhir)
    """
    # Mulai dari level terkecil (base image)
    img = laplacian_pyr[-1]

    # Iterasi dari level terkecil kedua hingga level terbesar
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        # Upsampling gambar saat ini
        img = cv2.pyrUp(img)

        # Menyesuaikan ukuran
        h, w = laplacian_pyr[i].shape[:2]
        img = cv2.resize(img, (w, h))

        # Menambahkan detail dari level Laplacian
        img = cv2.add(img, laplacian_pyr[i])

    return img


def multi_band_blend(img1, img2, mask_blend, levels=3):
    """
    Melakukan multi-band blending menggunakan Laplacian pyramid.
    Teknik ini memblend frekuensi rendah dan tinggi secara terpisah
    sehingga menghasilkan transisi yang paling halus.

    Parameter:
    - img1       : Gambar pertama (warped left)
    - img2       : Gambar kedua (canvas right)
    - mask_blend : Mask alpha (float32, 0-1) untuk blending
    - levels     : Jumlah level pyramid (default=3)
    """
    # Memastikan tipe data yang konsisten (float32)
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    # Membuat mask 3 channel untuk per-piksel blending
    if len(mask_blend.shape) == 2:
        mask_3ch = np.stack([mask_blend] * 3, axis=-1)
    else:
        mask_3ch = mask_blend

    # Membangun Laplacian pyramid untuk kedua gambar
    lap_pyr1 = bangun_laplacian_pyramid(img1_f, levels)
    lap_pyr2 = bangun_laplacian_pyramid(img2_f, levels)

    # Membangun Gaussian pyramid untuk mask (untuk blending per level)
    mask_pyr = bangun_gaussian_pyramid(mask_3ch.astype(np.float32), levels)

    # Melakukan blending pada setiap level pyramid
    blended_pyr = []
    for la, lb, mask_level in zip(lap_pyr1, lap_pyr2, mask_pyr):
        # Menyesuaikan ukuran mask jika perlu
        h, w = la.shape[:2]
        mask_resized = cv2.resize(mask_level, (w, h))
        if len(mask_resized.shape) == 2:
            mask_resized = np.stack([mask_resized] * 3, axis=-1)

        # Blending pada level ini: mask * img1 + (1-mask) * img2
        blended_level = la * mask_resized + lb * (1 - mask_resized)
        blended_pyr.append(blended_level)

    # Merekonstruksi gambar dari pyramid yang sudah di-blend
    result = rekonstruksi_dari_laplacian(blended_pyr)

    # Mengkonversi kembali ke uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


# Menjalankan multi-band blending dengan 3 level (default)
waktu_mulai = time.time()

# Menggunakan alpha_left sebagai mask blending
result_multiband = multi_band_blend(warped_left, canvas_right, alpha_left, levels=3)

# Menghitung waktu
waktu_multiband = time.time() - waktu_mulai

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "03_multiband_blending_3level.jpg"), result_multiband)
print(f"  Waktu (3 level): {waktu_multiband*1000:.2f} ms")
print("  [OK] Hasil multi-band blending (3 level) disimpan.")

# ============================================================
# LANGKAH 8: Perbandingan Multi-Band dengan Berbagai Level
# ============================================================
print("\n[LANGKAH 8] Membandingkan multi-band blending berbagai level...")

# Dictionary untuk menyimpan hasil setiap level
multiband_results = {}

# Menguji dengan level 2, 3, 4, dan 5
for lvl in [2, 3, 4, 5]:
    # Mengukur waktu eksekusi
    waktu_mulai = time.time()

    # Melakukan multi-band blending dengan jumlah level tertentu
    result_lvl = multi_band_blend(warped_left, canvas_right, alpha_left, levels=lvl)

    # Menghitung waktu
    waktu_lvl = time.time() - waktu_mulai

    # Menyimpan hasil
    multiband_results[lvl] = {
        'result': result_lvl,
        'waktu': waktu_lvl
    }
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"03_multiband_{lvl}level.jpg"), result_lvl)
    print(f"  Level {lvl}: {waktu_lvl*1000:.2f} ms")

print("  [OK] Semua variasi multi-band disimpan.")

# ============================================================
# LANGKAH 9: Zoom ke Area Seam untuk Analisis Detail
# ============================================================
print("\n[LANGKAH 9] Melakukan zoom ke area seam untuk analisis...")

# Mencari batas area overlap untuk zoom
overlap_coords = np.where(mask_overlap > 0)
if len(overlap_coords[0]) > 0:
    # Menghitung bounding box area overlap
    y_min_ov = overlap_coords[0].min()
    y_max_ov = overlap_coords[0].max()
    x_min_ov = overlap_coords[1].min()
    x_max_ov = overlap_coords[1].max()

    # Menghitung area zoom (tengah overlap, ukuran terbatas)
    cy = (y_min_ov + y_max_ov) // 2
    cx = (x_min_ov + x_max_ov) // 2
    zoom_size = 150  # Ukuran setengah kotak zoom

    # Menentukan batas zoom dengan clipping
    zy1 = max(0, cy - zoom_size)
    zy2 = min(canvas_h, cy + zoom_size)
    zx1 = max(0, cx - zoom_size)
    zx2 = min(canvas_w, cx + zoom_size)

    # Memotong area seam dari setiap metode
    zoom_no_blend = result_no_blend[zy1:zy2, zx1:zx2]
    zoom_average = result_average[zy1:zy2, zx1:zx2]
    zoom_feather = result_feather[zy1:zy2, zx1:zx2]
    zoom_multiband = result_multiband[zy1:zy2, zx1:zx2]

    # Menyimpan zoom area untuk setiap metode
    cv2.imwrite(os.path.join(OUTPUT_DIR, "03_zoom_no_blend.jpg"), zoom_no_blend)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "03_zoom_average.jpg"), zoom_average)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "03_zoom_feather.jpg"), zoom_feather)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "03_zoom_multiband.jpg"), zoom_multiband)
    print(f"  Area zoom: ({zx1},{zy1}) - ({zx2},{zy2})")
    print("  [OK] Semua zoom area disimpan.")
else:
    print("  [WARNING] Tidak ada area overlap terdeteksi untuk zoom.")
    zoom_no_blend = zoom_average = zoom_feather = zoom_multiband = None

# ============================================================
# LANGKAH 10: Grid Perbandingan Utama (2x2: Empat Metode)
# ============================================================
print("\n[LANGKAH 10] Membuat grid perbandingan utama...")

# Membuat figure 2x2 untuk keempat metode blending
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))

# Daftar metode dan hasilnya
metode_blending = [
    ("No Blending (Direct Overlay)", result_no_blend, waktu_no_blend),
    ("Average Blending (50-50)", result_average, waktu_average),
    ("Feather Blending (Gradien)", result_feather, waktu_feather),
    (f"Multi-Band Blend (3 Level)", result_multiband, waktu_multiband)
]

# Menampilkan setiap metode pada subplot
for idx, (nama, hasil, waktu) in enumerate(metode_blending):
    row = idx // 2
    col = idx % 2

    # Mengkonversi BGR ke RGB untuk matplotlib
    axes1[row, col].imshow(cv2.cvtColor(hasil, cv2.COLOR_BGR2RGB))
    axes1[row, col].set_title(f"{nama}\n(Waktu: {waktu*1000:.2f} ms)", fontsize=11)
    axes1[row, col].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 3: Perbandingan 4 Teknik Blending",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid perbandingan utama
plt.savefig(os.path.join(OUTPUT_DIR, "03_grid_perbandingan_blending.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid perbandingan utama disimpan.")
plt.close()

# ============================================================
# LANGKAH 11: Grid Zoom Area Seam
# ============================================================
print("\n[LANGKAH 11] Membuat grid zoom area seam...")

if zoom_no_blend is not None:
    # Membuat figure 2x2 untuk zoom area seam
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # Daftar zoom images
    zoom_images = [
        ("No Blending (Zoom)", zoom_no_blend),
        ("Average Blending (Zoom)", zoom_average),
        ("Feather Blending (Zoom)", zoom_feather),
        ("Multi-Band Blend (Zoom)", zoom_multiband)
    ]

    for idx, (nama, zoom_img) in enumerate(zoom_images):
        row = idx // 2
        col = idx % 2
        axes2[row, col].imshow(cv2.cvtColor(zoom_img, cv2.COLOR_BGR2RGB))
        axes2[row, col].set_title(nama, fontsize=12)
        axes2[row, col].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 3: Zoom Area Seam/Overlap",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Menyimpan grid zoom
    plt.savefig(os.path.join(OUTPUT_DIR, "03_grid_zoom_seam.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Grid zoom area seam disimpan.")
    plt.close()

# ============================================================
# LANGKAH 12: Grid Perbandingan Multi-Band Level
# ============================================================
print("\n[LANGKAH 12] Membuat grid perbandingan level multi-band...")

# Membuat figure 2x2 untuk berbagai level multi-band
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))

level_list = sorted(multiband_results.keys())
for idx, lvl in enumerate(level_list):
    row = idx // 2
    col = idx % 2
    data = multiband_results[lvl]

    # Menampilkan hasil pada subplot
    axes3[row, col].imshow(cv2.cvtColor(data['result'], cv2.COLOR_BGR2RGB))
    axes3[row, col].set_title(
        f"Multi-Band Level {lvl}\n(Waktu: {data['waktu']*1000:.2f} ms)",
        fontsize=11
    )
    axes3[row, col].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 3: Perbandingan Level Pyramid Multi-Band Blending",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid level
plt.savefig(os.path.join(OUTPUT_DIR, "03_grid_multiband_level.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid perbandingan level disimpan.")
plt.close()

# ============================================================
# LANGKAH 13: Tabel Kualitas dan Ringkasan
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 3: TEKNIK BLENDING - PERBANDINGAN")
print("=" * 65)

# Tabel perbandingan metode blending
print("\n  Tabel Perbandingan Metode Blending:")
print(f"  {'Metode':<30} | {'Waktu (ms)':>10} | {'Kualitas Seam':<18}")
print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*18}")
print(f"  {'No Blending':<30} | {waktu_no_blend*1000:>9.2f} | {'Buruk (terlihat)':<18}")
print(f"  {'Average (50-50)':<30} | {waktu_average*1000:>9.2f} | {'Sedang (ghosting)':<18}")
print(f"  {'Feather (Gradien)':<30} | {waktu_feather*1000:>9.2f} | {'Baik (halus)':<18}")
print(f"  {'Multi-Band (3 Level)':<30} | {waktu_multiband*1000:>9.2f} | {'Sangat Baik':<18}")

# Tabel perbandingan level multi-band
print(f"\n  Perbandingan Level Multi-Band Blending:")
print(f"  {'Level':>6} | {'Waktu (ms)':>10}")
print(f"  {'-'*6}-+-{'-'*10}")
for lvl in level_list:
    print(f"  {lvl:>6} | {multiband_results[lvl]['waktu']*1000:>9.2f}")

# Penjelasan metode
print("\n  Penjelasan tiap metode:")
print("  1. No Blending    : Gambar kanan langsung ditimpa di atas kiri")
print("                      → Seam terlihat jelas, artifact warna")
print("  2. Average (50-50): Rata-rata piksel di area overlap")
print("                      → Ada efek ghosting jika ada perbedaan")
print("  3. Feather Blend  : Gradien linear alpha di area overlap")
print("                      → Transisi halus, ada sedikit blur")
print("  4. Multi-Band     : Blending per frekuensi via Laplacian pyramid")
print("                      → Paling halus, mempertahankan detail")

# Daftar output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("03_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.warpPerspective()     → Warping perspektif gambar")
print("    cv2.distanceTransform()   → Menghitung jarak ke tepi (untuk alpha)")
print("    cv2.pyrDown() / pyrUp()   → Operasi Gaussian pyramid")
print("    bangun_laplacian_pyramid()→ Membangun Laplacian pyramid (custom)")
print("    multi_band_blend()        → Multi-band blending (custom)")
print("    np.where()                → Seleksi piksel berdasarkan mask")
print("=" * 65)
