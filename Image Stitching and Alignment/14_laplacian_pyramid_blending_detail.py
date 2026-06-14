"""
==========================================================================
PERCOBAAN 14: LAPLACIAN PYRAMID BLENDING - DETAIL
==========================================================================
Program ini mempelajari multi-band blending menggunakan Laplacian pyramid
secara mendalam. Laplacian pyramid memungkinkan blending dua gambar pada
frekuensi yang berbeda-beda, sehingga transisi terlihat sangat halus.

Konsep yang dipelajari:
- Gaussian pyramid: representasi multi-resolusi gambar
- Laplacian pyramid: menyimpan detail pada setiap level
- Rekonstruksi gambar dari Laplacian pyramid
- Multi-band blending: blending per level frekuensi
- Pengaruh jumlah level pada kualitas blending
- Aplikasi pada panorama stitching

Fungsi utama yang dipelajari:
- cv2.pyrDown()       : Downsampling gambar (Gaussian pyramid)
- cv2.pyrUp()         : Upsampling gambar (untuk rekonstruksi)
- cv2.subtract()      : Menghitung Laplacian sebagai selisih Gaussian
- cv2.add()           : Rekonstruksi gambar dari Laplacian pyramid
- cv2.GaussianBlur()  : Smoothing untuk mask pyramid
- cv2.resize()        : Resize gambar untuk matching ukuran
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
print("PERCOBAAN 14: LAPLACIAN PYRAMID BLENDING - DETAIL")
print("=" * 65)


# ============================================================
# LANGKAH 1: Implementasi Build Gaussian Pyramid
# ============================================================
print("\n[LANGKAH 1] Implementasi build_gaussian_pyramid()...")

def build_gaussian_pyramid(img, levels):
    """
    Membangun Gaussian pyramid dari gambar.
    Gaussian pyramid adalah urutan gambar yang semakin kecil,
    di mana setiap level di-smooth dan di-downsample 2x.

    Parameter:
    - img    : Gambar input (BGR)
    - levels : Jumlah level pyramid (termasuk gambar asli)

    Returns:
    - pyramid : List gambar dari resolusi tertinggi ke terendah
    """
    # Level pertama adalah gambar asli
    pyramid = [img.copy()]

    # Membuat setiap level dengan downsampling
    current = img.copy()
    for i in range(1, levels):
        # pyrDown melakukan Gaussian blur + downsampling 2x
        # Ukuran gambar menjadi setengahnya di setiap level
        current = cv2.pyrDown(current)
        pyramid.append(current)

    return pyramid


# Memuat gambar untuk demo pyramid
img_left = cv2.imread(os.path.join(IMAGE_DIR, "pair_left.jpg"))
img_right = cv2.imread(os.path.join(IMAGE_DIR, "pair_right.jpg"))

if img_left is None or img_right is None:
    print("[ERROR] Gambar pair tidak ditemukan! Jalankan download_image.py.")
    exit()

# Menyesuaikan ukuran agar genap dan sama (kebutuhan pyramid)
# Ukuran harus kelipatan 2^levels agar pyrDown/pyrUp konsisten
LEVELS = 5
target_h = img_left.shape[0]
target_w = img_left.shape[1]

# Membuat ukuran kelipatan 2^LEVELS
divisor = 2 ** LEVELS
target_h = (target_h // divisor) * divisor
target_w = (target_w // divisor) * divisor

# Mengubah ukuran kedua gambar
img_left = cv2.resize(img_left, (target_w, target_h))
img_right = cv2.resize(img_right, (target_w, target_h))

print(f"  Gambar kiri  : {img_left.shape[1]}x{img_left.shape[0]}")
print(f"  Gambar kanan : {img_right.shape[1]}x{img_right.shape[0]}")
print(f"  Jumlah level : {LEVELS}")

# Membangun Gaussian pyramid untuk gambar kiri
gauss_left = build_gaussian_pyramid(img_left, LEVELS)
print(f"\n  Gaussian Pyramid (gambar kiri):")
for i, g in enumerate(gauss_left):
    print(f"    Level {i}: {g.shape[1]}x{g.shape[0]}")


# ============================================================
# LANGKAH 2: Implementasi Build Laplacian Pyramid
# ============================================================
print("\n[LANGKAH 2] Implementasi build_laplacian_pyramid()...")

def build_laplacian_pyramid(gaussian_pyramid):
    """
    Membangun Laplacian pyramid dari Gaussian pyramid.
    Laplacian pyramid menyimpan detail (frekuensi tinggi) pada setiap level.

    Laplacian_i = Gaussian_i - pyrUp(Gaussian_{i+1})
    Level terakhir = Gaussian level terakhir (low-frequency residual)

    Parameter:
    - gaussian_pyramid : Gaussian pyramid (list gambar)

    Returns:
    - laplacian : Laplacian pyramid (list gambar)
    """
    laplacian = []
    levels = len(gaussian_pyramid)

    for i in range(levels - 1):
        # Mengambil level saat ini dari Gaussian pyramid
        current = gaussian_pyramid[i]

        # Melakukan upsampling level berikutnya agar ukurannya cocok
        # pyrUp melakukan upsampling 2x + Gaussian blur
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1])

        # Menyesuaikan ukuran (kadang beda 1 piksel setelah pyrUp)
        h, w = current.shape[:2]
        upsampled = cv2.resize(upsampled, (w, h))

        # Laplacian = selisih antara Gaussian level saat ini dan upsampled level berikutnya
        # Ini menyimpan detail yang hilang saat downsampling
        lap = cv2.subtract(current, upsampled)
        laplacian.append(lap)

    # Level terakhir: menyimpan Gaussian level terakhir (low-frequency residual)
    # Ini diperlukan untuk rekonstruksi sempurna
    laplacian.append(gaussian_pyramid[-1])

    return laplacian


# Membangun Laplacian pyramid untuk gambar kiri
lap_left = build_laplacian_pyramid(gauss_left)
print(f"  Laplacian Pyramid (gambar kiri):")
for i, l in enumerate(lap_left):
    print(f"    Level {i}: {l.shape[1]}x{l.shape[0]} "
          f"(mean={np.mean(l):.1f}, std={np.std(l):.1f})")


# ============================================================
# LANGKAH 3: Implementasi Rekonstruksi dari Laplacian Pyramid
# ============================================================
print("\n[LANGKAH 3] Implementasi reconstruct_from_laplacian()...")

def reconstruct_from_laplacian(laplacian_pyramid):
    """
    Merekonstruksi gambar dari Laplacian pyramid.
    Proses: dimulai dari level paling bawah (terendah), upsample,
    dan tambahkan Laplacian level di atasnya secara berurutan.

    Rekonstruksi: G_i = L_i + pyrUp(G_{i+1})

    Parameter:
    - laplacian_pyramid : Laplacian pyramid (list gambar)

    Returns:
    - reconstructed : Gambar yang direkonstruksi
    """
    # Memulai dari level terendah (low-frequency residual)
    current = laplacian_pyramid[-1].copy()

    # Merekonstruksi dari level terendah ke tertinggi
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # Melakukan upsampling level saat ini
        upsampled = cv2.pyrUp(current)

        # Menyesuaikan ukuran agar cocok dengan Laplacian level ini
        h, w = laplacian_pyramid[i].shape[:2]
        upsampled = cv2.resize(upsampled, (w, h))

        # Menambahkan Laplacian (detail) pada level ini
        current = cv2.add(upsampled, laplacian_pyramid[i])

    return current


# ============================================================
# LANGKAH 4: Verifikasi Rekonstruksi → Mendekati Original
# ============================================================
print("\n[LANGKAH 4] Verifikasi rekonstruksi dari Laplacian pyramid...")

# Merekonstruksi gambar kiri dari Laplacian pyramid-nya
reconstructed_left = reconstruct_from_laplacian(lap_left)

# Menghitung error rekonstruksi (harus sangat kecil)
diff = cv2.absdiff(img_left, reconstructed_left)
mean_error = np.mean(diff)
max_error = np.max(diff)

print(f"  Mean reconstruction error : {mean_error:.6f}")
print(f"  Max reconstruction error  : {max_error}")
print(f"  Error rendah = rekonstruksi berhasil (≈ lossless)")

# Menyimpan perbandingan
cv2.imwrite(os.path.join(OUTPUT_DIR, "14_original_left.jpg"), img_left)
cv2.imwrite(os.path.join(OUTPUT_DIR, "14_reconstructed_left.jpg"), reconstructed_left)
cv2.imwrite(os.path.join(OUTPUT_DIR, "14_reconstruction_diff.jpg"), diff * 10)
print("  [OK] Gambar original, rekonstruksi, dan selisih disimpan.")


# ============================================================
# LANGKAH 5: Visualisasi Setiap Level Gaussian Pyramid
# ============================================================
print("\n[LANGKAH 5] Memvisualisasikan setiap level Gaussian pyramid...")

# Membuat figure untuk menampilkan semua level Gaussian
fig_gauss, axes_gauss = plt.subplots(1, LEVELS, figsize=(20, 4))

for i in range(LEVELS):
    # Menampilkan setiap level (konversi BGR ke RGB)
    axes_gauss[i].imshow(cv2.cvtColor(gauss_left[i], cv2.COLOR_BGR2RGB))
    axes_gauss[i].set_title(f"Level {i}\n{gauss_left[i].shape[1]}x{gauss_left[i].shape[0]}",
                             fontsize=10)
    axes_gauss[i].axis("off")

plt.suptitle("Gaussian Pyramid (Gambar Kiri)\n"
             "Setiap level = 1/2 ukuran level sebelumnya",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "14_grid_gaussian_pyramid.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Visualisasi Gaussian pyramid disimpan.")
plt.close()


# ============================================================
# LANGKAH 6: Visualisasi Setiap Level Laplacian Pyramid
# ============================================================
print("\n[LANGKAH 6] Memvisualisasikan setiap level Laplacian pyramid...")

def normalize_laplacian_for_display(lap_img):
    """
    Menormalisasi gambar Laplacian untuk ditampilkan.
    Laplacian memiliki nilai negatif, sehingga perlu dinormalisasi
    ke rentang 0-255 untuk visualisasi.

    Parameter:
    - lap_img : Gambar Laplacian

    Returns:
    - normalized : Gambar yang dinormalisasi (0-255)
    """
    # Mengkonversi ke float untuk perhitungan
    lap_float = lap_img.astype(np.float64)

    # Menormalisasi ke rentang 0-255
    min_val = lap_float.min()
    max_val = lap_float.max()

    if max_val - min_val > 0:
        normalized = ((lap_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(lap_img, dtype=np.uint8)

    return normalized


# Membuat figure untuk Laplacian pyramid
fig_lap, axes_lap = plt.subplots(1, LEVELS, figsize=(20, 4))

for i in range(LEVELS):
    # Menormalisasi Laplacian untuk display
    lap_display = normalize_laplacian_for_display(lap_left[i])
    axes_lap[i].imshow(cv2.cvtColor(lap_display, cv2.COLOR_BGR2RGB))

    # Label berbeda untuk level terakhir (residual)
    if i == LEVELS - 1:
        axes_lap[i].set_title(f"Level {i} (Residual)\n{lap_left[i].shape[1]}x{lap_left[i].shape[0]}",
                               fontsize=10)
    else:
        axes_lap[i].set_title(f"Level {i} (Detail)\n{lap_left[i].shape[1]}x{lap_left[i].shape[0]}",
                               fontsize=10)
    axes_lap[i].axis("off")

plt.suptitle("Laplacian Pyramid (Gambar Kiri)\n"
             "Level 0-3: detail frekuensi tinggi → rendah, Level 4: residual",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "14_grid_laplacian_pyramid.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Visualisasi Laplacian pyramid disimpan.")
plt.close()


# ============================================================
# LANGKAH 7: Blending 2 Gambar Menggunakan Laplacian Pyramid
# ============================================================
print("\n[LANGKAH 7] Melakukan blending 2 gambar dengan Laplacian pyramid...")

def laplacian_blend(img_a, img_b, mask, levels):
    """
    Melakukan multi-band blending menggunakan Laplacian pyramid.

    Algoritma:
    1. Build Gaussian pyramid untuk img_a, img_b, dan mask
    2. Build Laplacian pyramid untuk img_a dan img_b
    3. Blend setiap level Laplacian: L_blend = mask*L_a + (1-mask)*L_b
    4. Rekonstruksi dari blended Laplacian pyramid

    Parameter:
    - img_a  : Gambar pertama (BGR)
    - img_b  : Gambar kedua (BGR)
    - mask   : Mask blending (0=img_b, 255=img_a), grayscale
    - levels : Jumlah level pyramid

    Returns:
    - blended : Gambar hasil blending
    """
    # Mengkonversi mask ke float32 range [0, 1] dengan 3 channel
    mask_float = mask.astype(np.float32) / 255.0
    if len(mask_float.shape) == 2:
        mask_float = cv2.merge([mask_float, mask_float, mask_float])

    # Mengkonversi gambar ke float32
    img_a_f = img_a.astype(np.float32)
    img_b_f = img_b.astype(np.float32)

    # Step 1: Build Gaussian pyramid untuk kedua gambar dan mask
    gauss_a = build_gaussian_pyramid(img_a_f, levels)
    gauss_b = build_gaussian_pyramid(img_b_f, levels)
    gauss_mask = build_gaussian_pyramid(mask_float, levels)

    # Step 2: Build Laplacian pyramid untuk kedua gambar
    lap_a = build_laplacian_pyramid(gauss_a)
    lap_b = build_laplacian_pyramid(gauss_b)

    # Step 3: Blend setiap level menggunakan mask pyramid
    lap_blended = []
    for i in range(levels):
        # Menyesuaikan ukuran mask agar cocok dengan level ini
        h, w = lap_a[i].shape[:2]
        mask_level = cv2.resize(gauss_mask[i], (w, h))

        if len(mask_level.shape) == 2:
            mask_level = cv2.merge([mask_level, mask_level, mask_level])

        # Blending per level: L_blend = mask * L_a + (1 - mask) * L_b
        blended_level = mask_level * lap_a[i] + (1 - mask_level) * lap_b[i]
        lap_blended.append(blended_level.astype(np.float32))

    # Step 4: Rekonstruksi dari blended Laplacian pyramid
    blended = reconstruct_from_laplacian(lap_blended)

    # Mengkonversi kembali ke uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


# Membuat mask untuk blending (setengah kiri = gambar A, setengah kanan = gambar B)
h_blend, w_blend = img_left.shape[:2]
mask_half = np.zeros((h_blend, w_blend), dtype=np.uint8)
mask_half[:, :w_blend // 2] = 255  # Setengah kiri putih = gambar A

# Melakukan Laplacian pyramid blending
waktu_start = time.time()
blended_lap = laplacian_blend(img_left, img_right, mask_half, LEVELS)
waktu_lap = time.time() - waktu_start

# Membuat direct cut (tanpa blending) untuk perbandingan
direct_cut = img_left.copy()
direct_cut[:, w_blend // 2:] = img_right[:, w_blend // 2:]

# Membuat linear blend (alpha blending) untuk perbandingan
alpha_blend = cv2.addWeighted(img_left, 0.5, img_right, 0.5, 0)

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "14_blend_direct_cut.jpg"), direct_cut)
cv2.imwrite(os.path.join(OUTPUT_DIR, "14_blend_alpha.jpg"), alpha_blend)
cv2.imwrite(os.path.join(OUTPUT_DIR, "14_blend_laplacian.jpg"), blended_lap)
print(f"  Direct cut    : selesai")
print(f"  Alpha blending: selesai")
print(f"  Laplacian     : selesai ({waktu_lap*1000:.1f} ms)")
print("  [OK] Ketiga metode blending disimpan.")


# ============================================================
# LANGKAH 8: Membandingkan Jumlah Level Pyramid
# ============================================================
print("\n[LANGKAH 8] Membandingkan kualitas blending dengan jumlah level berbeda...")

# Menguji blending dengan 2, 3, 4, 5, 6 level
level_options = [2, 3, 4, 5, 6]
level_results = {}

for n_level in level_options:
    try:
        # Melakukan blending dengan jumlah level tertentu
        waktu_start = time.time()
        blended_n = laplacian_blend(img_left, img_right, mask_half, n_level)
        waktu_n = time.time() - waktu_start

        # Menghitung metrik: perbedaan di area seam (kolom tengah ±10px)
        seam_region = blended_n[:, w_blend // 2 - 10:w_blend // 2 + 10]
        seam_std = np.std(seam_region.astype(np.float64))

        # Menyimpan hasil
        level_results[n_level] = {
            "gambar": blended_n,
            "waktu": waktu_n * 1000,
            "seam_std": seam_std
        }

        print(f"  Level {n_level}: waktu={waktu_n*1000:.1f}ms, seam_std={seam_std:.2f}")

        # Menyimpan setiap hasil
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"14_blend_level_{n_level}.jpg"), blended_n)

    except Exception as e:
        print(f"  Level {n_level}: Error - {e}")

print("  [OK] Perbandingan level selesai.")


# ============================================================
# LANGKAH 9: Aplikasi pada Panorama Stitching
# ============================================================
print("\n[LANGKAH 9] Menerapkan Laplacian blending pada panorama stitching...")

# Memuat gambar panorama outdoor
img_out1 = cv2.imread(os.path.join(IMAGE_DIR, "panorama_outdoor_1.jpg"))
img_out2 = cv2.imread(os.path.join(IMAGE_DIR, "panorama_outdoor_2.jpg"))

if img_out1 is not None and img_out2 is not None:
    # Menghitung homography antara gambar 1 dan gambar 2
    gray_o1 = cv2.cvtColor(img_out1, cv2.COLOR_BGR2GRAY)
    gray_o2 = cv2.cvtColor(img_out2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray_o1, None)
    kp2, desc2 = sift.detectAndCompute(gray_o2, None)

    # FLANN matching
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50)
    )
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) >= 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Menghitung ukuran canvas
        h1, w1 = img_out1.shape[:2]
        h2, w2 = img_out2.shape[:2]
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners1_t = cv2.perspectiveTransform(corners1, H)
        all_c = np.concatenate([corners1_t, corners2], axis=0)
        x_min = int(np.floor(all_c[:, :, 0].min()))
        y_min = int(np.floor(all_c[:, :, 1].min()))
        x_max = int(np.ceil(all_c[:, :, 0].max()))
        y_max = int(np.ceil(all_c[:, :, 1].max()))
        x_min, y_min = min(x_min, 0), min(y_min, 0)

        canvas_w = x_max - x_min
        canvas_h = y_max - y_min
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

        # Warping gambar 1
        warped1 = cv2.warpPerspective(img_out1, T @ H, (canvas_w, canvas_h))

        # Menempatkan gambar 2 pada canvas
        warped2 = np.zeros_like(warped1)
        ox, oy = -x_min, -y_min
        ye = min(oy + h2, canvas_h)
        xe = min(ox + w2, canvas_w)
        warped2[oy:ye, ox:xe] = img_out2[:ye - oy, :xe - ox]

        # Menyesuaikan ukuran untuk pyramid (kelipatan 2^LEVELS)
        ch_adj = (canvas_h // divisor) * divisor
        cw_adj = (canvas_w // divisor) * divisor
        warped1_adj = cv2.resize(warped1, (cw_adj, ch_adj))
        warped2_adj = cv2.resize(warped2, (cw_adj, ch_adj))

        # Membuat mask: area warped1 valid = putih
        gray_w1 = cv2.cvtColor(warped1_adj, cv2.COLOR_BGR2GRAY)
        gray_w2 = cv2.cvtColor(warped2_adj, cv2.COLOR_BGR2GRAY)
        mask_w1 = (gray_w1 > 5).astype(np.uint8) * 255
        mask_w2 = (gray_w2 > 5).astype(np.uint8) * 255

        # Mask: hanya di area overlap, gunakan gradien; di area non-overlap, 0 atau 255
        overlap_mask = mask_w1 & mask_w2

        # Membuat mask gradien untuk area overlap
        pano_mask = mask_w1.copy()  # Gambar 1 di area non-overlap
        # Di area overlap, gunakan gradien dari kiri ke kanan
        ys_ov, xs_ov = np.where(overlap_mask > 0)
        if len(xs_ov) > 0:
            x_min_ov = xs_ov.min()
            x_max_ov = xs_ov.max()
            if x_max_ov > x_min_ov:
                for x in range(x_min_ov, x_max_ov + 1):
                    alpha = (x - x_min_ov) / (x_max_ov - x_min_ov)
                    col_mask = overlap_mask[:, x] > 0
                    pano_mask[col_mask, x] = int((1 - alpha) * 255)

        # Menerapkan Laplacian blending pada panorama
        pano_blend = laplacian_blend(warped1_adj, warped2_adj, pano_mask, LEVELS)

        # Membuat direct stitch untuk perbandingan
        pano_direct = warped1_adj.copy()
        pano_direct[gray_w2 > 5] = warped2_adj[gray_w2 > 5]

        # Menyimpan hasil
        cv2.imwrite(os.path.join(OUTPUT_DIR, "14_pano_direct_stitch.jpg"), pano_direct)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "14_pano_laplacian_blend.jpg"), pano_blend)
        print(f"  Panorama direct stitch disimpan: {pano_direct.shape[1]}x{pano_direct.shape[0]}")
        print(f"  Panorama Laplacian blend disimpan: {pano_blend.shape[1]}x{pano_blend.shape[0]}")
    else:
        print("  [WARN] Tidak cukup matches untuk panorama stitching.")
        pano_direct = None
        pano_blend = None
else:
    print("  [WARN] Gambar panorama outdoor tidak ditemukan.")
    pano_direct = None
    pano_blend = None


# ============================================================
# LANGKAH 10: Visualisasi Blended Pyramid Per Level
# ============================================================
print("\n[LANGKAH 10] Memvisualisasikan blended pyramid per level...")

# Membangun pyramid untuk visualisasi blending step-by-step
mask_float_vis = mask_half.astype(np.float32) / 255.0
mask_float_vis = cv2.merge([mask_float_vis, mask_float_vis, mask_float_vis])

gauss_a_vis = build_gaussian_pyramid(img_left.astype(np.float32), LEVELS)
gauss_b_vis = build_gaussian_pyramid(img_right.astype(np.float32), LEVELS)
gauss_mask_vis = build_gaussian_pyramid(mask_float_vis, LEVELS)

lap_a_vis = build_laplacian_pyramid(gauss_a_vis)
lap_b_vis = build_laplacian_pyramid(gauss_b_vis)

# Blend per level dan simpan untuk visualisasi
lap_blended_vis = []
for i in range(LEVELS):
    h_lev, w_lev = lap_a_vis[i].shape[:2]
    mask_lev = cv2.resize(gauss_mask_vis[i], (w_lev, h_lev))
    if len(mask_lev.shape) == 2:
        mask_lev = cv2.merge([mask_lev, mask_lev, mask_lev])
    blended_lev = mask_lev * lap_a_vis[i] + (1 - mask_lev) * lap_b_vis[i]
    lap_blended_vis.append(blended_lev.astype(np.float32))

# Membuat visualisasi per level
fig_blend_lev, axes_blend_lev = plt.subplots(3, LEVELS, figsize=(20, 12))

for i in range(LEVELS):
    # Baris 1: Laplacian gambar A
    disp_a = normalize_laplacian_for_display(
        np.clip(lap_a_vis[i], 0, 255).astype(np.uint8)
    )
    axes_blend_lev[0, i].imshow(cv2.cvtColor(disp_a, cv2.COLOR_BGR2RGB))
    axes_blend_lev[0, i].set_title(f"Lap A (Lev {i})", fontsize=9)
    axes_blend_lev[0, i].axis("off")

    # Baris 2: Laplacian gambar B
    disp_b = normalize_laplacian_for_display(
        np.clip(lap_b_vis[i], 0, 255).astype(np.uint8)
    )
    axes_blend_lev[1, i].imshow(cv2.cvtColor(disp_b, cv2.COLOR_BGR2RGB))
    axes_blend_lev[1, i].set_title(f"Lap B (Lev {i})", fontsize=9)
    axes_blend_lev[1, i].axis("off")

    # Baris 3: Blended Laplacian
    disp_bl = normalize_laplacian_for_display(
        np.clip(lap_blended_vis[i], 0, 255).astype(np.uint8)
    )
    axes_blend_lev[2, i].imshow(cv2.cvtColor(disp_bl, cv2.COLOR_BGR2RGB))
    axes_blend_lev[2, i].set_title(f"Blended (Lev {i})", fontsize=9)
    axes_blend_lev[2, i].axis("off")

plt.suptitle("Laplacian Pyramid Blending - Per Level\n"
             "Baris 1: Gambar A | Baris 2: Gambar B | Baris 3: Blended",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "14_grid_blended_per_level.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Visualisasi blended per level disimpan.")
plt.close()


# ============================================================
# LANGKAH 11: Grid Perbandingan Komprehensif
# ============================================================
print("\n[LANGKAH 11] Membuat grid perbandingan komprehensif...")

# --- Grid 1: Perbandingan metode blending ---
fig1, axes1 = plt.subplots(1, 4, figsize=(22, 5))

axes1[0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
axes1[0].set_title("Gambar Kiri (A)", fontsize=11)
axes1[0].axis("off")

axes1[1].imshow(cv2.cvtColor(direct_cut, cv2.COLOR_BGR2RGB))
axes1[1].set_title("Direct Cut\n(seam terlihat jelas)", fontsize=11)
axes1[1].axis("off")

axes1[2].imshow(cv2.cvtColor(alpha_blend, cv2.COLOR_BGR2RGB))
axes1[2].set_title("Alpha Blending\n(ghosting di mana-mana)", fontsize=11)
axes1[2].axis("off")

axes1[3].imshow(cv2.cvtColor(blended_lap, cv2.COLOR_BGR2RGB))
axes1[3].set_title(f"Laplacian Blending\n({LEVELS} levels, {waktu_lap*1000:.0f}ms)", fontsize=11)
axes1[3].axis("off")

plt.suptitle("Percobaan 14: Perbandingan Metode Blending",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "14_grid_perbandingan_blending.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid perbandingan blending disimpan.")
plt.close()

# --- Grid 2: Perbandingan jumlah level ---
n_valid = len(level_results)
if n_valid > 0:
    fig2, axes2 = plt.subplots(1, n_valid + 1, figsize=(4 * (n_valid + 1), 4))

    # Direct cut sebagai baseline
    axes2[0].imshow(cv2.cvtColor(direct_cut, cv2.COLOR_BGR2RGB))
    axes2[0].set_title("Direct Cut\n(0 levels)", fontsize=10)
    axes2[0].axis("off")

    for idx, (n_lev, data) in enumerate(level_results.items()):
        axes2[idx + 1].imshow(cv2.cvtColor(data["gambar"], cv2.COLOR_BGR2RGB))
        axes2[idx + 1].set_title(f"{n_lev} Levels\n({data['waktu']:.0f}ms, "
                                  f"std={data['seam_std']:.1f})", fontsize=10)
        axes2[idx + 1].axis("off")

    plt.suptitle("Pengaruh Jumlah Level Pyramid pada Kualitas Blending",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "14_grid_level_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Grid perbandingan level disimpan.")
    plt.close()

# --- Grid 3: Panorama stitching (direct vs Laplacian) ---
if pano_direct is not None and pano_blend is not None:
    fig3, axes3 = plt.subplots(2, 1, figsize=(18, 8))

    axes3[0].imshow(cv2.cvtColor(pano_direct, cv2.COLOR_BGR2RGB))
    axes3[0].set_title("Panorama - Direct Stitch (tanpa blending)", fontsize=12)
    axes3[0].axis("off")

    axes3[1].imshow(cv2.cvtColor(pano_blend, cv2.COLOR_BGR2RGB))
    axes3[1].set_title("Panorama - Laplacian Pyramid Blending", fontsize=12)
    axes3[1].axis("off")

    plt.suptitle("Percobaan 14: Laplacian Blending pada Panorama Stitching",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "14_grid_panorama_blending.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Grid panorama blending disimpan.")
    plt.close()

# --- Grid 4: Grafik kinerja level ---
if n_valid > 0:
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

    levels_list = list(level_results.keys())
    waktu_list = [level_results[l]["waktu"] for l in levels_list]
    std_list = [level_results[l]["seam_std"] for l in levels_list]

    # Grafik waktu vs jumlah level
    ax4a.plot(levels_list, waktu_list, 'bo-', linewidth=2, markersize=8)
    ax4a.set_xlabel("Jumlah Level Pyramid")
    ax4a.set_ylabel("Waktu (ms)")
    ax4a.set_title("Waktu Eksekusi vs Jumlah Level")
    ax4a.grid(True, alpha=0.3)

    # Grafik seam quality vs jumlah level
    ax4b.plot(levels_list, std_list, 'rs-', linewidth=2, markersize=8)
    ax4b.set_xlabel("Jumlah Level Pyramid")
    ax4b.set_ylabel("Seam Std Dev (lebih rendah = lebih halus)")
    ax4b.set_title("Kualitas Seam vs Jumlah Level")
    ax4b.grid(True, alpha=0.3)

    plt.suptitle("Percobaan 14: Analisis Kinerja Laplacian Blending",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "14_grid_kinerja_level.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Grid kinerja level disimpan.")
    plt.close()


# ============================================================
# LANGKAH 12: Ringkasan dan Kesimpulan
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 14: LAPLACIAN PYRAMID BLENDING")
print("=" * 65)

# Tabel perbandingan level
print("\n  Perbandingan Jumlah Level Pyramid:")
print(f"  {'Level':>6} | {'Waktu (ms)':>10} | {'Seam Std':>10}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}")
for n_lev, data in level_results.items():
    print(f"  {n_lev:>6} | {data['waktu']:>8.1f}ms | {data['seam_std']:>10.2f}")

# Rekonstruksi error
print(f"\n  Verifikasi Rekonstruksi:")
print(f"    Mean error  : {mean_error:.6f}")
print(f"    Max error   : {max_error}")
print(f"    Rekonstruksi: {'Berhasil (lossless)' if max_error < 2 else 'Ada sedikit error'}")

# Daftar file output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("14_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

# Kesimpulan
print("\n  Kesimpulan:")
print("    - Laplacian pyramid blending menghasilkan transisi paling halus")
print("    - Semakin banyak level → blending lebih halus, tapi lebih lambat")
print("    - 4-5 level biasanya cukup untuk hasil yang baik")
print("    - Direct cut menghasilkan seam yang terlihat jelas")
print("    - Alpha blending menyebabkan ghosting di area non-overlap")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.pyrDown()       → Downsampling (Gaussian pyramid)")
print("    cv2.pyrUp()         → Upsampling (rekonstruksi)")
print("    cv2.subtract()      → Menghitung Laplacian level")
print("    cv2.add()           → Rekonstruksi dari Laplacian")
print("    cv2.GaussianBlur()  → Smoothing mask pyramid")
print("    cv2.resize()        → Penyesuaian ukuran antar level")
print("=" * 65)
