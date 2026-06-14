"""
==========================================================================
PERCOBAAN 5: CYLINDRICAL PROJECTION
==========================================================================
Program ini mengimplementasikan cylindrical warping untuk mengurangi
distorsi yang terjadi pada panorama wide-angle. Proyeksi silindris
memetakan gambar planar ke permukaan silinder, memungkinkan stitching
yang lebih baik untuk panorama beberapa gambar.

Konsep yang dipelajari:
- Proyeksi silindris dan mengapa dibutuhkan untuk panorama lebar
- Hubungan antara focal length dan distorsi proyeksi
- Penggunaan cv2.remap() untuk custom warping
- Perbandingan stitching planar vs cylindrical
- Estimasi translasi vs homography penuh pada gambar silindris

Fungsi utama yang dipelajari:
- np.arctan() / np.arctan2() : Fungsi arctangent untuk proyeksi silindris
- cv2.remap()                : Melakukan remapping piksel custom
- np.meshgrid()              : Membuat grid koordinat untuk remapping
- cv2.SIFT_create()          : Detektor fitur pada gambar silindris
- cv2.warpPerspective()      : Warping planar untuk perbandingan
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor library NumPy untuk operasi array, matriks, dan trigonometri
import numpy as np

# Mengimpor library os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi dan grid perbandingan
import matplotlib.pyplot as plt

# Mengimpor modul time untuk mengukur waktu eksekusi
import time

# Mengimpor modul math untuk konstanta dan fungsi matematika
import math

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
print("PERCOBAAN 5: CYLINDRICAL PROJECTION")
print("=" * 65)


# ============================================================
# LANGKAH 1: Implementasi Fungsi Cylindrical Warp
# ============================================================
print("\n[LANGKAH 1] Mengimplementasikan fungsi cylindrical warp...")


def cylindrical_warp(img, focal_length):
    """
    Melakukan cylindrical warping pada gambar.

    Proyeksi silindris memetakan koordinat planar (x, y) ke koordinat
    pada permukaan silinder menggunakan rumus:
        x' = f * arctan((x - cx) / f)
        y' = f * (y - cy) / sqrt((x - cx)^2 + f^2)

    Parameter:
    - img          : Gambar input (BGR)
    - focal_length : Focal length kamera (dalam piksel)

    Returns:
    - warped       : Gambar yang sudah di-warp ke proyeksi silindris
    - mask         : Mask area yang valid (terisi)
    """
    # Mendapatkan dimensi gambar
    h, w = img.shape[:2]

    # Menghitung titik pusat gambar (principal point)
    cx = w / 2.0
    cy = h / 2.0

    # Membuat grid koordinat output (destination) menggunakan meshgrid
    # y_coords dan x_coords berisi koordinat setiap piksel output
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Konversi ke float untuk perhitungan trigonometri
    x_coords = x_coords.astype(np.float32)
    y_coords = y_coords.astype(np.float32)

    # Menghitung inverse cylindrical mapping
    # Dari koordinat output (cylindrical) → koordinat input (planar)
    # Ini inverse mapping sehingga untuk setiap piksel output, kita cari
    # dari mana piksel input berasal

    # Koordinat relatif terhadap pusat
    x_centered = x_coords - cx
    y_centered = y_coords - cy

    # Inverse cylindrical projection:
    # Dari koordinat silindris (x', y') kembali ke planar (x, y)
    # x = f * tan(x' / f)
    # y = y' * sqrt(x^2 + f^2) / f
    # Di mana x' dan y' adalah koordinat relatif terhadap pusat

    # Menghitung theta (sudut horizontal pada silinder)
    theta = x_centered / focal_length

    # Menghitung koordinat planar x (sumber piksel)
    x_planar = focal_length * np.tan(theta) + cx

    # Menghitung koordinat planar y (sumber piksel)
    # Faktor koreksi vertikal berdasarkan posisi horizontal
    y_planar = y_centered / np.cos(theta) + cy

    # Membuat mask untuk piksel yang valid (dalam batas gambar asli)
    mask = ((x_planar >= 0) & (x_planar < w - 1) &
            (y_planar >= 0) & (y_planar < h - 1)).astype(np.uint8) * 255

    # Melakukan remapping menggunakan cv2.remap()
    # remap() mengambil nilai piksel dari posisi (map_x, map_y) ke posisi output
    warped = cv2.remap(
        img,                             # Gambar sumber
        x_planar,                        # Peta koordinat x (dari mana ambil piksel)
        y_planar,                        # Peta koordinat y (dari mana ambil piksel)
        cv2.INTER_LINEAR,               # Interpolasi bilinear untuk kualitas baik
        borderMode=cv2.BORDER_CONSTANT, # Isi area di luar dengan konstanta
        borderValue=(0, 0, 0)           # Isi dengan hitam
    )

    return warped, mask


# Menampilkan info implementasi
print("  Fungsi cylindrical_warp() berhasil diimplementasikan.")
print("  Rumus proyeksi silindris:")
print("    x' = f * arctan((x - cx) / f)")
print("    y' = f * (y - cy) / sqrt((x-cx)^2 + f^2)")
print("  Inverse (untuk remap):")
print("    x = f * tan(x'/f) + cx")
print("    y = y'/cos(x'/f) + cy")

# ============================================================
# LANGKAH 2: Memuat Gambar Panorama Outdoor
# ============================================================
print("\n[LANGKAH 2] Memuat gambar panorama outdoor...")

# Memuat 3 gambar panorama outdoor
outdoor_images = []
for i in range(1, 4):
    path = os.path.join(IMAGE_DIR, f"panorama_outdoor_{i}.jpg")
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] panorama_outdoor_{i}.jpg tidak ditemukan!")
        exit()
    outdoor_images.append(img)
    print(f"  panorama_outdoor_{i}.jpg: {img.shape[1]}x{img.shape[0]}")

# ============================================================
# LANGKAH 3: Efek Focal Length pada Cylindrical Warp
# ============================================================
print("\n[LANGKAH 3] Membandingkan efek focal length...")

# Daftar focal length yang akan diuji
focal_lengths = [200, 400, 600, 800, 1000]

# Menggunakan gambar pertama untuk demonstrasi
img_demo = outdoor_images[0]

# Dictionary untuk menyimpan hasil setiap focal length
focal_results = {}

for f_len in focal_lengths:
    # Melakukan cylindrical warp dengan focal length tertentu
    warped, mask = cylindrical_warp(img_demo, f_len)

    # Menyimpan hasil
    focal_results[f_len] = {
        'warped': warped,
        'mask': mask
    }

    # Menghitung area yang valid (terisi)
    valid_area = np.sum(mask > 0)
    total_area = mask.shape[0] * mask.shape[1]
    fill_ratio = valid_area / total_area * 100

    # Menyimpan gambar hasil
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"05_focal_{f_len}.jpg"), warped)
    print(f"  f={f_len}: Area terisi {fill_ratio:.1f}%")

print("  [OK] Semua variasi focal length disimpan.")

# ============================================================
# LANGKAH 4: Visualisasi Grid Test dalam Cylindrical Coords
# ============================================================
print("\n[LANGKAH 4] Memvisualisasikan grid test dalam proyeksi silindris...")

# Membaca gambar grid test
grid_test = cv2.imread(os.path.join(IMAGE_DIR, "grid_test.jpg"))
if grid_test is not None:
    # Menerapkan cylindrical warp pada gambar grid
    for f_len in [300, 500, 800]:
        grid_warped, grid_mask = cylindrical_warp(grid_test, f_len)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"05_grid_cylindrical_f{f_len}.jpg"), grid_warped)

    print("  [OK] Grid test dalam cylindrical disimpan.")
else:
    print("  [WARNING] grid_test.jpg tidak ditemukan.")

# ============================================================
# LANGKAH 5: Cylindrical Warping pada Semua Gambar
# ============================================================
print("\n[LANGKAH 5] Melakukan cylindrical warp pada semua gambar...")

# Memilih focal length optimal (berdasarkan eksperimen)
# Focal length yang baik biasanya sekitar lebar gambar
optimal_focal = outdoor_images[0].shape[1]  # Menggunakan lebar gambar sebagai estimasi
print(f"  Focal length optimal (estimasi): {optimal_focal}")

# Melakukan cylindrical warp pada semua gambar
cylindrical_images = []
cylindrical_masks = []

for i, img in enumerate(outdoor_images):
    # Menerapkan cylindrical warp
    warped, mask = cylindrical_warp(img, optimal_focal)
    cylindrical_images.append(warped)
    cylindrical_masks.append(mask)

    # Menyimpan gambar silindris
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"05_cylindrical_{i + 1}.jpg"), warped)
    print(f"  Gambar {i + 1}: Cylindrical warp selesai")

print("  [OK] Semua gambar telah di-warp ke cylindrical.")

# ============================================================
# LANGKAH 6: Mencocokkan Fitur pada Gambar Cylindrical
# ============================================================
print("\n[LANGKAH 6] Mencocokkan fitur pada gambar cylindrical...")

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# Menghitung homography antara pasangan gambar cylindrical bersebelahan
cyl_homographies = []
cyl_translations = []  # Untuk estimasi translasi murni

for i in range(len(cylindrical_images) - 1):
    # Mengkonversi ke grayscale
    gray1 = cv2.cvtColor(cylindrical_images[i], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cylindrical_images[i + 1], cv2.COLOR_BGR2GRAY)

    # Mendeteksi fitur SIFT
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    print(f"  Pasangan ({i + 1},{i + 2}): {len(kp1)} vs {len(kp2)} keypoints")

    # Mencocokkan fitur menggunakan FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Melakukan knnMatch
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Menerapkan ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"    Good matches: {len(good)}")

    # Mengekstrak titik korespondensi
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Estimasi homography penuh
    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    cyl_homographies.append(H)

    # Estimasi translasi murni (rata-rata displacement inlier)
    # Pada gambar cylindrical, transformasi idealnya hanya translasi
    if mask_h is not None:
        inlier_src = src_pts[mask_h.ravel() == 1]
        inlier_dst = dst_pts[mask_h.ravel() == 1]

        # Menghitung rata-rata translasi dari inlier
        if len(inlier_src) > 0:
            tx = np.mean(inlier_dst[:, 0, 0] - inlier_src[:, 0, 0])
            ty = np.mean(inlier_dst[:, 0, 1] - inlier_src[:, 0, 1])
        else:
            tx, ty = 0, 0
    else:
        tx, ty = 0, 0

    cyl_translations.append((tx, ty))
    print(f"    Translasi estimasi: tx={tx:.1f}, ty={ty:.1f}")
    print(f"    Inliers: {mask_h.ravel().sum() if mask_h is not None else 0}")

# ============================================================
# LANGKAH 7: Stitching Cylindrical dengan Translasi
# ============================================================
print("\n[LANGKAH 7] Melakukan stitching cylindrical menggunakan translasi...")

# Menghitung total translasi untuk setiap gambar relatif ke gambar pertama
cumulative_tx = [0]  # Gambar pertama: offset 0
cumulative_ty = [0]

for tx, ty in cyl_translations:
    # Akumulasi translasi (negatif karena mapping dari dst ke src)
    cumulative_tx.append(cumulative_tx[-1] - tx)
    cumulative_ty.append(cumulative_ty[-1] - ty)

# Menampilkan translasi kumulatif
for i in range(len(cylindrical_images)):
    print(f"  Gambar {i + 1}: offset = ({cumulative_tx[i]:.1f}, {cumulative_ty[i]:.1f})")

# Menentukan ukuran canvas
h_img, w_img = cylindrical_images[0].shape[:2]

# Menghitung batas canvas berdasarkan translasi
x_offsets = cumulative_tx
y_offsets = cumulative_ty

x_min_c = int(min(x_offsets))
y_min_c = int(min(y_offsets))
x_max_c = int(max(x_offsets)) + w_img
y_max_c = int(max(y_offsets)) + h_img

canvas_cyl_w = x_max_c - x_min_c
canvas_cyl_h = y_max_c - y_min_c

# Membatasi ukuran canvas
MAX_CANVAS = 6000
if canvas_cyl_w > MAX_CANVAS:
    canvas_cyl_w = MAX_CANVAS
if canvas_cyl_h > MAX_CANVAS:
    canvas_cyl_h = MAX_CANVAS

print(f"  Ukuran canvas cylindrical: {canvas_cyl_w} x {canvas_cyl_h}")

# Membuat canvas dan counter untuk blending
canvas_cyl = np.zeros((canvas_cyl_h, canvas_cyl_w, 3), dtype=np.float64)
count_cyl = np.zeros((canvas_cyl_h, canvas_cyl_w), dtype=np.float32)

# Menempatkan setiap gambar pada canvas
for i, img_c in enumerate(cylindrical_images):
    # Menghitung offset untuk gambar ini
    ox = int(cumulative_tx[i] - x_min_c)
    oy = int(cumulative_ty[i] - y_min_c)

    # Mendapatkan dimensi gambar
    h_c, w_c = img_c.shape[:2]

    # Memastikan tidak melebihi batas canvas
    y1 = max(0, oy)
    y2 = min(canvas_cyl_h, oy + h_c)
    x1 = max(0, ox)
    x2 = min(canvas_cyl_w, ox + w_c)

    # Source region dari gambar
    sy1 = max(0, -oy)
    sy2 = sy1 + (y2 - y1)
    sx1 = max(0, -ox)
    sx2 = sx1 + (x2 - x1)

    # Mengambil region gambar
    region = img_c[sy1:sy2, sx1:sx2]

    # Membuat mask non-hitam
    mask_region = (cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

    # Menambahkan ke canvas dan counter
    actual_h = min(y2 - y1, region.shape[0])
    actual_w = min(x2 - x1, region.shape[1])

    for c in range(3):
        canvas_cyl[y1:y1 + actual_h, x1:x1 + actual_w, c] += (
            region[:actual_h, :actual_w, c].astype(np.float64) *
            mask_region[:actual_h, :actual_w]
        )
    count_cyl[y1:y1 + actual_h, x1:x1 + actual_w] += mask_region[:actual_h, :actual_w]

# Normalisasi (rata-rata di area overlap)
count_cyl[count_cyl == 0] = 1
result_cyl = np.zeros((canvas_cyl_h, canvas_cyl_w, 3), dtype=np.uint8)
for c in range(3):
    result_cyl[:, :, c] = np.clip(canvas_cyl[:, :, c] / count_cyl, 0, 255).astype(np.uint8)

# Menyimpan hasil stitching cylindrical
cv2.imwrite(os.path.join(OUTPUT_DIR, "05_stitching_cylindrical.jpg"), result_cyl)
print("  [OK] Stitching cylindrical disimpan.")


# ============================================================
# LANGKAH 8: Stitching Cylindrical dengan Feather Blending
# ============================================================
print("\n[LANGKAH 8] Menerapkan feather blending pada stitching cylindrical...")


def feather_blend_images(images_list, offsets_x, offsets_y, canvas_shape):
    """
    Melakukan feather blending pada kumpulan gambar yang sudah diketahui offset-nya.
    Menggunakan distance transform untuk membuat alpha gradient.

    Parameter:
    - images_list : List gambar (sudah di-warp cylindrical)
    - offsets_x   : List offset horizontal
    - offsets_y   : List offset vertikal
    - canvas_shape: Tuple (height, width) canvas
    """
    ch, cw = canvas_shape

    # Membuat akumulator berbobot
    weighted_sum = np.zeros((ch, cw, 3), dtype=np.float64)
    weight_sum = np.zeros((ch, cw), dtype=np.float64)

    for i, img_c in enumerate(images_list):
        # Menghitung offset
        ox = int(offsets_x[i])
        oy = int(offsets_y[i])
        h_c, w_c = img_c.shape[:2]

        # Menghitung batas region pada canvas
        y1 = max(0, oy)
        y2 = min(ch, oy + h_c)
        x1 = max(0, ox)
        x2 = min(cw, ox + w_c)
        sy1 = max(0, -oy)
        sx1 = max(0, -ox)

        actual_h = y2 - y1
        actual_w = x2 - x1

        # Mengambil region gambar
        region = img_c[sy1:sy1 + actual_h, sx1:sx1 + actual_w]

        # Membuat mask non-hitam
        mask_r = (cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        # Menghitung distance transform sebagai weight (feather effect)
        dist = cv2.distanceTransform(mask_r, cv2.DIST_L2, 5).astype(np.float64)

        # Normalisasi distance ke range [0, 1]
        max_dist = dist.max()
        if max_dist > 0:
            dist = dist / max_dist

        # Menambahkan ke akumulator berbobot
        for c in range(3):
            weighted_sum[y1:y1 + actual_h, x1:x1 + actual_w, c] += (
                region[:actual_h, :actual_w, c].astype(np.float64) * dist[:actual_h, :actual_w]
            )
        weight_sum[y1:y1 + actual_h, x1:x1 + actual_w] += dist[:actual_h, :actual_w]

    # Normalisasi
    weight_sum[weight_sum == 0] = 1
    result = np.zeros((ch, cw, 3), dtype=np.uint8)
    for c in range(3):
        result[:, :, c] = np.clip(weighted_sum[:, :, c] / weight_sum, 0, 255).astype(np.uint8)

    return result


# Menghitung offset relatif
offsets_x_abs = [cumulative_tx[i] - x_min_c for i in range(len(cylindrical_images))]
offsets_y_abs = [cumulative_ty[i] - y_min_c for i in range(len(cylindrical_images))]

# Melakukan feather blending
result_cyl_feather = feather_blend_images(
    cylindrical_images, offsets_x_abs, offsets_y_abs,
    (canvas_cyl_h, canvas_cyl_w)
)

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "05_stitching_cylindrical_feather.jpg"), result_cyl_feather)
print("  [OK] Stitching cylindrical dengan feather blending disimpan.")

# ============================================================
# LANGKAH 9: Stitching Planar untuk Perbandingan
# ============================================================
print("\n[LANGKAH 9] Melakukan stitching planar untuk perbandingan...")

# Melakukan stitching biasa (planar) menggunakan Stitcher API
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
waktu_mulai = time.time()
status_planar, result_planar = stitcher.stitch(outdoor_images)
waktu_planar = time.time() - waktu_mulai

if status_planar == cv2.Stitcher_OK:
    # Crop border hitam dari hasil planar
    gray_p = cv2.cvtColor(result_planar, cv2.COLOR_BGR2GRAY)
    _, thresh_p = cv2.threshold(gray_p, 1, 255, cv2.THRESH_BINARY)
    contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_p:
        largest_p = max(contours_p, key=cv2.contourArea)
        xp, yp, wp, hp = cv2.boundingRect(largest_p)
        result_planar_cropped = result_planar[yp:yp + hp, xp:xp + wp]
    else:
        result_planar_cropped = result_planar

    cv2.imwrite(os.path.join(OUTPUT_DIR, "05_stitching_planar.jpg"), result_planar_cropped)
    print(f"  Stitching planar berhasil: {result_planar_cropped.shape[1]}x{result_planar_cropped.shape[0]}")
    print(f"  Waktu: {waktu_planar:.3f} detik")
else:
    result_planar_cropped = None
    print(f"  [WARNING] Stitching planar gagal.")

# ============================================================
# LANGKAH 10: Perbandingan Distorsi pada Tepi
# ============================================================
print("\n[LANGKAH 10] Membandingkan distorsi pada tepi panorama...")

# Crop hasil cylindrical
gray_cyl = cv2.cvtColor(result_cyl, cv2.COLOR_BGR2GRAY)
_, thresh_cyl = cv2.threshold(gray_cyl, 1, 255, cv2.THRESH_BINARY)
contours_cyl, _ = cv2.findContours(thresh_cyl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours_cyl:
    largest_cyl = max(contours_cyl, key=cv2.contourArea)
    xc, yc, wc, hc = cv2.boundingRect(largest_cyl)
    result_cyl_cropped = result_cyl[yc:yc + hc, xc:xc + wc]
else:
    result_cyl_cropped = result_cyl

# Analisis distorsi
print("\n  Analisis distorsi tepi:")

# Cylindrical
h_cy, w_cy = result_cyl_cropped.shape[:2]
strip_w = min(80, w_cy // 6)
left_strip_cyl = result_cyl_cropped[:, :strip_w]
right_strip_cyl = result_cyl_cropped[:, w_cy - strip_w:]

mean_left_cyl = np.mean(left_strip_cyl)
mean_right_cyl = np.mean(right_strip_cyl)
black_left_cyl = np.sum(cv2.cvtColor(left_strip_cyl, cv2.COLOR_BGR2GRAY) < 5)
black_right_cyl = np.sum(cv2.cvtColor(right_strip_cyl, cv2.COLOR_BGR2GRAY) < 5)

print(f"  Cylindrical:")
print(f"    Ukuran: {w_cy}x{h_cy}")
print(f"    Strip kiri  - Mean: {mean_left_cyl:.1f}, Black: {black_left_cyl}")
print(f"    Strip kanan - Mean: {mean_right_cyl:.1f}, Black: {black_right_cyl}")

# Planar
if result_planar_cropped is not None:
    h_pl, w_pl = result_planar_cropped.shape[:2]
    strip_w_pl = min(80, w_pl // 6)
    left_strip_pl = result_planar_cropped[:, :strip_w_pl]
    right_strip_pl = result_planar_cropped[:, w_pl - strip_w_pl:]

    mean_left_pl = np.mean(left_strip_pl)
    mean_right_pl = np.mean(right_strip_pl)
    black_left_pl = np.sum(cv2.cvtColor(left_strip_pl, cv2.COLOR_BGR2GRAY) < 5)
    black_right_pl = np.sum(cv2.cvtColor(right_strip_pl, cv2.COLOR_BGR2GRAY) < 5)

    print(f"\n  Planar:")
    print(f"    Ukuran: {w_pl}x{h_pl}")
    print(f"    Strip kiri  - Mean: {mean_left_pl:.1f}, Black: {black_left_pl}")
    print(f"    Strip kanan - Mean: {mean_right_pl:.1f}, Black: {black_right_pl}")

# ============================================================
# LANGKAH 11: Membuat Grid Perbandingan Focal Length
# ============================================================
print("\n[LANGKAH 11] Membuat grid perbandingan focal length...")

# Grid focal length (5 gambar + gambar asli)
n_focal = len(focal_lengths)
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))

# Subplot (0,0): Gambar asli (tanpa warp)
axes1[0, 0].imshow(cv2.cvtColor(img_demo, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title("Gambar Asli (Planar)", fontsize=11)
axes1[0, 0].axis("off")

# Subplot lainnya: hasil cylindrical warp dengan berbagai focal length
for idx, f_len in enumerate(focal_lengths):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    warped_img = focal_results[f_len]['warped']
    axes1[row, col].imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    axes1[row, col].set_title(f"Cylindrical f={f_len}", fontsize=11)
    axes1[row, col].axis("off")

# Judul utama
plt.suptitle("Percobaan 5: Efek Focal Length pada Cylindrical Warp",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid focal length
plt.savefig(os.path.join(OUTPUT_DIR, "05_grid_focal_length.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid focal length disimpan.")
plt.close()

# ============================================================
# LANGKAH 12: Grid Perbandingan Planar vs Cylindrical
# ============================================================
print("\n[LANGKAH 12] Membuat grid perbandingan planar vs cylindrical...")

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

# Subplot (0,0): Input gambar montage
montage = outdoor_images[0].copy()
for im in outdoor_images[1:]:
    h_m = montage.shape[0]
    w_new = int(im.shape[1] * h_m / im.shape[0])
    im_r = cv2.resize(im, (w_new, h_m))
    montage = np.hstack([montage, im_r])
axes2[0, 0].imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title(f"Input: {len(outdoor_images)} Gambar Outdoor", fontsize=11)
axes2[0, 0].axis("off")

# Subplot (0,1): Gambar cylindrical
cyl_montage = cylindrical_images[0].copy()
for im_c in cylindrical_images[1:]:
    h_mc = cyl_montage.shape[0]
    w_new_c = int(im_c.shape[1] * h_mc / im_c.shape[0])
    im_rc = cv2.resize(im_c, (w_new_c, h_mc))
    cyl_montage = np.hstack([cyl_montage, im_rc])
axes2[0, 1].imshow(cv2.cvtColor(cyl_montage, cv2.COLOR_BGR2RGB))
axes2[0, 1].set_title(f"Setelah Cylindrical Warp (f={optimal_focal})", fontsize=11)
axes2[0, 1].axis("off")

# Subplot (1,0): Stitching planar
if result_planar_cropped is not None:
    axes2[1, 0].imshow(cv2.cvtColor(result_planar_cropped, cv2.COLOR_BGR2RGB))
    axes2[1, 0].set_title("Stitching Planar (Stitcher API)", fontsize=11)
else:
    axes2[1, 0].text(0.5, 0.5, "GAGAL", ha='center', va='center', fontsize=16)
    axes2[1, 0].set_title("Stitching Planar (gagal)", fontsize=11)
axes2[1, 0].axis("off")

# Subplot (1,1): Stitching cylindrical dengan feather
axes2[1, 1].imshow(cv2.cvtColor(result_cyl_cropped, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title("Stitching Cylindrical (Translasi + Average)", fontsize=11)
axes2[1, 1].axis("off")

# Judul utama
plt.suptitle("Percobaan 5: Perbandingan Stitching Planar vs Cylindrical",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid perbandingan
plt.savefig(os.path.join(OUTPUT_DIR, "05_grid_planar_vs_cylindrical.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid planar vs cylindrical disimpan.")
plt.close()

# ============================================================
# LANGKAH 13: Grid Test Image Cylindrical
# ============================================================
print("\n[LANGKAH 13] Membuat grid test image dalam cylindrical coords...")

if grid_test is not None:
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

    # Gambar grid asli
    axes3[0, 0].imshow(cv2.cvtColor(grid_test, cv2.COLOR_BGR2RGB))
    axes3[0, 0].set_title("Grid Test Asli", fontsize=12)
    axes3[0, 0].axis("off")

    # Grid cylindrical dengan berbagai focal length
    test_focals = [300, 500, 800]
    for idx, f_test in enumerate(test_focals):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        g_path = os.path.join(OUTPUT_DIR, f"05_grid_cylindrical_f{f_test}.jpg")
        g_img = cv2.imread(g_path)
        if g_img is not None:
            axes3[row, col].imshow(cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))
            axes3[row, col].set_title(f"Cylindrical f={f_test}", fontsize=12)
        axes3[row, col].axis("off")

    plt.suptitle("Percobaan 5: Visualisasi Distorsi Cylindrical pada Grid",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "05_grid_test_cylindrical.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Grid test cylindrical disimpan.")
    plt.close()

# ============================================================
# LANGKAH 14: Ringkasan dan Statistik
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 5: CYLINDRICAL PROJECTION")
print("=" * 65)

# Tabel focal length
print("\n  Tabel Efek Focal Length:")
print(f"  {'Focal Length':>12} | {'Area Terisi %':>14}")
print(f"  {'-'*12}-+-{'-'*14}")
for f_len in focal_lengths:
    mask_f = focal_results[f_len]['mask']
    fill = np.sum(mask_f > 0) / (mask_f.shape[0] * mask_f.shape[1]) * 100
    print(f"  {f_len:>12} | {fill:>13.1f}%")

# Tabel translasi antar pasangan
print(f"\n  Translasi Antar Pasangan (Cylindrical):")
print(f"  {'Pasangan':<12} | {'tx':>8} | {'ty':>8}")
print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}")
for i, (tx, ty) in enumerate(cyl_translations):
    print(f"  ({i + 1},{i + 2}){'':<7} | {tx:>7.1f} | {ty:>7.1f}")

# Tabel perbandingan planar vs cylindrical
print(f"\n  Perbandingan Planar vs Cylindrical:")
print(f"  {'Metode':<25} | {'Ukuran':<15}")
print(f"  {'-'*25}-+-{'-'*15}")
if result_planar_cropped is not None:
    print(f"  {'Planar (Stitcher API)':<25} | {result_planar_cropped.shape[1]}x{result_planar_cropped.shape[0]}")
print(f"  {'Cylindrical + Translasi':<25} | {result_cyl_cropped.shape[1]}x{result_cyl_cropped.shape[0]}")

# Penjelasan konsep
print("\n  Konsep Cylindrical Projection:")
print("  - Memetakan gambar planar ke permukaan silinder")
print("  - Mengurangi distorsi perspektif pada panorama lebar")
print("  - Setelah warp, transformasi antar gambar ≈ translasi murni")
print("  - Cocok untuk panorama horizontal > 90 derajat")
print("  - Focal length menentukan 'kelengkungan' proyeksi")

# Daftar output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("05_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    cylindrical_warp()      → Custom cylindrical projection")
print("    np.arctan() / np.tan()  → Fungsi trigonometri untuk proyeksi")
print("    cv2.remap()             → Remapping piksel (inverse mapping)")
print("    np.meshgrid()           → Grid koordinat untuk remapping")
print("    cv2.distanceTransform() → Menghitung jarak ke tepi (feather)")
print("    cv2.SIFT_create()       → Detektor fitur pada gambar silindris")
print("=" * 65)
