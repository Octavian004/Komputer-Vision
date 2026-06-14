"""
==========================================================================
PERCOBAAN 6: SPHERICAL PROJECTION
==========================================================================
Program ini mengimplementasikan spherical warping untuk panorama full-view
dan membandingkan tiga jenis proyeksi: planar, cylindrical, dan spherical.
Proyeksi spherical memetakan gambar planar ke permukaan bola, cocok untuk
panorama yang mencakup field-of-view sangat lebar (>180 derajat).

Konsep yang dipelajari:
- Proyeksi spherical dan perbedaannya dengan cylindrical
- Rumus mapping spherical menggunakan theta dan phi
- Perbandingan tiga jenis proyeksi: planar, cylindrical, spherical
- Pengaruh focal length pada distorsi proyeksi spherical
- Stitching pada gambar spherical menggunakan translasi
- Visualisasi distorsi grid pada ketiga proyeksi

Fungsi utama yang dipelajari:
- np.arctan2()  : Menghitung sudut theta untuk spherical mapping
- np.arctan()   : Menghitung sudut phi untuk spherical mapping
- cv2.remap()   : Remapping piksel berdasarkan koordinat custom
- np.meshgrid() : Membuat grid koordinat 2D untuk remapping
- np.sqrt()     : Menghitung akar kuadrat untuk jarak radial
- np.tan()      : Fungsi tangen untuk inverse mapping
- np.cos()      : Fungsi cosinus untuk koreksi vertikal
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan computer vision
import cv2

# Mengimpor library NumPy untuk operasi array, matriks, dan trigonometri
import numpy as np

# Mengimpor library os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi dan grid perbandingan
import matplotlib.pyplot as plt

# Mengimpor modul time untuk mengukur waktu eksekusi
import time

# Mengimpor modul math untuk konstanta matematika (pi, dll)
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
print("PERCOBAAN 6: SPHERICAL PROJECTION")
print("=" * 65)


# ============================================================
# LANGKAH 1: Implementasi Fungsi Spherical Warp
# ============================================================
print("\n[LANGKAH 1] Mengimplementasikan fungsi spherical warp...")


def spherical_warp(img, focal_length):
    """
    Melakukan spherical warping pada gambar.

    Proyeksi spherical memetakan koordinat planar (x, y) ke koordinat
    pada permukaan bola menggunakan rumus:
        theta = arctan((x - cx) / f)
        phi   = arctan((y - cy) / sqrt((x - cx)^2 + f^2))
        x'    = f * theta + cx
        y'    = f * phi + cy

    Parameter:
    - img          : Gambar input (BGR)
    - focal_length : Focal length kamera (dalam piksel)

    Returns:
    - warped       : Gambar yang sudah di-warp ke proyeksi spherical
    - mask         : Mask area yang valid (terisi)
    """
    # Mendapatkan dimensi gambar
    h, w = img.shape[:2]

    # Menghitung titik pusat gambar (principal point)
    cx = w / 2.0
    cy = h / 2.0

    # Membuat grid koordinat output menggunakan meshgrid
    # Setiap elemen berisi koordinat (x, y) piksel output
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Konversi ke float32 untuk perhitungan trigonometri presisi
    x_coords = x_coords.astype(np.float32)
    y_coords = y_coords.astype(np.float32)

    # Koordinat relatif terhadap pusat gambar
    x_centered = x_coords - cx
    y_centered = y_coords - cy

    # ================================================================
    # INVERSE SPHERICAL MAPPING
    # Dari koordinat output (spherical) → koordinat input (planar)
    # ================================================================

    # Menghitung theta (sudut horizontal) dari koordinat spherical
    # theta = x_centered / focal_length (koordinat pada bola)
    theta = x_centered / focal_length

    # Menghitung phi (sudut vertikal) dari koordinat spherical
    # phi = y_centered / focal_length (koordinat pada bola)
    phi = y_centered / focal_length

    # Inverse spherical projection:
    # Dari koordinat bola (theta, phi) kembali ke planar (x, y)
    # x_planar = f * tan(theta) + cx
    # y_planar = f * tan(phi) / cos(theta) + cy
    x_planar = focal_length * np.tan(theta) + cx
    y_planar = focal_length * np.tan(phi) / np.cos(theta) + cy

    # Membuat mask untuk piksel yang valid (dalam batas gambar asli)
    mask = ((x_planar >= 0) & (x_planar < w - 1) &
            (y_planar >= 0) & (y_planar < h - 1)).astype(np.uint8) * 255

    # Melakukan remapping menggunakan cv2.remap()
    # remap() mengambil nilai piksel dari posisi (map_x, map_y) ke posisi output
    warped = cv2.remap(
        img,                             # Gambar sumber
        x_planar,                        # Peta koordinat x sumber
        y_planar,                        # Peta koordinat y sumber
        cv2.INTER_LINEAR,               # Interpolasi bilinear untuk kualitas
        borderMode=cv2.BORDER_CONSTANT, # Isi area luar dengan konstanta
        borderValue=(0, 0, 0)           # Warna border hitam
    )

    return warped, mask


# Menampilkan info implementasi spherical warp
print("  Fungsi spherical_warp() berhasil diimplementasikan.")
print("  Rumus proyeksi spherical:")
print("    theta = arctan((x - cx) / f)")
print("    phi   = arctan((y - cy) / sqrt((x-cx)^2 + f^2))")
print("    x' = f * theta + cx")
print("    y' = f * phi + cy")


# ============================================================
# LANGKAH 2: Implementasi Fungsi Cylindrical Warp (untuk Perbandingan)
# ============================================================
print("\n[LANGKAH 2] Mengimplementasikan fungsi cylindrical warp...")


def cylindrical_warp(img, focal_length):
    """
    Melakukan cylindrical warping pada gambar (untuk perbandingan).

    Proyeksi silindris memetakan:
        x' = f * arctan((x - cx) / f)
        y' = f * (y - cy) / sqrt((x - cx)^2 + f^2)

    Parameter:
    - img          : Gambar input (BGR)
    - focal_length : Focal length kamera (dalam piksel)

    Returns:
    - warped       : Gambar hasil warp cylindrical
    - mask         : Mask area valid
    """
    # Mendapatkan dimensi gambar
    h, w = img.shape[:2]

    # Menghitung titik pusat gambar
    cx = w / 2.0
    cy = h / 2.0

    # Membuat grid koordinat output
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Konversi ke float32
    x_coords = x_coords.astype(np.float32)
    y_coords = y_coords.astype(np.float32)

    # Koordinat relatif terhadap pusat
    x_centered = x_coords - cx
    y_centered = y_coords - cy

    # Inverse cylindrical mapping:
    # theta = x'/f → x = f * tan(theta)
    # y = y'/cos(theta) + cy
    theta = x_centered / focal_length

    # Menghitung koordinat planar sumber
    x_planar = focal_length * np.tan(theta) + cx
    y_planar = y_centered / np.cos(theta) + cy

    # Membuat mask valid
    mask = ((x_planar >= 0) & (x_planar < w - 1) &
            (y_planar >= 0) & (y_planar < h - 1)).astype(np.uint8) * 255

    # Melakukan remapping
    warped = cv2.remap(
        img, x_planar, y_planar,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return warped, mask


# Menampilkan info implementasi cylindrical warp
print("  Fungsi cylindrical_warp() berhasil diimplementasikan.")


# ============================================================
# LANGKAH 3: Implementasi Fungsi Planar (Identity) Warp
# ============================================================
print("\n[LANGKAH 3] Mengimplementasikan fungsi planar (identity) warp...")


def planar_warp(img, focal_length=None):
    """
    Planar warp (identity) - tidak ada transformasi, hanya copy.
    Digunakan sebagai baseline perbandingan.

    Parameter:
    - img          : Gambar input (BGR)
    - focal_length : Tidak digunakan (untuk konsistensi API)

    Returns:
    - warped       : Copy gambar asli
    - mask         : Mask penuh (semua valid)
    """
    # Planar tidak melakukan transformasi apapun
    warped = img.copy()

    # Mask penuh karena semua piksel valid
    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

    return warped, mask


# Menampilkan info implementasi planar warp
print("  Fungsi planar_warp() (identity) berhasil diimplementasikan.")


# ============================================================
# LANGKAH 4: Memuat Gambar Grid Test untuk Visualisasi Distorsi
# ============================================================
print("\n[LANGKAH 4] Memuat gambar grid test...")

# Membaca gambar grid test yang memiliki garis lurus jelas
grid_test = cv2.imread(os.path.join(IMAGE_DIR, "grid_test.jpg"))

if grid_test is not None:
    # Menampilkan informasi gambar grid
    print(f"  grid_test.jpg: {grid_test.shape[1]}x{grid_test.shape[0]} piksel")
else:
    # Jika grid test tidak ada, buat gambar grid secara manual
    print("  [WARNING] grid_test.jpg tidak ditemukan, membuat grid manual...")

    # Membuat gambar grid 600x400 piksel
    grid_w, grid_h = 600, 400
    grid_test = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # Menggambar garis vertikal setiap 40 piksel
    for x in range(0, grid_w, 40):
        # Garis tebal setiap 200 piksel, tipis di antara
        thickness = 2 if x % 200 == 0 else 1
        cv2.line(grid_test, (x, 0), (x, grid_h), (100, 100, 100), thickness)

    # Menggambar garis horizontal setiap 40 piksel
    for y in range(0, grid_h, 40):
        thickness = 2 if y % 200 == 0 else 1
        cv2.line(grid_test, (0, y), (grid_w, y), (100, 100, 100), thickness)

    # Menambahkan lingkaran di tengah dan sudut sebagai landmark
    cv2.circle(grid_test, (grid_w // 2, grid_h // 2), 50, (255, 0, 0), 2)
    cv2.circle(grid_test, (100, 100), 30, (0, 255, 0), 2)
    cv2.circle(grid_test, (grid_w - 100, 100), 30, (0, 0, 255), 2)

    # Menambahkan label
    cv2.putText(grid_test, "GRID TEST", (grid_w // 2 - 70, grid_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    print(f"  Grid manual dibuat: {grid_w}x{grid_h} piksel")


# ============================================================
# LANGKAH 5: Visualisasi Distorsi pada Grid Test (3 Proyeksi)
# ============================================================
print("\n[LANGKAH 5] Menerapkan 3 proyeksi pada grid test...")

# Focal length untuk demonstrasi distorsi grid
demo_focal = 400

# Menerapkan planar warp (identity - tidak berubah)
grid_planar, mask_planar = planar_warp(grid_test, demo_focal)
print(f"  Planar (identity): tidak ada distorsi")

# Menerapkan cylindrical warp
grid_cylindrical, mask_cylindrical = cylindrical_warp(grid_test, demo_focal)
print(f"  Cylindrical (f={demo_focal}): barrel distortion horizontal")

# Menerapkan spherical warp
grid_spherical, mask_spherical = spherical_warp(grid_test, demo_focal)
print(f"  Spherical (f={demo_focal}): barrel distortion kedua arah")

# Menyimpan hasil distorsi grid
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_grid_planar.jpg"), grid_planar)
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_grid_cylindrical.jpg"), grid_cylindrical)
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_grid_spherical.jpg"), grid_spherical)
print("  [OK] Hasil distorsi grid disimpan.")


# ============================================================
# LANGKAH 6: Memuat Gambar Panorama Outdoor
# ============================================================
print("\n[LANGKAH 6] Memuat gambar panorama outdoor...")

# Memuat 3 gambar panorama outdoor untuk stitching
outdoor_images = []
for i in range(1, 4):
    # Membaca gambar panorama outdoor
    path = os.path.join(IMAGE_DIR, f"panorama_outdoor_{i}.jpg")
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] panorama_outdoor_{i}.jpg tidak ditemukan!")
        print("  Jalankan download_image.py terlebih dahulu.")
        exit()
    outdoor_images.append(img)
    print(f"  panorama_outdoor_{i}.jpg: {img.shape[1]}x{img.shape[0]} piksel")

# Menampilkan jumlah gambar yang berhasil dimuat
print(f"  Total gambar: {len(outdoor_images)}")


# ============================================================
# LANGKAH 7: Menerapkan Spherical Warp pada Semua Gambar
# ============================================================
print("\n[LANGKAH 7] Melakukan spherical warp pada semua gambar...")

# Estimasi focal length optimal (≈ lebar gambar)
optimal_focal = outdoor_images[0].shape[1]
print(f"  Focal length optimal (estimasi): {optimal_focal}")

# Melakukan spherical warp pada setiap gambar
spherical_images = []
spherical_masks = []

for i, img in enumerate(outdoor_images):
    # Mengukur waktu warping
    t_start = time.time()

    # Menerapkan spherical warp
    warped, mask = spherical_warp(img, optimal_focal)
    t_elapsed = time.time() - t_start

    # Menyimpan hasil
    spherical_images.append(warped)
    spherical_masks.append(mask)

    # Menyimpan gambar spherical individual
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"06_spherical_{i + 1}.jpg"), warped)
    print(f"  Gambar {i + 1}: Spherical warp selesai ({t_elapsed:.3f} detik)")

# Melakukan cylindrical warp pada semua gambar juga untuk perbandingan
cylindrical_images = []

for i, img in enumerate(outdoor_images):
    # Menerapkan cylindrical warp
    warped_cyl, _ = cylindrical_warp(img, optimal_focal)
    cylindrical_images.append(warped_cyl)
    print(f"  Gambar {i + 1}: Cylindrical warp selesai")

print("  [OK] Semua gambar telah di-warp ke spherical dan cylindrical.")


# ============================================================
# LANGKAH 8: Mencocokkan Fitur dan Estimasi Translasi
# ============================================================
print("\n[LANGKAH 8] Mencocokkan fitur pada gambar spherical...")

# Membuat detektor SIFT
sift = cv2.SIFT_create()


def hitung_translasi(images_list, label=""):
    """
    Menghitung translasi antar pasangan gambar bersebelahan.

    Parameter:
    - images_list : List gambar yang sudah di-warp
    - label       : Label untuk logging

    Returns:
    - translations : List tuple (tx, ty) translasi antar pasangan
    """
    translations = []

    for i in range(len(images_list) - 1):
        # Mengkonversi ke grayscale untuk deteksi fitur
        gray1 = cv2.cvtColor(images_list[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images_list[i + 1], cv2.COLOR_BGR2GRAY)

        # Mendeteksi fitur SIFT
        kp1, desc1 = sift.detectAndCompute(gray1, None)
        kp2, desc2 = sift.detectAndCompute(gray2, None)

        # Memeriksa apakah deskriptor valid
        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            print(f"    {label} Pasangan ({i + 1},{i + 2}): Tidak cukup fitur")
            translations.append((0, 0))
            continue

        # Mengonfigurasi FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Melakukan knnMatch
        matches = flann.knnMatch(desc1, desc2, k=2)

        # Menerapkan Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # Memeriksa jumlah kecocokan yang cukup
        if len(good) < 4:
            print(f"    {label} Pasangan ({i + 1},{i + 2}): Terlalu sedikit matches")
            translations.append((0, 0))
            continue

        # Mengekstrak titik korespondensi
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Estimasi homography dengan RANSAC
        H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Menghitung translasi dari inlier
        if mask_h is not None:
            inlier_src = src_pts[mask_h.ravel() == 1]
            inlier_dst = dst_pts[mask_h.ravel() == 1]

            if len(inlier_src) > 0:
                # Rata-rata translasi dari inlier
                tx = np.mean(inlier_dst[:, 0, 0] - inlier_src[:, 0, 0])
                ty = np.mean(inlier_dst[:, 0, 1] - inlier_src[:, 0, 1])
            else:
                tx, ty = 0, 0
        else:
            tx, ty = 0, 0

        translations.append((tx, ty))
        n_inliers = mask_h.ravel().sum() if mask_h is not None else 0
        print(f"    {label} Pasangan ({i + 1},{i + 2}): "
              f"matches={len(good)}, inliers={n_inliers}, "
              f"tx={tx:.1f}, ty={ty:.1f}")

    return translations


# Menghitung translasi untuk gambar spherical
print("  [Spherical]")
sph_translations = hitung_translasi(spherical_images, label="[Sph]")

# Menghitung translasi untuk gambar cylindrical
print("  [Cylindrical]")
cyl_translations = hitung_translasi(cylindrical_images, label="[Cyl]")


# ============================================================
# LANGKAH 9: Stitching Translasi dengan Blending
# ============================================================
print("\n[LANGKAH 9] Melakukan stitching translasi pada ketiga proyeksi...")


def stitch_with_translation(images_list, translations):
    """
    Melakukan stitching gambar menggunakan estimasi translasi.
    Menggunakan distance transform untuk feather blending.

    Parameter:
    - images_list  : List gambar yang akan di-stitch
    - translations : List tuple (tx, ty) translasi antar pasangan

    Returns:
    - result : Gambar panorama hasil stitching
    """
    # Menghitung translasi kumulatif relatif ke gambar pertama
    cum_tx = [0.0]
    cum_ty = [0.0]
    for tx, ty in translations:
        cum_tx.append(cum_tx[-1] - tx)
        cum_ty.append(cum_ty[-1] - ty)

    # Menentukan dimensi canvas
    h_img, w_img = images_list[0].shape[:2]
    x_min = int(min(cum_tx))
    y_min = int(min(cum_ty))
    x_max = int(max(cum_tx)) + w_img
    y_max = int(max(cum_ty)) + h_img

    canvas_w = min(x_max - x_min, 6000)
    canvas_h = min(y_max - y_min, 3000)

    # Membuat akumulator untuk blending berbobot
    weighted_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    for i, img_c in enumerate(images_list):
        # Menghitung offset absolut pada canvas
        ox = int(cum_tx[i] - x_min)
        oy = int(cum_ty[i] - y_min)
        h_c, w_c = img_c.shape[:2]

        # Menghitung batas region pada canvas
        y1 = max(0, oy)
        y2 = min(canvas_h, oy + h_c)
        x1 = max(0, ox)
        x2 = min(canvas_w, ox + w_c)
        sy1 = max(0, -oy)
        sx1 = max(0, -ox)

        actual_h = y2 - y1
        actual_w = x2 - x1

        if actual_h <= 0 or actual_w <= 0:
            continue

        # Mengambil region gambar
        region = img_c[sy1:sy1 + actual_h, sx1:sx1 + actual_w]

        # Membuat mask non-hitam
        mask_r = (cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        # Menghitung distance transform sebagai weight (feather blending)
        dist = cv2.distanceTransform(mask_r, cv2.DIST_L2, 5).astype(np.float64)
        max_dist = dist.max()
        if max_dist > 0:
            dist = dist / max_dist

        # Menambahkan ke akumulator berbobot
        for c in range(3):
            weighted_sum[y1:y1 + actual_h, x1:x1 + actual_w, c] += (
                region[:actual_h, :actual_w, c].astype(np.float64) *
                dist[:actual_h, :actual_w]
            )
        weight_sum[y1:y1 + actual_h, x1:x1 + actual_w] += dist[:actual_h, :actual_w]

    # Normalisasi
    weight_sum[weight_sum == 0] = 1
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for c in range(3):
        result[:, :, c] = np.clip(
            weighted_sum[:, :, c] / weight_sum, 0, 255
        ).astype(np.uint8)

    return result


# Stitching spherical
t_start = time.time()
result_spherical = stitch_with_translation(spherical_images, sph_translations)
t_spherical = time.time() - t_start
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_stitching_spherical.jpg"), result_spherical)
print(f"  Stitching spherical selesai ({t_spherical:.3f} detik)")

# Stitching cylindrical
t_start = time.time()
result_cylindrical = stitch_with_translation(cylindrical_images, cyl_translations)
t_cylindrical = time.time() - t_start
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_stitching_cylindrical.jpg"), result_cylindrical)
print(f"  Stitching cylindrical selesai ({t_cylindrical:.3f} detik)")

# Stitching planar menggunakan Stitcher API
t_start = time.time()
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status_planar, result_planar_api = stitcher.stitch(outdoor_images)
t_planar = time.time() - t_start

if status_planar == cv2.Stitcher_OK:
    # Crop black borders dari hasil planar
    gray_p = cv2.cvtColor(result_planar_api, cv2.COLOR_BGR2GRAY)
    _, thresh_p = cv2.threshold(gray_p, 1, 255, cv2.THRESH_BINARY)
    contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if contours_p:
        largest_p = max(contours_p, key=cv2.contourArea)
        xp, yp, wp, hp = cv2.boundingRect(largest_p)
        result_planar_cropped = result_planar_api[yp:yp + hp, xp:xp + wp]
    else:
        result_planar_cropped = result_planar_api
    cv2.imwrite(os.path.join(OUTPUT_DIR, "06_stitching_planar.jpg"),
                result_planar_cropped)
    print(f"  Stitching planar selesai ({t_planar:.3f} detik)")
else:
    result_planar_cropped = None
    print(f"  [WARNING] Stitching planar gagal (status={status_planar})")


# ============================================================
# LANGKAH 10: Crop dan Analisis Distorsi Garis Lurus
# ============================================================
print("\n[LANGKAH 10] Menganalisis distorsi garis lurus pada setiap proyeksi...")

# Crop border hitam dari hasil spherical dan cylindrical
def crop_black_border(img):
    """Memotong area hitam di tepi gambar."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return img[y:y + h, x:x + w]
    return img


# Crop hasil stitching
result_sph_cropped = crop_black_border(result_spherical)
result_cyl_cropped = crop_black_border(result_cylindrical)

# Menyimpan versi cropped
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_stitching_spherical_cropped.jpg"),
            result_sph_cropped)
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_stitching_cylindrical_cropped.jpg"),
            result_cyl_cropped)

# Mengukur distorsi garis vertikal pada tepi gambar
print("\n  Analisis distorsi tepi panorama:")

# Fungsi untuk mengukur distorsi tepi
def analisis_distorsi_tepi(img, label):
    """
    Menganalisis distorsi pada strip kiri dan kanan gambar.
    Menghitung rasio piksel hitam sebagai indikator distorsi.
    """
    h, w = img.shape[:2]
    strip_w = min(60, w // 8)

    # Mengambil strip kiri dan kanan
    strip_kiri = img[:, :strip_w]
    strip_kanan = img[:, w - strip_w:]

    # Menghitung piksel hitam (area kosong/distorsi)
    hitam_kiri = np.sum(cv2.cvtColor(strip_kiri, cv2.COLOR_BGR2GRAY) < 5)
    hitam_kanan = np.sum(cv2.cvtColor(strip_kanan, cv2.COLOR_BGR2GRAY) < 5)
    total_strip = strip_w * h

    # Menghitung rasio distorsi
    rasio_kiri = hitam_kiri / total_strip * 100
    rasio_kanan = hitam_kanan / total_strip * 100

    print(f"  {label}:")
    print(f"    Ukuran: {w}x{h}")
    print(f"    Distorsi kiri:  {rasio_kiri:.1f}% area hitam")
    print(f"    Distorsi kanan: {rasio_kanan:.1f}% area hitam")

    return rasio_kiri, rasio_kanan


# Menganalisis distorsi untuk setiap proyeksi
dist_sph = analisis_distorsi_tepi(result_sph_cropped, "Spherical")
dist_cyl = analisis_distorsi_tepi(result_cyl_cropped, "Cylindrical")
if result_planar_cropped is not None:
    dist_pln = analisis_distorsi_tepi(result_planar_cropped, "Planar")


# ============================================================
# LANGKAH 11: Efek Focal Length pada Spherical Warp
# ============================================================
print("\n[LANGKAH 11] Membandingkan efek focal length pada spherical warp...")

# Daftar focal length yang akan diuji
focal_lengths = [200, 400, 600, 800, 1000]

# Menggunakan gambar pertama untuk demonstrasi
img_demo = outdoor_images[0]

# Dictionary untuk menyimpan hasil
focal_results = {}

for f_len in focal_lengths:
    # Menerapkan spherical warp
    warped, mask = spherical_warp(img_demo, f_len)
    focal_results[f_len] = {'warped': warped, 'mask': mask}

    # Menghitung area terisi
    valid_area = np.sum(mask > 0)
    total_area = mask.shape[0] * mask.shape[1]
    fill_ratio = valid_area / total_area * 100

    # Menyimpan hasil
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"06_spherical_focal_{f_len}.jpg"), warped)
    print(f"  f={f_len}: Area terisi {fill_ratio:.1f}%")

print("  [OK] Semua variasi focal length disimpan.")


# ============================================================
# LANGKAH 12: Visualisasi Grid Distorsi - Ketiga Proyeksi
# ============================================================
print("\n[LANGKAH 12] Membuat visualisasi grid garis pada ketiga proyeksi...")

# Membuat gambar dengan garis horizontal dan vertikal yang jelas
def buat_grid_garis(width, height, spacing=50):
    """
    Membuat gambar grid dengan garis lurus berwarna untuk
    memvisualisasikan distorsi transformasi.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 240

    # Menggambar garis vertikal (biru)
    for x in range(0, width, spacing):
        thickness = 2 if x % (spacing * 4) == 0 else 1
        cv2.line(img, (x, 0), (x, height), (255, 0, 0), thickness)

    # Menggambar garis horizontal (merah)
    for y in range(0, height, spacing):
        thickness = 2 if y % (spacing * 4) == 0 else 1
        cv2.line(img, (0, y), (width, y), (0, 0, 255), thickness)

    # Menggambar diagonal untuk referensi tambahan
    cv2.line(img, (0, 0), (width, height), (0, 180, 0), 1)
    cv2.line(img, (width, 0), (0, height), (0, 180, 0), 1)

    return img


# Membuat grid sesuai ukuran gambar outdoor
h_demo, w_demo = img_demo.shape[:2]
grid_lines = buat_grid_garis(w_demo, h_demo, spacing=40)

# Menerapkan ketiga proyeksi pada grid garis
grid_lines_planar, _ = planar_warp(grid_lines)
grid_lines_cyl, _ = cylindrical_warp(grid_lines, optimal_focal)
grid_lines_sph, _ = spherical_warp(grid_lines, optimal_focal)

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_grid_lines_planar.jpg"),
            grid_lines_planar)
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_grid_lines_cylindrical.jpg"),
            grid_lines_cyl)
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_grid_lines_spherical.jpg"),
            grid_lines_sph)
print("  [OK] Grid garis untuk ketiga proyeksi disimpan.")


# ============================================================
# LANGKAH 13: Membuat Grid Perbandingan 3 Proyeksi (Stitching)
# ============================================================
print("\n[LANGKAH 13] Membuat grid perbandingan 3 proyeksi stitching...")

# Membuat figure perbandingan stitching
fig1, axes1 = plt.subplots(3, 1, figsize=(16, 14))

# Subplot 0: Planar
if result_planar_cropped is not None:
    axes1[0].imshow(cv2.cvtColor(result_planar_cropped, cv2.COLOR_BGR2RGB))
    axes1[0].set_title(f"Planar (Stitcher API) - "
                       f"{result_planar_cropped.shape[1]}x{result_planar_cropped.shape[0]} - "
                       f"Waktu: {t_planar:.3f}s", fontsize=12)
else:
    axes1[0].text(0.5, 0.5, "GAGAL", ha='center', va='center', fontsize=16)
    axes1[0].set_title("Planar (GAGAL)", fontsize=12)
axes1[0].axis("off")

# Subplot 1: Cylindrical
axes1[1].imshow(cv2.cvtColor(result_cyl_cropped, cv2.COLOR_BGR2RGB))
axes1[1].set_title(f"Cylindrical (f={optimal_focal}) - "
                   f"{result_cyl_cropped.shape[1]}x{result_cyl_cropped.shape[0]} - "
                   f"Waktu: {t_cylindrical:.3f}s", fontsize=12)
axes1[1].axis("off")

# Subplot 2: Spherical
axes1[2].imshow(cv2.cvtColor(result_sph_cropped, cv2.COLOR_BGR2RGB))
axes1[2].set_title(f"Spherical (f={optimal_focal}) - "
                   f"{result_sph_cropped.shape[1]}x{result_sph_cropped.shape[0]} - "
                   f"Waktu: {t_spherical:.3f}s", fontsize=12)
axes1[2].axis("off")

# Judul utama
plt.suptitle("Percobaan 6: Perbandingan 3 Proyeksi Stitching\n"
             "(Planar vs Cylindrical vs Spherical)",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan figure perbandingan
plt.savefig(os.path.join(OUTPUT_DIR, "06_grid_perbandingan_3_proyeksi.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid perbandingan 3 proyeksi disimpan.")
plt.close()


# ============================================================
# LANGKAH 14: Membuat Grid Perbandingan Distorsi Grid
# ============================================================
print("\n[LANGKAH 14] Membuat grid perbandingan distorsi...")

# Membuat figure distorsi grid - 2 baris: grid test / grid garis
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

# Baris 0: Grid test dengan 3 proyeksi
axes2[0, 0].imshow(cv2.cvtColor(grid_planar, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("Grid Test: Planar", fontsize=11)
axes2[0, 0].axis("off")

axes2[0, 1].imshow(cv2.cvtColor(grid_cylindrical, cv2.COLOR_BGR2RGB))
axes2[0, 1].set_title(f"Grid Test: Cylindrical (f={demo_focal})", fontsize=11)
axes2[0, 1].axis("off")

axes2[0, 2].imshow(cv2.cvtColor(grid_spherical, cv2.COLOR_BGR2RGB))
axes2[0, 2].set_title(f"Grid Test: Spherical (f={demo_focal})", fontsize=11)
axes2[0, 2].axis("off")

# Baris 1: Grid garis dengan 3 proyeksi
axes2[1, 0].imshow(cv2.cvtColor(grid_lines_planar, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("Grid Garis: Planar", fontsize=11)
axes2[1, 0].axis("off")

axes2[1, 1].imshow(cv2.cvtColor(grid_lines_cyl, cv2.COLOR_BGR2RGB))
axes2[1, 1].set_title(f"Grid Garis: Cylindrical (f={optimal_focal})", fontsize=11)
axes2[1, 1].axis("off")

axes2[1, 2].imshow(cv2.cvtColor(grid_lines_sph, cv2.COLOR_BGR2RGB))
axes2[1, 2].set_title(f"Grid Garis: Spherical (f={optimal_focal})", fontsize=11)
axes2[1, 2].axis("off")

plt.suptitle("Percobaan 6: Visualisasi Distorsi pada 3 Proyeksi",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan figure distorsi
plt.savefig(os.path.join(OUTPUT_DIR, "06_grid_distorsi_3_proyeksi.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid distorsi 3 proyeksi disimpan.")
plt.close()


# ============================================================
# LANGKAH 15: Membuat Grid Efek Focal Length
# ============================================================
print("\n[LANGKAH 15] Membuat grid efek focal length pada spherical...")

# Grid focal length (6 subplot: asli + 5 focal length)
n_focal = len(focal_lengths)
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))

# Subplot (0,0): Gambar asli
axes3[0, 0].imshow(cv2.cvtColor(img_demo, cv2.COLOR_BGR2RGB))
axes3[0, 0].set_title("Gambar Asli (Planar)", fontsize=11)
axes3[0, 0].axis("off")

# Subplot lainnya: berbagai focal length
for idx, f_len in enumerate(focal_lengths):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    warped_img = focal_results[f_len]['warped']
    mask_f = focal_results[f_len]['mask']
    fill = np.sum(mask_f > 0) / (mask_f.shape[0] * mask_f.shape[1]) * 100
    axes3[row, col].imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    axes3[row, col].set_title(f"Spherical f={f_len} ({fill:.0f}% terisi)",
                               fontsize=11)
    axes3[row, col].axis("off")

plt.suptitle("Percobaan 6: Efek Focal Length pada Spherical Warp",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid focal length
plt.savefig(os.path.join(OUTPUT_DIR, "06_grid_focal_length_spherical.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid focal length disimpan.")
plt.close()


# ============================================================
# LANGKAH 16: Ringkasan dan Statistik
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 6: SPHERICAL PROJECTION")
print("=" * 65)

# Tabel perbandingan proyeksi
print("\n  Tabel Perbandingan 3 Proyeksi:")
print(f"  {'Proyeksi':<15} | {'Ukuran Hasil':<15} | {'Waktu (s)':>10}")
print(f"  {'-' * 15}-+-{'-' * 15}-+-{'-' * 10}")

if result_planar_cropped is not None:
    sz_p = f"{result_planar_cropped.shape[1]}x{result_planar_cropped.shape[0]}"
    print(f"  {'Planar':<15} | {sz_p:<15} | {t_planar:>10.3f}")

sz_c = f"{result_cyl_cropped.shape[1]}x{result_cyl_cropped.shape[0]}"
print(f"  {'Cylindrical':<15} | {sz_c:<15} | {t_cylindrical:>10.3f}")

sz_s = f"{result_sph_cropped.shape[1]}x{result_sph_cropped.shape[0]}"
print(f"  {'Spherical':<15} | {sz_s:<15} | {t_spherical:>10.3f}")

# Tabel focal length
print(f"\n  Tabel Efek Focal Length (Spherical):")
print(f"  {'Focal Length':>12} | {'Area Terisi %':>14}")
print(f"  {'-' * 12}-+-{'-' * 14}")
for f_len in focal_lengths:
    mask_f = focal_results[f_len]['mask']
    fill = np.sum(mask_f > 0) / (mask_f.shape[0] * mask_f.shape[1]) * 100
    print(f"  {f_len:>12} | {fill:>13.1f}%")

# Penjelasan konsep
print("\n  Konsep Spherical vs Cylindrical Projection:")
print("  - Spherical: memetakan ke permukaan bola (2 sumbu kelengkungan)")
print("  - Cylindrical: memetakan ke permukaan silinder (1 sumbu kelengkungan)")
print("  - Spherical lebih baik untuk FoV vertikal + horizontal lebar")
print("  - Cylindrical lebih baik untuk panorama horizontal panjang")
print("  - Planar: tidak ada koreksi, distorsi besar di tepi")
print("  - Focal length kecil → distorsi besar; besar → mendekati planar")

# Daftar output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("06_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    spherical_warp()    → Custom spherical projection")
print("    cylindrical_warp()  → Custom cylindrical projection")
print("    np.arctan2()/tan()  → Fungsi trigonometri untuk proyeksi")
print("    cv2.remap()         → Remapping piksel (inverse mapping)")
print("    np.meshgrid()       → Grid koordinat 2D untuk remapping")
print("    np.sqrt()/cos()     → Koreksi vertikal spherical")
print("=" * 65)
