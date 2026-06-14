"""
==========================================================================
PERCOBAAN 7: BUNDLE ADJUSTMENT (KONSEP)
==========================================================================
Program ini memahami efek bundle adjustment pada kualitas panorama
multi-image. Bundle adjustment mengoptimasi parameter kamera secara
global untuk meminimalkan reprojection error pada seluruh pasangan
gambar, berbeda dengan chain homography yang mengakumulasi error.

Konsep yang dipelajari:
- Chain homography dan masalah akumulasi error (drift)
- Konsep bundle adjustment (optimasi global parameter kamera)
- Reprojection error: ukuran keselarasan fitur setelah transformasi
- Perbandingan kualitas: chain homography vs OpenCV Stitcher (with BA)
- Efek jumlah gambar terhadap akumulasi error
- Analisis alignment pada area seam/overlap

Fungsi utama yang dipelajari:
- cv2.Stitcher_create()     : Membuat Stitcher (termasuk BA internal)
- cv2.findHomography()      : Estimasi homography per pasangan
- cv2.perspectiveTransform(): Transform titik untuk menghitung error
- cv2.warpPerspective()     : Warping menggunakan homography
- np.linalg.norm()          : Menghitung norma vektor (error jarak)
- np.matmul() / operator @  : Perkalian matriks chain homography
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan stitching
import cv2

# Mengimpor library NumPy untuk operasi matriks dan array
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
print("PERCOBAAN 7: BUNDLE ADJUSTMENT (KONSEP)")
print("=" * 65)


# ============================================================
# FUNGSI HELPER: Mencocokkan Fitur dan Estimasi Homography
# ============================================================

def hitung_homography(img_src, img_dst, label=""):
    """
    Menghitung homography dari img_src ke img_dst menggunakan
    pipeline SIFT → FLANN → ratio test → RANSAC.

    Parameter:
    - img_src : Gambar sumber yang akan di-warp
    - img_dst : Gambar tujuan/referensi
    - label   : Label deskriptif untuk logging

    Returns:
    - H        : Matriks homography 3x3 (src → dst)
    - n_inlier : Jumlah inlier RANSAC
    - n_match  : Jumlah good matches
    - src_pts  : Titik sumber inlier
    - dst_pts  : Titik tujuan inlier
    """
    # Mengkonversi kedua gambar ke grayscale untuk deteksi fitur
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    # Membuat detektor SIFT
    sift = cv2.SIFT_create()

    # Mendeteksi keypoints dan menghitung deskriptor
    kp_src, desc_src = sift.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = sift.detectAndCompute(gray_dst, None)

    # Memeriksa apakah deskriptor valid
    if desc_src is None or desc_dst is None or len(desc_src) < 4 or len(desc_dst) < 4:
        if label:
            print(f"    {label}: Tidak cukup fitur untuk matching")
        return np.eye(3, dtype=np.float64), 0, 0, None, None

    # Mengonfigurasi FLANN matcher untuk pencarian cepat
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Melakukan kNN matching (k=2) untuk ratio test
    matches_knn = flann.knnMatch(desc_src, desc_dst, k=2)

    # Menerapkan Lowe's ratio test untuk memfilter kecocokan yang baik
    good = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Memastikan ada cukup kecocokan (minimal 4 untuk homography)
    if len(good) < 4:
        if label:
            print(f"    {label}: {len(good)} matches (terlalu sedikit)")
        return np.eye(3, dtype=np.float64), 0, len(good), None, None

    # Mengekstrak titik korespondensi dari good matches
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Mengestimasi homography menggunakan RANSAC
    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Fallback ke identity jika estimasi gagal
    if H is None:
        if label:
            print(f"    {label}: Homography gagal, gunakan identitas")
        return np.eye(3, dtype=np.float64), 0, len(good), None, None

    # Menghitung jumlah inlier
    n_inlier = int(mask_h.ravel().sum()) if mask_h is not None else 0

    # Mengekstrak inlier points
    inlier_src = src_pts[mask_h.ravel() == 1] if mask_h is not None else None
    inlier_dst = dst_pts[mask_h.ravel() == 1] if mask_h is not None else None

    if label:
        print(f"    {label}: matches={len(good)}, inliers={n_inlier}")

    return H, n_inlier, len(good), inlier_src, inlier_dst


# ============================================================
# LANGKAH 1: Memuat Gambar Panorama Wide (5 gambar)
# ============================================================
print("\n[LANGKAH 1] Memuat 5 gambar panorama wide...")

# Memuat 5 gambar panorama wide untuk demonstrasi multi-image
images_wide = []
for i in range(1, 6):
    # Membaca setiap gambar panorama wide
    path = os.path.join(IMAGE_DIR, f"panorama_wide_{i}.jpg")
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] panorama_wide_{i}.jpg tidak ditemukan!")
        print("  Jalankan download_image.py terlebih dahulu.")
        exit()
    images_wide.append(img)
    print(f"  panorama_wide_{i}.jpg: {img.shape[1]}x{img.shape[0]} piksel")

# Menampilkan total gambar yang dimuat
n_images = len(images_wide)
print(f"  Total gambar: {n_images}")


# ============================================================
# LANGKAH 2: Chain Homography TANPA Bundle Adjustment
# ============================================================
print("\n[LANGKAH 2] Menghitung chain homography TANPA bundle adjustment...")
print("  (Metode A: akumulasi homography berurutan)")

# Gambar tengah (indeks 2) sebagai referensi untuk meminimalkan distorsi
ref_idx = n_images // 2
print(f"  Gambar referensi: indeks {ref_idx + 1} (tengah)")

# Menghitung homography antara setiap pasangan bersebelahan
pairwise_H = []
pairwise_inliers = []
pairwise_matches = []
pairwise_errors = []

for i in range(n_images - 1):
    # Menghitung homography dari gambar i ke gambar i+1
    H, n_inlier, n_match, src_pts, dst_pts = hitung_homography(
        images_wide[i], images_wide[i + 1],
        label=f"H({i + 1}→{i + 2})"
    )
    pairwise_H.append(H)
    pairwise_inliers.append(n_inlier)
    pairwise_matches.append(n_match)

    # Menghitung reprojection error untuk pasangan ini
    if src_pts is not None and dst_pts is not None and len(src_pts) > 0:
        # Mentransformasi titik sumber menggunakan homography
        src_transformed = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H)

        # Menghitung error sebagai jarak Euclidean rata-rata
        errors = np.sqrt(np.sum((src_transformed - dst_pts.reshape(-1, 1, 2)) ** 2,
                                 axis=2))
        mean_error = np.mean(errors)
        max_error = np.max(errors)
    else:
        mean_error = 0
        max_error = 0

    pairwise_errors.append(mean_error)
    print(f"    Reprojection error: mean={mean_error:.2f}px, max={max_error:.2f}px")


# ============================================================
# LANGKAH 3: Menghitung Chain Homography ke Referensi
# ============================================================
print("\n[LANGKAH 3] Menghitung chain homography ke gambar referensi...")

# Array untuk menyimpan homography dari setiap gambar ke referensi
H_to_ref = [None] * n_images

# Gambar referensi menggunakan identity
H_to_ref[ref_idx] = np.eye(3, dtype=np.float64)

# Chain ke kanan dari referensi: H(i→ref) = H(i-1→ref) @ H(i→i-1)
for i in range(ref_idx + 1, n_images):
    # H(i→i-1) = inverse dari H(i-1→i)
    H_inv = np.linalg.inv(pairwise_H[i - 1])
    # Chain: H(i→ref) = H(i-1→ref) @ H(i→i-1)
    H_to_ref[i] = H_to_ref[i - 1] @ H_inv

# Chain ke kiri dari referensi: H(i→ref) = H(i+1→ref) @ H(i→i+1)
for i in range(ref_idx - 1, -1, -1):
    # H(i→i+1) = pairwise_H[i]
    # Chain: H(i→ref) = H(i+1→ref) @ H(i→i+1)
    H_to_ref[i] = H_to_ref[i + 1] @ pairwise_H[i]

# Menampilkan chain homography
for i in range(n_images):
    # Menghitung "chain length" (jumlah perkalian dari referensi)
    chain_len = abs(i - ref_idx)
    print(f"  Gambar {i + 1}: chain length = {chain_len}")


# ============================================================
# LANGKAH 4: Stitching dengan Chain Homography (Metode A)
# ============================================================
print("\n[LANGKAH 4] Melakukan stitching dengan chain homography (Metode A)...")

# Mengukur waktu stitching manual
t_start_manual = time.time()

# Menentukan ukuran canvas berdasarkan transformasi sudut gambar
h_img, w_img = images_wide[0].shape[:2]

# Menghitung posisi sudut setiap gambar setelah transformasi
all_corners = []
for i in range(n_images):
    # Sudut gambar dalam format (x, y)
    corners = np.float32([[0, 0], [w_img, 0],
                           [w_img, h_img], [0, h_img]]).reshape(-1, 1, 2)

    # Mentransformasi sudut menggunakan homography ke referensi
    transformed = cv2.perspectiveTransform(corners, H_to_ref[i])
    all_corners.append(transformed)

# Menghitung batas canvas
all_corners_cat = np.concatenate(all_corners)
x_min = int(np.floor(all_corners_cat[:, 0, 0].min()))
y_min = int(np.floor(all_corners_cat[:, 0, 1].min()))
x_max = int(np.ceil(all_corners_cat[:, 0, 0].max()))
y_max = int(np.ceil(all_corners_cat[:, 0, 1].max()))

# Membatasi ukuran canvas agar tidak terlalu besar
canvas_w = min(x_max - x_min, 8000)
canvas_h = min(y_max - y_min, 4000)

print(f"  Canvas: {canvas_w}x{canvas_h} piksel")

# Matriks translasi untuk menggeser canvas agar semua koordinat positif
H_translate = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
], dtype=np.float64)

# Membuat canvas dan melakukan warping setiap gambar
canvas_manual = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
count_manual = np.zeros((canvas_h, canvas_w), dtype=np.float32)

for i in range(n_images):
    # Menghitung homography gabungan (translasi + chain)
    H_combined = H_translate @ H_to_ref[i]

    # Melakukan perspective warping
    warped = cv2.warpPerspective(
        images_wide[i], H_combined,
        (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Membuat mask non-hitam
    mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

    # Menambahkan ke akumulator (averaging blending)
    for c in range(3):
        canvas_manual[:, :, c] += warped[:, :, c].astype(np.float64) * mask
    count_manual += mask

# Normalisasi canvas manual
count_manual[count_manual == 0] = 1
result_manual = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
for c in range(3):
    result_manual[:, :, c] = np.clip(
        canvas_manual[:, :, c] / count_manual, 0, 255
    ).astype(np.uint8)

t_manual = time.time() - t_start_manual

# Crop border hitam
gray_m = cv2.cvtColor(result_manual, cv2.COLOR_BGR2GRAY)
_, thresh_m = cv2.threshold(gray_m, 1, 255, cv2.THRESH_BINARY)
contours_m, _ = cv2.findContours(thresh_m, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
if contours_m:
    largest_m = max(contours_m, key=cv2.contourArea)
    xm, ym, wm, hm = cv2.boundingRect(largest_m)
    result_manual_cropped = result_manual[ym:ym + hm, xm:xm + wm]
else:
    result_manual_cropped = result_manual

# Menyimpan hasil method A
cv2.imwrite(os.path.join(OUTPUT_DIR, "07_chain_homography_no_ba.jpg"),
            result_manual_cropped)
print(f"  [OK] Stitching chain homography tanpa BA disimpan.")
print(f"  Ukuran: {result_manual_cropped.shape[1]}x{result_manual_cropped.shape[0]}")
print(f"  Waktu: {t_manual:.3f} detik")


# ============================================================
# LANGKAH 5: Stitching dengan OpenCV Stitcher (Metode B - with BA)
# ============================================================
print("\n[LANGKAH 5] Melakukan stitching dengan OpenCV Stitcher (Metode B)...")
print("  OpenCV Stitcher menggunakan bundle adjustment secara internal.")

# Mengukur waktu stitching dengan Stitcher API
t_start_stitcher = time.time()

# Membuat Stitcher dalam mode PANORAMA
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# Melakukan stitching otomatis (termasuk BA internal)
status_stitch, result_stitcher = stitcher.stitch(images_wide)

t_stitcher = time.time() - t_start_stitcher

if status_stitch == cv2.Stitcher_OK:
    # Crop border hitam
    gray_s = cv2.cvtColor(result_stitcher, cv2.COLOR_BGR2GRAY)
    _, thresh_s = cv2.threshold(gray_s, 1, 255, cv2.THRESH_BINARY)
    contours_s, _ = cv2.findContours(thresh_s, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if contours_s:
        largest_s = max(contours_s, key=cv2.contourArea)
        xs, ys, ws, hs = cv2.boundingRect(largest_s)
        result_stitcher_cropped = result_stitcher[ys:ys + hs, xs:xs + ws]
    else:
        result_stitcher_cropped = result_stitcher

    # Menyimpan hasil method B
    cv2.imwrite(os.path.join(OUTPUT_DIR, "07_stitcher_with_ba.jpg"),
                result_stitcher_cropped)
    print(f"  [OK] Stitching dengan BA (Stitcher API) disimpan.")
    print(f"  Ukuran: {result_stitcher_cropped.shape[1]}x{result_stitcher_cropped.shape[0]}")
    print(f"  Waktu: {t_stitcher:.3f} detik")
else:
    result_stitcher_cropped = None
    print(f"  [WARNING] Stitcher API gagal (status={status_stitch})")


# ============================================================
# LANGKAH 6: Menghitung Reprojection Error (Konsep BA Sederhana)
# ============================================================
print("\n[LANGKAH 6] Menghitung reprojection error untuk konsep BA...")


def hitung_reprojection_error_global(images, homographies_to_ref, ref_idx):
    """
    Menghitung reprojection error global menggunakan chain homography.
    Ini menunjukkan akumulasi error pada stitching tanpa BA.

    Parameter:
    - images            : List gambar
    - homographies_to_ref : List matriks H ke referensi
    - ref_idx           : Indeks gambar referensi

    Returns:
    - errors_per_pair  : Error rata-rata per pasangan
    - total_mean_error : Error rata-rata total
    """
    # Membuat detektor SIFT
    sift = cv2.SIFT_create()

    errors_per_pair = []

    # Menghitung error untuk setiap pasangan bersebelahan
    for i in range(len(images) - 1):
        # Mendeteksi fitur pada kedua gambar
        gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)

        kp1, desc1 = sift.detectAndCompute(gray1, None)
        kp2, desc2 = sift.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            errors_per_pair.append(0)
            continue

        # Matching fitur
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 4:
            errors_per_pair.append(0)
            continue

        # Mengekstrak titik korespondensi
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Mentransformasi kedua set titik ke ruang referensi
        pts1_ref = cv2.perspectiveTransform(pts1, homographies_to_ref[i])
        pts2_ref = cv2.perspectiveTransform(pts2, homographies_to_ref[i + 1])

        # Menghitung error sebagai jarak antara titik yang seharusnya sama
        distances = np.sqrt(np.sum((pts1_ref - pts2_ref) ** 2, axis=2))
        mean_err = np.mean(distances)
        errors_per_pair.append(mean_err)

    # Menghitung error total
    total_mean = np.mean(errors_per_pair) if errors_per_pair else 0

    return errors_per_pair, total_mean


# Menghitung reprojection error pada chain homography (tanpa BA)
print("  Menghitung reprojection error pada chain homography...")
errors_chain, total_error_chain = hitung_reprojection_error_global(
    images_wide, H_to_ref, ref_idx
)

print(f"\n  Reprojection Error (Chain Homography - tanpa BA):")
for i, err in enumerate(errors_chain):
    print(f"    Pasangan ({i + 1},{i + 2}): {err:.2f} piksel")
print(f"    RATA-RATA TOTAL: {total_error_chain:.2f} piksel")


# ============================================================
# LANGKAH 7: Perbandingan Stitching Side-by-Side
# ============================================================
print("\n[LANGKAH 7] Membuat perbandingan side-by-side...")

# Menyiapkan gambar untuk perbandingan
fig1, axes1 = plt.subplots(2, 1, figsize=(16, 10))

# Subplot atas: Chain homography (tanpa BA)
axes1[0].imshow(cv2.cvtColor(result_manual_cropped, cv2.COLOR_BGR2RGB))
axes1[0].set_title(f"Metode A: Chain Homography TANPA Bundle Adjustment\n"
                   f"Reprojection Error: {total_error_chain:.2f}px | "
                   f"Waktu: {t_manual:.3f}s", fontsize=11)
axes1[0].axis("off")

# Subplot bawah: Stitcher API (dengan BA)
if result_stitcher_cropped is not None:
    axes1[1].imshow(cv2.cvtColor(result_stitcher_cropped, cv2.COLOR_BGR2RGB))
    axes1[1].set_title(f"Metode B: OpenCV Stitcher DENGAN Bundle Adjustment\n"
                       f"(BA built-in) | Waktu: {t_stitcher:.3f}s", fontsize=11)
else:
    axes1[1].text(0.5, 0.5, "GAGAL", ha='center', va='center', fontsize=20)
    axes1[1].set_title("Metode B: OpenCV Stitcher (GAGAL)", fontsize=11)
axes1[1].axis("off")

plt.suptitle("Percobaan 7: Perbandingan Chain Homography vs Bundle Adjustment",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_grid_comparison_ba.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Perbandingan side-by-side disimpan.")
plt.close()


# ============================================================
# LANGKAH 8: Zoom Area Seam/Alignment
# ============================================================
print("\n[LANGKAH 8] Melakukan zoom pada area seam/alignment...")

# Mengambil area tengah panorama (area overlap paling banyak)
def zoom_area_tengah(img, label, prefix):
    """
    Mengambil crop bagian tengah panorama untuk melihat detail alignment.
    """
    h, w = img.shape[:2]

    # Area tengah (25% tengah horizontal, seluruh tinggi)
    x_start = w // 4
    x_end = 3 * w // 4
    crop_tengah = img[:, x_start:x_end]

    # Menyimpan crop
    path = os.path.join(OUTPUT_DIR, f"07_zoom_tengah_{prefix}.jpg")
    cv2.imwrite(path, crop_tengah)
    print(f"  Zoom tengah {label}: {crop_tengah.shape[1]}x{crop_tengah.shape[0]}")
    return crop_tengah


# Zoom area tengah dari kedua metode
zoom_manual = zoom_area_tengah(result_manual_cropped, "Chain Homography", "chain")
if result_stitcher_cropped is not None:
    zoom_stitcher = zoom_area_tengah(result_stitcher_cropped, "Stitcher+BA", "stitcher")
else:
    zoom_stitcher = None


# ============================================================
# LANGKAH 9: Efek Jumlah Gambar terhadap Akumulasi Error
# ============================================================
print("\n[LANGKAH 9] Menganalisis efek jumlah gambar pada akumulasi error...")

# Menguji stitching dengan 3, 4, dan 5 gambar
error_vs_n_images = {}
time_chain_vs_n = {}
time_stitcher_vs_n = {}

for n_test in [3, 4, 5]:
    print(f"\n  --- Tes dengan {n_test} gambar ---")

    # Mengambil subset gambar
    subset = images_wide[:n_test]
    sub_ref = n_test // 2

    # Chain homography untuk subset
    sub_pairwise_H = pairwise_H[:n_test - 1]
    sub_H_to_ref = [None] * n_test
    sub_H_to_ref[sub_ref] = np.eye(3, dtype=np.float64)

    # Chain ke kanan
    for i in range(sub_ref + 1, n_test):
        H_inv = np.linalg.inv(sub_pairwise_H[i - 1])
        sub_H_to_ref[i] = sub_H_to_ref[i - 1] @ H_inv

    # Chain ke kiri
    for i in range(sub_ref - 1, -1, -1):
        sub_H_to_ref[i] = sub_H_to_ref[i + 1] @ sub_pairwise_H[i]

    # Menghitung error chain
    sub_errors, sub_total = hitung_reprojection_error_global(
        subset, sub_H_to_ref, sub_ref
    )
    error_vs_n_images[n_test] = sub_total
    print(f"    Reprojection error (chain): {sub_total:.2f}px")

    # Stitching manual chain
    t_start = time.time()

    # Menghitung batas canvas untuk subset
    sub_corners = []
    for i in range(n_test):
        corners = np.float32([[0, 0], [w_img, 0],
                               [w_img, h_img], [0, h_img]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, sub_H_to_ref[i])
        sub_corners.append(transformed)

    sub_all = np.concatenate(sub_corners)
    sx_min = int(np.floor(sub_all[:, 0, 0].min()))
    sy_min = int(np.floor(sub_all[:, 0, 1].min()))
    sx_max = int(np.ceil(sub_all[:, 0, 0].max()))
    sy_max = int(np.ceil(sub_all[:, 0, 1].max()))

    scw = min(sx_max - sx_min, 8000)
    sch = min(sy_max - sy_min, 4000)

    sH_translate = np.array([[1, 0, -sx_min], [0, 1, -sy_min], [0, 0, 1]],
                             dtype=np.float64)

    sub_canvas = np.zeros((sch, scw, 3), dtype=np.float64)
    sub_count = np.zeros((sch, scw), dtype=np.float32)

    for i in range(n_test):
        H_comb = sH_translate @ sub_H_to_ref[i]
        warped = cv2.warpPerspective(subset[i], H_comb, (scw, sch),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
        mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
        for c in range(3):
            sub_canvas[:, :, c] += warped[:, :, c].astype(np.float64) * mask
        sub_count += mask

    sub_count[sub_count == 0] = 1
    sub_result = np.zeros((sch, scw, 3), dtype=np.uint8)
    for c in range(3):
        sub_result[:, :, c] = np.clip(
            sub_canvas[:, :, c] / sub_count, 0, 255
        ).astype(np.uint8)

    time_chain_vs_n[n_test] = time.time() - t_start

    # Crop dan simpan
    gray_sub = cv2.cvtColor(sub_result, cv2.COLOR_BGR2GRAY)
    _, thresh_sub = cv2.threshold(gray_sub, 1, 255, cv2.THRESH_BINARY)
    cnt_sub, _ = cv2.findContours(thresh_sub, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if cnt_sub:
        largest_sub = max(cnt_sub, key=cv2.contourArea)
        xsub, ysub, wsub, hsub = cv2.boundingRect(largest_sub)
        sub_result_cropped = sub_result[ysub:ysub + hsub, xsub:xsub + wsub]
    else:
        sub_result_cropped = sub_result

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"07_chain_{n_test}img.jpg"),
                sub_result_cropped)

    # Stitcher API untuk subset
    t_start = time.time()
    stitcher_sub = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    st_sub, res_sub = stitcher_sub.stitch(subset)
    time_stitcher_vs_n[n_test] = time.time() - t_start

    if st_sub == cv2.Stitcher_OK:
        # Crop dan simpan
        g_sub = cv2.cvtColor(res_sub, cv2.COLOR_BGR2GRAY)
        _, t_sub = cv2.threshold(g_sub, 1, 255, cv2.THRESH_BINARY)
        c_sub, _ = cv2.findContours(t_sub, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        if c_sub:
            lg = max(c_sub, key=cv2.contourArea)
            xg, yg, wg, hg = cv2.boundingRect(lg)
            res_sub_cropped = res_sub[yg:yg + hg, xg:xg + wg]
        else:
            res_sub_cropped = res_sub
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"07_stitcher_{n_test}img.jpg"),
                    res_sub_cropped)
        print(f"    Stitcher: OK, waktu={time_stitcher_vs_n[n_test]:.3f}s")
    else:
        print(f"    Stitcher: GAGAL")

    print(f"    Chain: waktu={time_chain_vs_n[n_test]:.3f}s")


# ============================================================
# LANGKAH 10: Visualisasi Reprojection Error per Pasangan
# ============================================================
print("\n[LANGKAH 10] Membuat visualisasi reprojection error...")

# Membuat grafik error per pasangan
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Grafik 1: Error per pasangan (chain homography)
x_pairs = [f"({i + 1},{i + 2})" for i in range(len(pairwise_errors))]
bars = axes2[0].bar(x_pairs, pairwise_errors, color='steelblue', alpha=0.8)
axes2[0].set_xlabel("Pasangan Gambar", fontsize=10)
axes2[0].set_ylabel("Reprojection Error (piksel)", fontsize=10)
axes2[0].set_title("Pairwise Reprojection Error", fontsize=12)

# Menambahkan label nilai pada bar
for bar, val in zip(bars, pairwise_errors):
    axes2[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                  f"{val:.2f}", ha='center', va='bottom', fontsize=9)

# Grafik 2: Error vs jumlah gambar
n_list = sorted(error_vs_n_images.keys())
err_list = [error_vs_n_images[n] for n in n_list]
axes2[1].plot(n_list, err_list, 'ro-', linewidth=2, markersize=8, label="Chain Homography")
axes2[1].set_xlabel("Jumlah Gambar", fontsize=10)
axes2[1].set_ylabel("Reprojection Error Rata-rata (piksel)", fontsize=10)
axes2[1].set_title("Efek Jumlah Gambar pada Error", fontsize=12)
axes2[1].set_xticks(n_list)
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.suptitle("Percobaan 7: Analisis Reprojection Error",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_grid_reprojection_error.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grafik reprojection error disimpan.")
plt.close()


# ============================================================
# LANGKAH 11: Grid Perbandingan per Jumlah Gambar
# ============================================================
print("\n[LANGKAH 11] Membuat grid perbandingan per jumlah gambar...")

# 3 baris (3,4,5 gambar) x 2 kolom (chain vs stitcher)
fig3, axes3 = plt.subplots(3, 2, figsize=(16, 14))

for idx, n_test in enumerate([3, 4, 5]):
    # Kolom kiri: chain homography
    chain_path = os.path.join(OUTPUT_DIR, f"07_chain_{n_test}img.jpg")
    chain_img = cv2.imread(chain_path)
    if chain_img is not None:
        axes3[idx, 0].imshow(cv2.cvtColor(chain_img, cv2.COLOR_BGR2RGB))
        err_val = error_vs_n_images.get(n_test, 0)
        axes3[idx, 0].set_title(f"Chain ({n_test} gambar) - Error: {err_val:.2f}px",
                                 fontsize=10)
    else:
        axes3[idx, 0].text(0.5, 0.5, "N/A", ha='center', fontsize=14)
    axes3[idx, 0].axis("off")

    # Kolom kanan: stitcher API
    stitch_path = os.path.join(OUTPUT_DIR, f"07_stitcher_{n_test}img.jpg")
    stitch_img = cv2.imread(stitch_path)
    if stitch_img is not None:
        axes3[idx, 1].imshow(cv2.cvtColor(stitch_img, cv2.COLOR_BGR2RGB))
        axes3[idx, 1].set_title(f"Stitcher+BA ({n_test} gambar)", fontsize=10)
    else:
        axes3[idx, 1].text(0.5, 0.5, "GAGAL", ha='center', fontsize=14)
        axes3[idx, 1].set_title(f"Stitcher+BA ({n_test} gambar) - GAGAL",
                                 fontsize=10)
    axes3[idx, 1].axis("off")

plt.suptitle("Percobaan 7: Chain Homography vs Stitcher+BA (3/4/5 Gambar)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_grid_n_images_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid perbandingan per jumlah gambar disimpan.")
plt.close()


# ============================================================
# LANGKAH 12: Grid Zoom Area Alignment
# ============================================================
print("\n[LANGKAH 12] Membuat grid zoom area alignment...")

fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))

# Zoom chain homography
if zoom_manual is not None:
    axes4[0].imshow(cv2.cvtColor(zoom_manual, cv2.COLOR_BGR2RGB))
    axes4[0].set_title("Zoom Tengah: Chain Homography (tanpa BA)", fontsize=11)
axes4[0].axis("off")

# Zoom stitcher
if zoom_stitcher is not None:
    axes4[1].imshow(cv2.cvtColor(zoom_stitcher, cv2.COLOR_BGR2RGB))
    axes4[1].set_title("Zoom Tengah: Stitcher API (dengan BA)", fontsize=11)
else:
    axes4[1].text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=16)
axes4[1].axis("off")

plt.suptitle("Percobaan 7: Zoom Area Alignment",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_grid_zoom_alignment.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid zoom alignment disimpan.")
plt.close()


# ============================================================
# LANGKAH 13: Ringkasan dan Statistik Lengkap
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 7: BUNDLE ADJUSTMENT (KONSEP)")
print("=" * 65)

# Tabel pairwise error
print("\n  Tabel Pairwise Reprojection Error:")
print(f"  {'Pasangan':<12} | {'Matches':>8} | {'Inliers':>8} | {'Error (px)':>10}")
print(f"  {'-' * 12}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}")
for i in range(len(pairwise_errors)):
    print(f"  ({i + 1},{i + 2}){'':<7} | "
          f"{pairwise_matches[i]:>8} | "
          f"{pairwise_inliers[i]:>8} | "
          f"{pairwise_errors[i]:>9.2f}")

# Tabel efek jumlah gambar
print(f"\n  Efek Jumlah Gambar pada Reprojection Error:")
print(f"  {'N Gambar':>8} | {'Error Chain':>12} | {'Waktu Chain':>12} | {'Waktu Stitcher':>14}")
print(f"  {'-' * 8}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 14}")
for n_test in [3, 4, 5]:
    err = error_vs_n_images.get(n_test, 0)
    tc = time_chain_vs_n.get(n_test, 0)
    ts = time_stitcher_vs_n.get(n_test, 0)
    print(f"  {n_test:>8} | {err:>11.2f}px | {tc:>11.3f}s | {ts:>13.3f}s")

# Tabel perbandingan metode
print(f"\n  Perbandingan Metode (5 Gambar):")
print(f"  {'Metode':<30} | {'Waktu (s)':>10}")
print(f"  {'-' * 30}-+-{'-' * 10}")
print(f"  {'Chain Homography (tanpa BA)':<30} | {t_manual:>10.3f}")
print(f"  {'OpenCV Stitcher (dengan BA)':<30} | {t_stitcher:>10.3f}")

# Penjelasan konsep BA
print("\n  Konsep Bundle Adjustment:")
print("  - BA mengoptimasi SEMUA parameter kamera secara simultan")
print("  - Meminimalkan total reprojection error di semua pasangan")
print("  - Chain homography: error terakumulasi → drift/misalignment")
print("  - BA: error didistribusikan merata → alignment lebih baik")
print("  - Semakin banyak gambar, semakin terlihat manfaat BA")
print("  - OpenCV Stitcher mengintegrasikan BA secara otomatis")

# Daftar output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("07_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.Stitcher_create()      → Stitcher dengan BA internal")
print("    cv2.findHomography()       → Estimasi homography (RANSAC)")
print("    cv2.perspectiveTransform() → Transform titik untuk error")
print("    cv2.warpPerspective()      → Warping perspektif gambar")
print("    np.linalg.inv()            → Invers matriks homography")
print("    np.matmul() / @            → Chain multiplication homography")
print("=" * 65)
