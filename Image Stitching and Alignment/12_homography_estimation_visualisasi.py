"""
==========================================================================
PERCOBAAN 12: HOMOGRAPHY ESTIMATION DAN VISUALISASI
==========================================================================
Program ini mempelajari homography secara mendalam: estimasi, dekomposisi,
dan visualisasi efek transformasi perspektif. Homography adalah matriks 3x3
yang memetakan titik-titik dari satu bidang ke bidang lain, dan merupakan
fondasi utama dalam image stitching.

Konsep yang dipelajari:
- Homography: definisi, derajat kebebasan (8 DOF), dan interpretasi
- Estimasi homography dari 4 pasangan titik (exact solution)
- Estimasi homography dari banyak titik (over-determined, RANSAC)
- Dekomposisi homography: translasi, rotasi, skala
- Reprojection error sebagai metrik kualitas estimasi
- Robustness terhadap noise dan outlier
- Visualisasi efek setiap komponen transformasi

Fungsi utama yang dipelajari:
- cv2.findHomography()        : Estimasi homography dari korespondensi titik
- cv2.getPerspectiveTransform(): Homography dari 4 pasangan titik (exact)
- cv2.perspectiveTransform()  : Transformasi titik dengan homography
- cv2.warpPerspective()       : Warping gambar dengan homography matrix
- np.linalg.svd()             : SVD untuk analisis matriks homography
- cv2.drawMatches()           : Visualisasi kecocokan fitur
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan computer vision
import cv2

# Mengimpor library NumPy untuk operasi array, matriks, dan aljabar linear
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
print("PERCOBAAN 12: HOMOGRAPHY ESTIMATION DAN VISUALISASI")
print("=" * 65)


# ============================================================
# LANGKAH 1: Homography dari 4 Pasangan Titik (Exact Solution)
# ============================================================
print("\n[LANGKAH 1] Menghitung homography dari 4 pasangan titik...")

# Memuat gambar grid test untuk visualisasi efek transformasi
img_grid = cv2.imread(os.path.join(IMAGE_DIR, "grid_test.jpg"))
if img_grid is None:
    print("[ERROR] grid_test.jpg tidak ditemukan! Jalankan download_image.py.")
    exit()

print(f"  Gambar grid test: {img_grid.shape[1]}x{img_grid.shape[0]}")

# Mendapatkan dimensi gambar
h_img, w_img = img_grid.shape[:2]

# Mendefinisikan 4 titik sumber (sudut gambar asli)
# Titik-titik ini membentuk persegi panjang dari gambar asli
src_pts_4 = np.float32([
    [0, 0],                  # Sudut kiri atas
    [w_img - 1, 0],          # Sudut kanan atas
    [w_img - 1, h_img - 1],  # Sudut kanan bawah
    [0, h_img - 1]           # Sudut kiri bawah
])

# Mendefinisikan 4 titik destinasi (sudut gambar yang ditransformasi)
# Titik ini membentuk trapesium → efek perspektif
dst_pts_4 = np.float32([
    [50, 30],                    # Kiri atas bergeser ke dalam
    [w_img - 60, 20],            # Kanan atas bergeser masuk
    [w_img + 20, h_img - 40],    # Kanan bawah bergeser keluar
    [-30, h_img - 50]            # Kiri bawah bergeser keluar
])

# Menghitung homography menggunakan getPerspectiveTransform (4 titik exact)
# Ini memberikan solusi exact karena 4 korespondensi = 8 persamaan untuk 8 DOF
H_4pt = cv2.getPerspectiveTransform(src_pts_4, dst_pts_4)

# Menampilkan matriks homography
print(f"\n  Matriks Homography (4-point exact):")
print(f"  {'-' * 45}")
for i in range(3):
    print(f"  | {H_4pt[i, 0]:12.6f}  {H_4pt[i, 1]:12.6f}  {H_4pt[i, 2]:12.6f} |")
print(f"  {'-' * 45}")

# Menampilkan informasi titik korespondensi
print(f"\n  Titik sumber → Titik destinasi:")
for i in range(4):
    print(f"    ({src_pts_4[i][0]:.0f}, {src_pts_4[i][1]:.0f}) → "
          f"({dst_pts_4[i][0]:.0f}, {dst_pts_4[i][1]:.0f})")


# ============================================================
# LANGKAH 2: Visualisasi Efek Homography (Warping)
# ============================================================
print("\n[LANGKAH 2] Melakukan warping gambar menggunakan homography...")

# Menghitung ukuran output yang cukup untuk menampung gambar yang ditransformasi
# Mentransformasi keempat sudut untuk menentukan batas
corners = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]).reshape(-1, 1, 2)
corners_warped = cv2.perspectiveTransform(corners, H_4pt)

# Menghitung batas kanvas
all_pts = corners_warped.reshape(-1, 2)
x_min = int(np.floor(all_pts[:, 0].min()))
y_min = int(np.floor(all_pts[:, 1].min()))
x_max = int(np.ceil(all_pts[:, 0].max()))
y_max = int(np.ceil(all_pts[:, 1].max()))

# Menghitung offset translasi agar semua koordinat positif
tx = max(-x_min, 0)
ty = max(-y_min, 0)
T_offset = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

# Ukuran canvas output
out_w = x_max - x_min + 1
out_h = y_max - y_min + 1

# Melakukan warping dengan matriks gabungan (translasi + homography)
img_warped_4pt = cv2.warpPerspective(img_grid, T_offset @ H_4pt, (out_w, out_h))

# Menyimpan hasil warping
cv2.imwrite(os.path.join(OUTPUT_DIR, "12_warped_4point.jpg"), img_warped_4pt)
print(f"  Gambar asli   : {w_img}x{h_img}")
print(f"  Gambar warped : {out_w}x{out_h}")
print("  [OK] Hasil warping 4-point disimpan.")


# ============================================================
# LANGKAH 3: Dekomposisi Homography (Analisis Komponen)
# ============================================================
print("\n[LANGKAH 3] Dekomposisi matriks homography...")

def dekomposisi_homography(H):
    """
    Melakukan dekomposisi sederhana matriks homography 3x3 untuk
    menganalisis komponen translasi, rotasi, skala, dan shear.

    Homography H dapat dianggap sebagai:
    H ≈ [sR  t; v^T  1] dimana s=skala, R=rotasi, t=translasi, v=perspektif

    Parameter:
    - H : Matriks homography 3x3

    Returns:
    - info : Dictionary berisi komponen dekomposisi
    """
    # Normalisasi H agar H[2,2] = 1 (konvensi standar)
    H_norm = H / H[2, 2]

    # Mengekstrak komponen translasi (kolom ketiga, baris 0 dan 1)
    tx = H_norm[0, 2]
    ty = H_norm[1, 2]

    # Mengekstrak sub-matriks 2x2 (bagian rotasi + skala + shear)
    A = H_norm[:2, :2]

    # Melakukan SVD pada sub-matriks A untuk mendapatkan rotasi dan skala
    # A = U @ S @ Vt dimana U,Vt = rotasi, S = skala
    U, S, Vt = np.linalg.svd(A)

    # Menghitung skala (rata-rata singular values)
    skala_x = S[0]
    skala_y = S[1]

    # Menghitung sudut rotasi dari matriks rotasi R = U @ Vt
    R = U @ Vt
    sudut_rad = np.arctan2(R[1, 0], R[0, 0])
    sudut_deg = np.degrees(sudut_rad)

    # Mengekstrak komponen perspektif (baris terakhir)
    v1 = H_norm[2, 0]
    v2 = H_norm[2, 1]

    # Mengumpulkan hasil dekomposisi
    info = {
        "translasi": (tx, ty),
        "rotasi_deg": sudut_deg,
        "skala": (skala_x, skala_y),
        "perspektif": (v1, v2),
        "singular_values": S,
        "H_norm": H_norm
    }

    return info


# Melakukan dekomposisi pada homography 4-point
info_4pt = dekomposisi_homography(H_4pt)

# Menampilkan hasil dekomposisi
print(f"\n  Komponen Homography (4-point):")
print(f"    Translasi     : tx={info_4pt['translasi'][0]:.2f}, ty={info_4pt['translasi'][1]:.2f}")
print(f"    Rotasi        : {info_4pt['rotasi_deg']:.2f} derajat")
print(f"    Skala         : sx={info_4pt['skala'][0]:.4f}, sy={info_4pt['skala'][1]:.4f}")
print(f"    Perspektif    : v1={info_4pt['perspektif'][0]:.6f}, v2={info_4pt['perspektif'][1]:.6f}")
print(f"    Singular values: {info_4pt['singular_values']}")


# ============================================================
# LANGKAH 4: Estimasi Homography dari Feature Matches (SIFT+FLANN)
# ============================================================
print("\n[LANGKAH 4] Estimasi homography dari feature matches (SIFT+FLANN)...")

# Memuat gambar pasangan untuk stitching
img_left = cv2.imread(os.path.join(IMAGE_DIR, "pair_left.jpg"))
img_right = cv2.imread(os.path.join(IMAGE_DIR, "pair_right.jpg"))

if img_left is None or img_right is None:
    print("[ERROR] Gambar pair tidak ditemukan! Jalankan download_image.py.")
    exit()

print(f"  Gambar kiri  : {img_left.shape[1]}x{img_left.shape[0]}")
print(f"  Gambar kanan : {img_right.shape[1]}x{img_right.shape[0]}")

# Mengkonversi ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Mendeteksi fitur SIFT pada kedua gambar
sift = cv2.SIFT_create()
kp_left, desc_left = sift.detectAndCompute(gray_left, None)
kp_right, desc_right = sift.detectAndCompute(gray_right, None)
print(f"  Keypoints kiri : {len(kp_left)}")
print(f"  Keypoints kanan: {len(kp_right)}")

# Mencocokkan fitur menggunakan FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches_knn = flann.knnMatch(desc_left, desc_right, k=2)

# Menerapkan Lowe's ratio test
good_matches = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"  Good matches   : {len(good_matches)}")

# Mengekstrak titik korespondensi
src_pts_feat = np.float32([kp_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts_feat = np.float32([kp_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Mengestimasi homography menggunakan RANSAC
waktu_start = time.time()
H_ransac, mask_ransac = cv2.findHomography(src_pts_feat, dst_pts_feat, cv2.RANSAC, 5.0)
waktu_ransac = time.time() - waktu_start

# Menghitung statistik inlier
n_inlier_ransac = int(mask_ransac.ravel().sum())
n_total = len(good_matches)

print(f"\n  Homography (RANSAC):")
print(f"  {'-' * 45}")
for i in range(3):
    print(f"  | {H_ransac[i, 0]:12.6f}  {H_ransac[i, 1]:12.6f}  {H_ransac[i, 2]:12.6f} |")
print(f"  {'-' * 45}")
print(f"  Inliers: {n_inlier_ransac}/{n_total} ({n_inlier_ransac/n_total*100:.1f}%)")
print(f"  Waktu  : {waktu_ransac*1000:.2f} ms")


# ============================================================
# LANGKAH 5: Perbandingan getPerspectiveTransform vs findHomography
# ============================================================
print("\n[LANGKAH 5] Membandingkan getPerspectiveTransform vs findHomography...")

# Mengambil 4 titik korespondensi dari good matches untuk getPerspectiveTransform
# Memilih 4 titik yang tersebar baik menggunakan langkah seleksi
n_matches = len(good_matches)
indices_4 = [0, n_matches // 3, 2 * n_matches // 3, n_matches - 1]

src_4_from_feat = np.float32([kp_left[good_matches[i].queryIdx].pt for i in indices_4])
dst_4_from_feat = np.float32([kp_right[good_matches[i].trainIdx].pt for i in indices_4])

# Menghitung homography menggunakan getPerspectiveTransform (exact 4-point)
waktu_start = time.time()
H_4pt_feat = cv2.getPerspectiveTransform(src_4_from_feat, dst_4_from_feat)
waktu_4pt = time.time() - waktu_start

print(f"\n  Homography (4-point exact dari fitur):")
print(f"  {'-' * 45}")
for i in range(3):
    print(f"  | {H_4pt_feat[i, 0]:12.6f}  {H_4pt_feat[i, 1]:12.6f}  {H_4pt_feat[i, 2]:12.6f} |")
print(f"  {'-' * 45}")
print(f"  Waktu: {waktu_4pt*1000:.2f} ms")


# ============================================================
# LANGKAH 6: Visualisasi Titik Sumber → Destinasi
# ============================================================
print("\n[LANGKAH 6] Memvisualisasikan transformasi titik (src → dst)...")

# Mentransformasi semua titik sumber menggunakan H_ransac
src_all = src_pts_feat.copy()
dst_predicted = cv2.perspectiveTransform(src_all, H_ransac)

# Membuat visualisasi: titik asli (hijau) vs titik prediksi (merah) vs titik aktual (biru)
vis_pts = img_right.copy()

for i in range(min(50, len(good_matches))):
    # Titik aktual di gambar kanan (biru)
    px_actual = tuple(np.int32(dst_pts_feat[i, 0]))
    cv2.circle(vis_pts, px_actual, 5, (255, 0, 0), -1)

    # Titik prediksi menggunakan H (merah)
    px_pred = tuple(np.int32(dst_predicted[i, 0]))
    cv2.circle(vis_pts, px_pred, 5, (0, 0, 255), -1)

    # Garis penghubung (kuning) menunjukkan error
    cv2.line(vis_pts, px_actual, px_pred, (0, 255, 255), 1)

# Menyimpan visualisasi
cv2.imwrite(os.path.join(OUTPUT_DIR, "12_titik_src_dst_visualisasi.jpg"), vis_pts)
print("  [OK] Visualisasi titik sumber-destinasi disimpan.")
print("  Biru = titik aktual, Merah = prediksi H, Kuning = error vector")


# ============================================================
# LANGKAH 7: Menghitung Reprojection Error
# ============================================================
print("\n[LANGKAH 7] Menghitung reprojection error untuk setiap metode...")

def hitung_reprojection_error(H, src_pts, dst_pts):
    """
    Menghitung reprojection error: jarak antara titik destinasi aktual
    dan titik destinasi yang diprediksi oleh homography H.

    Parameter:
    - H       : Matriks homography 3x3
    - src_pts : Titik sumber (Nx1x2)
    - dst_pts : Titik destinasi aktual (Nx1x2)

    Returns:
    - mean_err  : Rata-rata error (piksel)
    - max_err   : Error maksimum
    - errors    : Array error per titik
    """
    # Mentransformasi titik sumber menggunakan H
    dst_pred = cv2.perspectiveTransform(src_pts, H)

    # Menghitung jarak Euclidean antara prediksi dan aktual
    diff = dst_pred - dst_pts
    errors = np.sqrt(np.sum(diff ** 2, axis=2)).ravel()

    # Menghitung statistik error
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    median_err = np.median(errors)

    return mean_err, max_err, median_err, errors


# Menghitung reprojection error untuk RANSAC homography
mean_r, max_r, med_r, errors_ransac = hitung_reprojection_error(
    H_ransac, src_pts_feat, dst_pts_feat
)

# Menghitung reprojection error untuk 4-point homography
# Menggunakan semua titik (bukan hanya 4 yang dipakai untuk estimasi)
mean_4, max_4, med_4, errors_4pt = hitung_reprojection_error(
    H_4pt_feat, src_pts_feat, dst_pts_feat
)

# Menampilkan perbandingan reprojection error
print(f"\n  {'Metode':<25} | {'Mean Err':>10} | {'Median':>8} | {'Max Err':>10}")
print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
print(f"  {'findHomography (RANSAC)':<25} | {mean_r:>8.3f}px | {med_r:>6.3f}px | {max_r:>8.3f}px")
print(f"  {'getPerspectiveTransform':<25} | {mean_4:>8.3f}px | {med_4:>6.3f}px | {max_4:>8.3f}px")


# ============================================================
# LANGKAH 8: Uji Robustness terhadap Noise
# ============================================================
print("\n[LANGKAH 8] Menguji robustness terhadap noise pada titik korespondensi...")

# Daftar level noise yang akan diuji (standar deviasi dalam piksel)
noise_levels = [0, 1, 2, 5, 10, 20]
noise_results = {}

for sigma in noise_levels:
    # Menambahkan noise Gaussian pada titik destinasi
    if sigma > 0:
        noise = np.random.normal(0, sigma, dst_pts_feat.shape).astype(np.float32)
        dst_noisy = dst_pts_feat + noise
    else:
        dst_noisy = dst_pts_feat.copy()

    # Mengestimasi homography dari titik yang ber-noise
    try:
        H_noisy, mask_noisy = cv2.findHomography(src_pts_feat, dst_noisy, cv2.RANSAC, 5.0)

        if H_noisy is not None:
            # Menghitung reprojection error terhadap titik asli (tanpa noise)
            mean_n, max_n, med_n, _ = hitung_reprojection_error(
                H_noisy, src_pts_feat, dst_pts_feat
            )
            n_inlier_n = int(mask_noisy.ravel().sum())
        else:
            mean_n, max_n, med_n, n_inlier_n = float('inf'), float('inf'), float('inf'), 0

    except Exception:
        mean_n, max_n, med_n, n_inlier_n = float('inf'), float('inf'), float('inf'), 0

    # Menyimpan hasil
    noise_results[sigma] = {
        'mean_err': mean_n,
        'max_err': max_n,
        'inliers': n_inlier_n
    }

    print(f"  Noise σ={sigma:>2d}px: mean_err={mean_n:.3f}px, "
          f"max_err={max_n:.3f}px, inliers={n_inlier_n}")


# ============================================================
# LANGKAH 9: Visualisasi Efek Setiap Parameter H (8 DOF)
# ============================================================
print("\n[LANGKAH 9] Memvisualisasikan efek setiap DOF homography...")

# Homography memiliki 8 derajat kebebasan (Degrees of Freedom):
# h00: skala-x, h01: shear-x, h02: translasi-x
# h10: shear-y, h11: skala-y, h12: translasi-y
# h20: perspektif-x, h21: perspektif-y (h22=1 normalisasi)

# Membuat gambar kecil untuk demo DOF
img_small = cv2.resize(img_grid, (300, 200))
h_s, w_s = img_small.shape[:2]

# Definisi 8 transformasi dasar, masing-masing memodifikasi 1 elemen H
dof_names = [
    "Skala-X (h00=1.3)",
    "Shear-X (h01=0.2)",
    "Translasi-X (h02=40)",
    "Shear-Y (h10=0.2)",
    "Skala-Y (h11=1.3)",
    "Translasi-Y (h12=30)",
    "Perspektif-X (h20=0.0005)",
    "Perspektif-Y (h21=0.0005)"
]

# Membuat matriks H untuk setiap DOF (dimulai dari identitas, ubah 1 elemen)
dof_matrices = []

# DOF 1: Skala X
H_dof = np.eye(3, dtype=np.float64)
H_dof[0, 0] = 1.3
dof_matrices.append(H_dof.copy())

# DOF 2: Shear X
H_dof = np.eye(3, dtype=np.float64)
H_dof[0, 1] = 0.2
dof_matrices.append(H_dof.copy())

# DOF 3: Translasi X
H_dof = np.eye(3, dtype=np.float64)
H_dof[0, 2] = 40
dof_matrices.append(H_dof.copy())

# DOF 4: Shear Y
H_dof = np.eye(3, dtype=np.float64)
H_dof[1, 0] = 0.2
dof_matrices.append(H_dof.copy())

# DOF 5: Skala Y
H_dof = np.eye(3, dtype=np.float64)
H_dof[1, 1] = 1.3
dof_matrices.append(H_dof.copy())

# DOF 6: Translasi Y
H_dof = np.eye(3, dtype=np.float64)
H_dof[1, 2] = 30
dof_matrices.append(H_dof.copy())

# DOF 7: Perspektif X
H_dof = np.eye(3, dtype=np.float64)
H_dof[2, 0] = 0.0005
dof_matrices.append(H_dof.copy())

# DOF 8: Perspektif Y
H_dof = np.eye(3, dtype=np.float64)
H_dof[2, 1] = 0.0005
dof_matrices.append(H_dof.copy())

# Melakukan warping untuk setiap DOF dan menyimpan hasilnya
dof_images = []
for i, (nama, H_d) in enumerate(zip(dof_names, dof_matrices)):
    # Menghitung ukuran output yang aman
    corners_d = np.float32([[0, 0], [w_s, 0], [w_s, h_s], [0, h_s]]).reshape(-1, 1, 2)
    try:
        corners_t = cv2.perspectiveTransform(corners_d, H_d)
        all_c = corners_t.reshape(-1, 2)
        xmin = max(int(np.floor(all_c[:, 0].min())), -200)
        ymin = max(int(np.floor(all_c[:, 1].min())), -200)
        xmax = min(int(np.ceil(all_c[:, 0].max())), w_s + 200)
        ymax = min(int(np.ceil(all_c[:, 1].max())), h_s + 200)

        off_x = max(-xmin, 0)
        off_y = max(-ymin, 0)
        T_d = np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]], dtype=np.float64)
        out_w_d = xmax - xmin + 1
        out_h_d = ymax - ymin + 1

        # Melakukan warping
        warped_d = cv2.warpPerspective(img_small, T_d @ H_d, (out_w_d, out_h_d))
        dof_images.append(warped_d)
    except Exception:
        dof_images.append(img_small.copy())

    print(f"  DOF {i+1}: {nama}")

# Menyimpan setiap hasil DOF secara individual
for i, img_d in enumerate(dof_images):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"12_dof_{i+1}_{dof_names[i][:8].replace(' ', '_')}.jpg"),
                img_d)
print("  [OK] Semua efek DOF disimpan.")


# ============================================================
# LANGKAH 10: Visualisasi Inliers vs Outliers (RANSAC)
# ============================================================
print("\n[LANGKAH 10] Memvisualisasikan inliers vs outliers RANSAC...")

# Memisahkan inliers dan outliers berdasarkan mask RANSAC
inlier_mask = mask_ransac.ravel().astype(bool)
outlier_mask = ~inlier_mask

# Mengonversi ke list DMatch yang terpisah
inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
outlier_matches = [good_matches[i] for i in range(len(good_matches)) if outlier_mask[i]]

print(f"  Total matches : {len(good_matches)}")
print(f"  Inliers       : {len(inlier_matches)}")
print(f"  Outliers      : {len(outlier_matches)}")

# Menggambar inliers (hijau) - kecocokan yang konsisten dengan homography
img_inliers = cv2.drawMatches(
    img_left, kp_left,
    img_right, kp_right,
    inlier_matches[:40],  # Batas 40 untuk kejelasan visual
    None,
    matchColor=(0, 255, 0),     # Hijau untuk inlier
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "12_ransac_inliers.jpg"), img_inliers)

# Menggambar outliers (merah) - kecocokan yang ditolak RANSAC
img_outliers = cv2.drawMatches(
    img_left, kp_left,
    img_right, kp_right,
    outlier_matches[:40],
    None,
    matchColor=(0, 0, 255),     # Merah untuk outlier
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "12_ransac_outliers.jpg"), img_outliers)

# Menggambar gabungan: inliers hijau + outliers merah pada satu gambar
img_combined_io = cv2.drawMatches(
    img_left, kp_left,
    img_right, kp_right,
    inlier_matches[:30],
    None,
    matchColor=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
# Menambahkan outliers di atas
for m in outlier_matches[:20]:
    pt_l = tuple(np.int32(kp_left[m.queryIdx].pt))
    pt_r = tuple(np.int32(kp_right[m.trainIdx].pt))
    # Offset untuk gambar kanan (side by side)
    pt_r_shifted = (pt_r[0] + img_left.shape[1], pt_r[1])
    cv2.circle(img_combined_io, pt_l, 4, (0, 0, 255), -1)
    cv2.circle(img_combined_io, pt_r_shifted, 4, (0, 0, 255), -1)
    cv2.line(img_combined_io, pt_l, pt_r_shifted, (0, 0, 255), 1)

cv2.imwrite(os.path.join(OUTPUT_DIR, "12_ransac_inlier_outlier_combined.jpg"), img_combined_io)
print("  [OK] Visualisasi inliers vs outliers disimpan.")


# ============================================================
# LANGKAH 11: Membuat Grid Visualisasi Komprehensif
# ============================================================
print("\n[LANGKAH 11] Membuat grid visualisasi komprehensif...")

# --- Grid 1: Overview homography (4-point vs RANSAC) ---
fig1, axes1 = plt.subplots(2, 3, figsize=(20, 12))

# Baris 1: 4-point homography
axes1[0, 0].imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title("Gambar Grid Asli", fontsize=11)
axes1[0, 0].axis("off")

axes1[0, 1].imshow(cv2.cvtColor(img_warped_4pt, cv2.COLOR_BGR2RGB))
axes1[0, 1].set_title("Warped (4-Point Exact)", fontsize=11)
axes1[0, 1].axis("off")

# Visualisasi titik src/dst pada gambar asli
vis_4pt = img_grid.copy()
for i in range(4):
    # Titik sumber (hijau)
    pt_s = tuple(np.int32(src_pts_4[i]))
    cv2.circle(vis_4pt, pt_s, 8, (0, 255, 0), -1)
    cv2.putText(vis_4pt, f"S{i+1}", (pt_s[0]+10, pt_s[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Titik destinasi (merah)
    pt_d = tuple(np.int32(dst_pts_4[i]))
    pt_d_clip = (max(0, min(pt_d[0], w_img-1)), max(0, min(pt_d[1], h_img-1)))
    cv2.circle(vis_4pt, pt_d_clip, 8, (0, 0, 255), -1)
axes1[0, 2].imshow(cv2.cvtColor(vis_4pt, cv2.COLOR_BGR2RGB))
axes1[0, 2].set_title("Titik Korespondensi (Hijau→Merah)", fontsize=11)
axes1[0, 2].axis("off")

# Baris 2: Feature-based homography
axes1[1, 0].imshow(cv2.cvtColor(img_inliers, cv2.COLOR_BGR2RGB))
axes1[1, 0].set_title(f"RANSAC Inliers ({len(inlier_matches)})", fontsize=11)
axes1[1, 0].axis("off")

axes1[1, 1].imshow(cv2.cvtColor(img_outliers, cv2.COLOR_BGR2RGB))
axes1[1, 1].set_title(f"RANSAC Outliers ({len(outlier_matches)})", fontsize=11)
axes1[1, 1].axis("off")

axes1[1, 2].imshow(cv2.cvtColor(vis_pts, cv2.COLOR_BGR2RGB))
axes1[1, 2].set_title("Reprojection (Biru=aktual, Merah=prediksi)", fontsize=10)
axes1[1, 2].axis("off")

plt.suptitle("Percobaan 12: Homography Estimation dan Visualisasi",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "12_grid_homography_overview.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid overview homography disimpan.")
plt.close()

# --- Grid 2: 8 DOF visualisasi ---
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))

for i in range(8):
    row = i // 4
    col = i % 4
    ax = axes2[row, col]
    ax.imshow(cv2.cvtColor(dof_images[i], cv2.COLOR_BGR2RGB))
    ax.set_title(dof_names[i], fontsize=9)
    ax.axis("off")

plt.suptitle("8 Derajat Kebebasan (DOF) Matriks Homography\n"
             "Setiap subplot memodifikasi 1 elemen matriks H",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "12_grid_8dof_homography.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid 8 DOF homography disimpan.")
plt.close()

# --- Grid 3: Robustness terhadap noise ---
fig3, (ax_noise1, ax_noise2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Reprojection error vs noise level
sigmas = list(noise_results.keys())
mean_errs = [noise_results[s]['mean_err'] for s in sigmas]
max_errs = [noise_results[s]['max_err'] for s in sigmas]

ax_noise1.plot(sigmas, mean_errs, 'bo-', linewidth=2, markersize=8, label='Mean Error')
ax_noise1.plot(sigmas, max_errs, 'rs--', linewidth=2, markersize=8, label='Max Error')
ax_noise1.set_xlabel("Noise Level σ (piksel)", fontsize=11)
ax_noise1.set_ylabel("Reprojection Error (piksel)", fontsize=11)
ax_noise1.set_title("Reprojection Error vs Noise Level")
ax_noise1.legend()
ax_noise1.grid(True, alpha=0.3)

# Plot 2: Jumlah inlier vs noise level
inliers_noise = [noise_results[s]['inliers'] for s in sigmas]
ax_noise2.bar(range(len(sigmas)), inliers_noise,
              tick_label=[f"σ={s}" for s in sigmas],
              color='#4CAF50', edgecolor='black', linewidth=0.5)
ax_noise2.set_xlabel("Noise Level", fontsize=11)
ax_noise2.set_ylabel("Jumlah Inliers", fontsize=11)
ax_noise2.set_title("Jumlah Inliers RANSAC vs Noise Level")
ax_noise2.grid(True, alpha=0.3, axis='y')

# Menambahkan label di atas bar
for i, val in enumerate(inliers_noise):
    ax_noise2.text(i, val + 1, str(val), ha='center', fontsize=9)

plt.suptitle("Percobaan 12: Robustness Homography terhadap Noise",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "12_grid_noise_robustness.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid noise robustness disimpan.")
plt.close()

# --- Grid 4: Histogram reprojection error ---
fig4, (ax_hist1, ax_hist2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram error untuk RANSAC
ax_hist1.hist(errors_ransac, bins=30, color='#2196F3', edgecolor='black',
              alpha=0.7, linewidth=0.5)
ax_hist1.axvline(mean_r, color='red', linewidth=2, linestyle='--',
                 label=f'Mean={mean_r:.2f}px')
ax_hist1.set_xlabel("Reprojection Error (piksel)")
ax_hist1.set_ylabel("Jumlah Titik")
ax_hist1.set_title("findHomography (RANSAC)")
ax_hist1.legend()

# Histogram error untuk 4-point
ax_hist2.hist(errors_4pt, bins=30, color='#FF9800', edgecolor='black',
              alpha=0.7, linewidth=0.5)
ax_hist2.axvline(mean_4, color='red', linewidth=2, linestyle='--',
                 label=f'Mean={mean_4:.2f}px')
ax_hist2.set_xlabel("Reprojection Error (piksel)")
ax_hist2.set_ylabel("Jumlah Titik")
ax_hist2.set_title("getPerspectiveTransform (4-point)")
ax_hist2.legend()

plt.suptitle("Distribusi Reprojection Error",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "12_grid_reprojection_error.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid reprojection error disimpan.")
plt.close()


# ============================================================
# LANGKAH 12: Ringkasan dan Kesimpulan
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 12: HOMOGRAPHY ESTIMATION DAN VISUALISASI")
print("=" * 65)

# Tabel perbandingan metode
print("\n  Perbandingan Metode Estimasi Homography:")
print(f"  {'Metode':<28} | {'Mean Err':>10} | {'Inliers':>8} | {'Waktu':>8}")
print(f"  {'-'*28}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
print(f"  {'getPerspectiveTransform':<28} | {mean_4:>8.3f}px | {'N/A':>8} | {waktu_4pt*1000:>6.2f}ms")
print(f"  {'findHomography (RANSAC)':<28} | {mean_r:>8.3f}px | {n_inlier_ransac:>8} | {waktu_ransac*1000:>6.2f}ms")

# Dekomposisi komponen
print(f"\n  Dekomposisi Homography (4-point):")
print(f"    Translasi (tx, ty)  : ({info_4pt['translasi'][0]:.2f}, {info_4pt['translasi'][1]:.2f})")
print(f"    Rotasi              : {info_4pt['rotasi_deg']:.2f}°")
print(f"    Skala (sx, sy)      : ({info_4pt['skala'][0]:.4f}, {info_4pt['skala'][1]:.4f})")

# Robustness terhadap noise
print(f"\n  Robustness terhadap Noise:")
for sigma, res in noise_results.items():
    print(f"    σ={sigma:>2d}px: mean_err={res['mean_err']:>7.3f}px, inliers={res['inliers']}")

# Daftar file output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("12_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.findHomography()         → Estimasi H dari banyak titik + RANSAC")
print("    cv2.getPerspectiveTransform() → Estimasi H exact dari 4 titik")
print("    cv2.perspectiveTransform()   → Transformasi titik via H")
print("    cv2.warpPerspective()        → Warping gambar via H")
print("    np.linalg.svd()              → Dekomposisi SVD matriks H")
print("    cv2.drawMatches()            → Visualisasi kecocokan fitur")
print("=" * 65)
