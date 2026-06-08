"""
==========================================================================
PERCOBAAN 14: VERIFIKASI GEOMETRI DETAIL (RANSAC & FUNDAMENTAL MATRIX)
==========================================================================
Program ini melakukan deep-dive ke dalam algoritma RANSAC untuk estimasi
homography, termasuk implementasi RANSAC sederhana dari nol (from scratch).
Selain itu, program ini juga mengestimasi Fundamental Matrix dan
memvisualisasikan garis epipolar.

Konsep yang dipelajari:
- Algoritma RANSAC step-by-step: sample -> fit -> count inliers -> iterate
- Implementasi RANSAC manual untuk estimasi homography
- Perbandingan RANSAC manual vs cv2.findHomography RANSAC
- Fundamental Matrix: hubungan geometri antara dua view (epipolar geometry)
- Garis epipolar: garis di view kedua dimana titik korespondensi berada
- Perbandingan homography vs fundamental matrix untuk verifikasi matching

Fungsi utama yang dipelajari:
- cv2.findHomography()        : Estimasi homography dengan RANSAC
- cv2.findFundamentalMat()    : Estimasi fundamental matrix
- cv2.computeCorrespondEpilines() : Menghitung garis epipolar
- cv2.perspectiveTransform()  : Transform titik menggunakan homography
- np.linalg.svd()             : Singular Value Decomposition untuk DLT

Hasil: Visualisasi iterasi RANSAC, epipolar lines, dan perbandingan metode
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan geometri
import cv2

# Mengimpor NumPy untuk operasi array, matriks, dan aljabar linear
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan
import time

# Mengimpor random untuk sampling acak pada RANSAC manual
import random

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
print("PERCOBAAN 14: VERIFIKASI GEOMETRI DETAIL")
print("=" * 60)

# ============================================================
# 1. Memuat dan Menyiapkan Gambar + Matching
# ============================================================

# Membaca gambar kiri dari pasangan overlapping
img_left = cv2.imread(os.path.join(IMAGE_DIR, "scene_left.jpg"))

# Membaca gambar kanan dari pasangan overlapping
img_right = cv2.imread(os.path.join(IMAGE_DIR, "scene_right.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_left is None or img_right is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi ukuran gambar
print(f"[INFO] Ukuran scene_left: {img_left.shape}")
print(f"[INFO] Ukuran scene_right: {img_right.shape}")

# Mengkonversi gambar kiri ke grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# Mengkonversi gambar kanan ke grayscale
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# Mendeteksi keypoints dan descriptor pada gambar kiri
kp_left, desc_left = sift.detectAndCompute(gray_left, None)

# Mendeteksi keypoints dan descriptor pada gambar kanan
kp_right, desc_right = sift.detectAndCompute(gray_right, None)

# Menampilkan jumlah keypoint
print(f"[SIFT] Keypoints kiri: {len(kp_left)}, kanan: {len(kp_right)}")

# Mendefinisikan parameter FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Membuat matcher FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Melakukan KNN matching
matches_knn = flann.knnMatch(desc_left, desc_right, k=2)

# Menerapkan ratio test
good_matches = []
for m, n in matches_knn:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Menampilkan jumlah good matches
print(f"[MATCH] Good matches: {len(good_matches)}")

# Mengekstrak lokasi keypoint sebagai array
src_pts = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

# ============================================================
# 2. Implementasi RANSAC Manual (Dari Nol)
# ============================================================

# Menampilkan header bagian RANSAC manual
print(f"\n--- Implementasi RANSAC Manual ---")

def hitung_homography_dlt(src, dst):
    """
    Menghitung matriks homography 3x3 dari minimal 4 pasangan titik
    menggunakan Direct Linear Transform (DLT) + SVD.
    """
    # Memeriksa apakah jumlah titik cukup
    assert len(src) >= 4, "Minimal 4 pasangan titik diperlukan"

    # Menyiapkan matriks A untuk sistem persamaan
    A = []

    # Mengiterasi setiap pasangan titik
    for i in range(len(src)):
        # Mengambil koordinat sumber
        x, y = src[i]

        # Mengambil koordinat tujuan
        xp, yp = dst[i]

        # Menambahkan 2 baris ke matriks A sesuai formulasi DLT
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    # Mengkonversi ke array NumPy
    A = np.array(A)

    # Menghitung SVD dari matriks A
    U, S, Vt = np.linalg.svd(A)

    # Mengambil baris terakhir Vt sebagai solusi homography
    H = Vt[-1].reshape(3, 3)

    # Menormalisasi agar H[2,2] = 1
    H = H / H[2, 2]

    return H


def hitung_error_reprojeksi(H, src, dst):
    """
    Menghitung error reprojeksi untuk setiap pasangan titik.
    Error = ||dst - H*src||^2
    """
    # Menyiapkan list untuk menyimpan error
    errors = []

    # Mengiterasi setiap pasangan titik
    for i in range(len(src)):
        # Membuat koordinat homogen sumber
        pt_src = np.array([src[i][0], src[i][1], 1.0])

        # Mentransformasikan titik menggunakan homography
        pt_proj = H @ pt_src

        # Menormalisasi koordinat homogen
        pt_proj = pt_proj / pt_proj[2]

        # Menghitung euclidean distance antara proyeksi dan tujuan
        error = np.sqrt((pt_proj[0] - dst[i][0]) ** 2 + (pt_proj[1] - dst[i][1]) ** 2)

        # Menambahkan error ke list
        errors.append(error)

    return np.array(errors)


def ransac_manual(src, dst, n_iterasi=200, threshold=5.0, n_sample=4):
    """
    Implementasi RANSAC sederhana untuk estimasi homography.
    Mengembalikan homography terbaik, mask inlier, dan log iterasi.
    """
    # Menghitung jumlah total titik
    n_total = len(src)

    # Menyiapkan variabel untuk model terbaik
    best_H = None
    best_inlier_count = 0
    best_mask = np.zeros(n_total, dtype=np.uint8)

    # Menyiapkan log untuk setiap iterasi
    log_iterasi = []

    # Menyiapkan variabel untuk model terburuk
    worst_H = None
    worst_inlier_count = n_total

    # Melakukan iterasi RANSAC
    for it in range(n_iterasi):
        # Memilih 4 sampel acak
        indices = random.sample(range(n_total), n_sample)

        # Mengambil titik sampel sumber
        src_sample = src[indices]

        # Mengambil titik sampel tujuan
        dst_sample = dst[indices]

        # Menghitung homography dari sampel menggunakan DLT
        try:
            H = hitung_homography_dlt(src_sample, dst_sample)
        except Exception:
            # Melewati iterasi jika DLT gagal
            continue

        # Memeriksa apakah homography valid (tidak NaN/Inf)
        if not np.isfinite(H).all():
            continue

        # Menghitung error reprojeksi untuk semua titik
        errors = hitung_error_reprojeksi(H, src, dst)

        # Menentukan inlier (error < threshold)
        mask = (errors < threshold).astype(np.uint8)

        # Menghitung jumlah inlier
        n_inlier = int(mask.sum())

        # Menyimpan log iterasi (setiap 20 iterasi)
        if it % 20 == 0 or n_inlier > best_inlier_count:
            log_iterasi.append({
                'iterasi': it,
                'inlier_count': n_inlier,
                'best_so_far': max(best_inlier_count, n_inlier)
            })

        # Memperbarui model terbaik jika inlier lebih banyak
        if n_inlier > best_inlier_count:
            best_H = H.copy()
            best_inlier_count = n_inlier
            best_mask = mask.copy()

        # Memperbarui model terburuk (inlier terkecil yang masih valid)
        if 0 < n_inlier < worst_inlier_count:
            worst_H = H.copy()
            worst_inlier_count = n_inlier

    return best_H, best_mask, log_iterasi, worst_H, worst_inlier_count

# Menetapkan seed random untuk reprodusibilitas
random.seed(42)

# Mencatat waktu mulai RANSAC manual
t_start = time.time()

# Menjalankan RANSAC manual
H_manual, mask_manual, log_iter, H_worst, worst_count = ransac_manual(
    src_pts, dst_pts, n_iterasi=300, threshold=5.0
)

# Menghitung waktu proses
t_manual = time.time() - t_start

# Menghitung jumlah inlier RANSAC manual
n_inlier_manual = int(mask_manual.sum())

# Menampilkan hasil RANSAC manual
print(f"[RANSAC Manual] Inliers: {n_inlier_manual}/{len(src_pts)}")
print(f"[RANSAC Manual] Waktu: {t_manual * 1000:.1f} ms")
print(f"[RANSAC Manual] Model terburuk: {worst_count} inliers")

# ============================================================
# 3. Perbandingan dengan cv2.findHomography RANSAC
# ============================================================

# Menampilkan header perbandingan
print(f"\n--- Perbandingan dengan OpenCV RANSAC ---")

# Menyiapkan titik dalam format yang dibutuhkan findHomography
src_cv = src_pts.reshape(-1, 1, 2)
dst_cv = dst_pts.reshape(-1, 1, 2)

# Mencatat waktu mulai OpenCV RANSAC
t_start = time.time()

# Menjalankan RANSAC bawaan OpenCV
H_cv, mask_cv = cv2.findHomography(src_cv, dst_cv, cv2.RANSAC, 5.0)

# Menghitung waktu proses
t_cv = time.time() - t_start

# Menghitung jumlah inlier OpenCV
n_inlier_cv = int(mask_cv.sum())

# Menampilkan hasil OpenCV RANSAC
print(f"[OpenCV RANSAC] Inliers: {n_inlier_cv}/{len(src_pts)}")
print(f"[OpenCV RANSAC] Waktu: {t_cv * 1000:.1f} ms")

# Menampilkan perbandingan matriks homography
print(f"\nHomography Manual:")
if H_manual is not None:
    print(H_manual)
print(f"\nHomography OpenCV:")
print(H_cv)

# ============================================================
# 4. Visualisasi Iterasi RANSAC
# ============================================================

# Membuat figure untuk visualisasi progres RANSAC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Grafik progres iterasi ---
# Mengekstrak data dari log iterasi
iter_nums = [entry['iterasi'] for entry in log_iter]
iter_inliers = [entry['inlier_count'] for entry in log_iter]
iter_best = [entry['best_so_far'] for entry in log_iter]

# Menggambar jumlah inlier per iterasi
axes[0].plot(iter_nums, iter_inliers, 'o-', color='lightcoral', alpha=0.6,
             markersize=4, label='Inlier per iterasi')

# Menggambar best-so-far
axes[0].plot(iter_nums, iter_best, 's-', color='steelblue',
             markersize=5, linewidth=2, label='Best so far')

# Menambahkan garis horizontal untuk hasil OpenCV
axes[0].axhline(y=n_inlier_cv, color='green', linestyle='--',
                label=f'OpenCV RANSAC ({n_inlier_cv})')

# Memberikan label sumbu X
axes[0].set_xlabel("Iterasi RANSAC", fontsize=12)

# Memberikan label sumbu Y
axes[0].set_ylabel("Jumlah Inlier", fontsize=12)

# Memberikan judul
axes[0].set_title("Progres RANSAC Manual", fontsize=13)

# Menambahkan legenda
axes[0].legend(fontsize=9)

# Menambahkan grid
axes[0].grid(True, alpha=0.3)

# --- Perbandingan RANSAC Manual vs OpenCV ---
# Menyiapkan data perbandingan
names_cmp = ['RANSAC Manual', 'OpenCV RANSAC']
inliers_cmp = [n_inlier_manual, n_inlier_cv]
times_cmp = [t_manual * 1000, t_cv * 1000]

# Membuat posisi bar
x_pos = np.arange(len(names_cmp))
width = 0.35

# Menggambar bar chart inlier
bars_inl = axes[1].bar(x_pos - width / 2, inliers_cmp, width,
                       label='Inliers', color='steelblue', edgecolor='black')

# Membuat sumbu Y kedua untuk waktu
ax2 = axes[1].twinx()

# Menggambar bar chart waktu
bars_time = ax2.bar(x_pos + width / 2, times_cmp, width,
                    label='Waktu (ms)', color='coral', edgecolor='black')

# Mengatur label sumbu X
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(names_cmp)

# Memberikan label sumbu Y kiri
axes[1].set_ylabel("Jumlah Inlier", color='steelblue', fontsize=12)

# Memberikan label sumbu Y kanan
ax2.set_ylabel("Waktu (ms)", color='coral', fontsize=12)

# Memberikan judul
axes[1].set_title("Manual vs OpenCV RANSAC", fontsize=13)

# Menambahkan legenda gabungan
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=9)

# Menambahkan grid
axes[1].grid(axis='y', alpha=0.3)

# Memberikan judul utama
fig.suptitle("Analisis RANSAC: Iterasi dan Perbandingan", fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi iterasi ke file
plt.savefig(os.path.join(OUTPUT_DIR, "14_ransac_iterasi.png"), dpi=150, bbox_inches='tight')
plt.show()

# Menampilkan pesan
print(f"\n[SAVED] 14_ransac_iterasi.png")

# Menutup figure
plt.close()

# ============================================================
# 5. Visualisasi Best Model vs Worst Model
# ============================================================

# Membuat figure untuk perbandingan model terbaik vs terburuk
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# --- Model Terbaik (RANSAC Manual) ---
# Menyiapkan parameter gambar untuk inlier (hijau) dan outlier (merah)
mask_best_list = mask_manual.tolist()

# Menyiapkan draw params untuk inlier
draw_inlier = dict(matchColor=(0, 255, 0), singlePointColor=None,
                   matchesMask=mask_best_list, flags=cv2.DrawMatchesFlags_DEFAULT)

# Menggambar inlier model terbaik
img_best = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                           good_matches, None, **draw_inlier)

# Menampilkan pada subplot atas
axes[0].imshow(cv2.cvtColor(img_best, cv2.COLOR_BGR2RGB))

# Memberikan judul
axes[0].set_title(f"Model Terbaik RANSAC Manual ({n_inlier_manual} inlier - Hijau)", fontsize=12)

# Menonaktifkan sumbu
axes[0].axis('off')

# --- Model Terburuk ---
# Memeriksa apakah model terburuk tersedia
if H_worst is not None:
    # Menghitung error model terburuk
    errors_worst = hitung_error_reprojeksi(H_worst, src_pts, dst_pts)

    # Menentukan inlier model terburuk
    mask_worst = (errors_worst < 5.0).astype(np.uint8).tolist()

    # Menyiapkan draw params
    draw_worst = dict(matchColor=(0, 0, 255), singlePointColor=None,
                      matchesMask=mask_worst, flags=cv2.DrawMatchesFlags_DEFAULT)

    # Menggambar model terburuk
    img_worst = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                                good_matches, None, **draw_worst)

    # Menampilkan pada subplot bawah
    axes[1].imshow(cv2.cvtColor(img_worst, cv2.COLOR_BGR2RGB))

    # Memberikan judul
    axes[1].set_title(f"Model Terburuk RANSAC Manual ({worst_count} inlier - Merah)", fontsize=12)
else:
    # Menampilkan pesan jika tidak ada model terburuk
    axes[1].text(0.5, 0.5, "Model terburuk tidak tersedia",
                 ha='center', va='center', fontsize=14, transform=axes[1].transAxes)

# Menonaktifkan sumbu
axes[1].axis('off')

# Memberikan judul utama
fig.suptitle("Perbandingan Model Terbaik vs Terburuk RANSAC",
             fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi ke file
plt.savefig(os.path.join(OUTPUT_DIR, "14_ransac_manual.png"), dpi=150, bbox_inches='tight')
plt.show()

# Menampilkan pesan
print(f"[SAVED] 14_ransac_manual.png")

# Menutup figure
plt.close()

# ============================================================
# 6. Estimasi Fundamental Matrix dan Garis Epipolar
# ============================================================

# Menampilkan header bagian fundamental matrix
print(f"\n--- Estimasi Fundamental Matrix ---")

# Menghitung fundamental matrix menggunakan RANSAC
F, mask_fund = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0)

# Menghitung jumlah inlier fundamental matrix
n_inlier_fund = int(mask_fund.sum())

# Menampilkan hasil fundamental matrix
print(f"[Fundamental Matrix] Inliers: {n_inlier_fund}/{len(src_pts)}")
print(f"[Fundamental Matrix] F =")
print(F)

# Mengambil subset inlier untuk visualisasi epipolar lines
inlier_idx = np.where(mask_fund.ravel() == 1)[0]

# Memilih beberapa titik inlier untuk visualisasi (maksimal 10)
n_vis = min(10, len(inlier_idx))
vis_idx = inlier_idx[:n_vis]

# Mengambil titik sumber dan tujuan yang dipilih
pts_vis_left = src_pts[vis_idx]
pts_vis_right = dst_pts[vis_idx]

# Menghitung garis epipolar pada gambar kanan dari titik di gambar kiri
lines_right = cv2.computeCorrespondEpilines(pts_vis_left.reshape(-1, 1, 2), 1, F)
lines_right = lines_right.reshape(-1, 3)

# Menghitung garis epipolar pada gambar kiri dari titik di gambar kanan
lines_left = cv2.computeCorrespondEpilines(pts_vis_right.reshape(-1, 1, 2), 2, F)
lines_left = lines_left.reshape(-1, 3)


def gambar_epipolar(img, lines, pts, pts_other):
    """
    Menggambar garis epipolar dan titik korespondensi pada gambar.
    """
    # Membuat salinan gambar
    img_out = img.copy()

    # Mendapatkan ukuran gambar
    h, w = img_out.shape[:2]

    # Mendefinisikan warna-warna berbeda untuk setiap garis
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
              (0, 0, 128), (128, 128, 0)]

    # Menggambar setiap garis epipolar
    for i, (line, pt) in enumerate(zip(lines, pts)):
        # Mengambil parameter garis (ax + by + c = 0)
        a, b, c = line

        # Menghitung titik awal garis (x=0)
        x0, y0 = 0, int(-c / b) if abs(b) > 1e-6 else 0

        # Menghitung titik akhir garis (x=w)
        x1, y1 = w, int(-(c + a * w) / b) if abs(b) > 1e-6 else 0

        # Mengambil warna untuk garis ini
        color = colors[i % len(colors)]

        # Menggambar garis epipolar
        cv2.line(img_out, (x0, y0), (x1, y1), color, 1)

        # Menggambar titik korespondensi
        pt_int = (int(pt[0]), int(pt[1]))
        cv2.circle(img_out, pt_int, 5, color, -1)

    return img_out

# Menggambar garis epipolar pada gambar kiri
img_epi_left = gambar_epipolar(img_left, lines_left, pts_vis_left, pts_vis_right)

# Menggambar garis epipolar pada gambar kanan
img_epi_right = gambar_epipolar(img_right, lines_right, pts_vis_right, pts_vis_left)

# Membuat figure untuk visualisasi epipolar
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Menampilkan epipolar lines pada gambar kiri
axes[0].imshow(cv2.cvtColor(img_epi_left, cv2.COLOR_BGR2RGB))

# Memberikan judul
axes[0].set_title(f"Garis Epipolar pada Gambar Kiri ({n_vis} titik)", fontsize=12)

# Menonaktifkan sumbu
axes[0].axis('off')

# Menampilkan epipolar lines pada gambar kanan
axes[1].imshow(cv2.cvtColor(img_epi_right, cv2.COLOR_BGR2RGB))

# Memberikan judul
axes[1].set_title(f"Garis Epipolar pada Gambar Kanan ({n_vis} titik)", fontsize=12)

# Menonaktifkan sumbu
axes[1].axis('off')

# Memberikan judul utama
fig.suptitle("Visualisasi Garis Epipolar (Fundamental Matrix)",
             fontsize=15, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi epipolar ke file
plt.savefig(os.path.join(OUTPUT_DIR, "14_epipolar_lines.png"), dpi=150, bbox_inches='tight')
plt.show()

# Menampilkan pesan
print(f"[SAVED] 14_epipolar_lines.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Perbandingan Homography vs Fundamental Matrix
# ============================================================

# Menampilkan header perbandingan
print(f"\n--- Perbandingan Homography vs Fundamental Matrix ---")

# Menampilkan jumlah inlier masing-masing
print(f"  Homography (RANSAC):    {n_inlier_cv} inlier dari {len(src_pts)}")
print(f"  Fundamental Matrix:     {n_inlier_fund} inlier dari {len(src_pts)}")

# Menghitung overlap inlier antara keduanya
mask_h = mask_cv.ravel()
mask_f = mask_fund.ravel()

# Menghitung inlier yang sama di kedua metode
overlap = int(np.logical_and(mask_h, mask_f).sum())

# Menghitung inlier unik homography
only_h = int(np.logical_and(mask_h, ~mask_f.astype(bool)).sum())

# Menghitung inlier unik fundamental
only_f = int(np.logical_and(~mask_h.astype(bool), mask_f).sum())

# Menampilkan analisis overlap
print(f"\n  Inlier kedua metode: {overlap}")
print(f"  Hanya Homography:   {only_h}")
print(f"  Hanya Fundamental:  {only_f}")

# Menampilkan penjelasan
print(f"\n  Homography cocok untuk scene planar (1 bidang datar).")
print(f"  Fundamental Matrix cocok untuk scene 3D umum (multi-depth).")

# ============================================================
# 8. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 14: VERIFIKASI GEOMETRI DETAIL")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan RANSAC dari nol
print("1. RANSAC bekerja dengan cara:")
print("   a. Memilih 4 sampel acak dari match")
print("   b. Menghitung homography (DLT + SVD)")
print("   c. Menghitung error reprojeksi semua titik")
print("   d. Menghitung inlier (error < threshold)")
print("   e. Mengulangi dan menyimpan model terbaik")

# Menampilkan perbandingan manual vs OpenCV
print(f"2. RANSAC manual menghasilkan {n_inlier_manual} inlier,")
print(f"   OpenCV RANSAC menghasilkan {n_inlier_cv} inlier.")

# Menampilkan penjelasan fundamental matrix
print("3. Fundamental Matrix (3x3, rank 2) menggambarkan geometri")
print("   epipolar: x'^T F x = 0 untuk setiap pasangan korespondensi.")

# Menampilkan penjelasan garis epipolar
print("4. Garis epipolar menunjukkan kemungkinan lokasi titik")
print("   korespondensi di gambar kedua (constraint 1D search).")

# Menampilkan perbandingan
print("5. Homography: untuk scene planar (8 DOF, 4 pasangan min).")
print("   Fundamental: untuk scene 3D umum (7 DOF, 7 pasangan min).")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 14_ransac_iterasi.png")
print("  - 14_ransac_manual.png")
print("  - 14_epipolar_lines.png")

# Menampilkan garis penutup
print("=" * 60)
