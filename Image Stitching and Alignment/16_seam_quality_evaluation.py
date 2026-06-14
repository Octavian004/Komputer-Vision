"""
==========================================================================
PERCOBAAN 16: EVALUASI KUALITAS STITCHING (PSNR, SSIM)
==========================================================================
Program ini mengimplementasikan metrik evaluasi kualitas panorama:
PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index),
dan analisis overlap region secara mendalam.

Evaluasi kualitas stitching sangat penting untuk membandingkan berbagai
metode blending dan menentukan teknik terbaik untuk skenario tertentu.

Konsep yang dipelajari:
- PSNR: mengukur rasio sinyal terhadap noise (semakin tinggi = lebih baik)
- SSIM: mengukur kemiripan struktural (range 0-1, 1 = identik)
- Difference map: visualisasi perbedaan piksel antar gambar
- Histogram analisis: distribusi error pada area overlap
- Edge alignment error: evaluasi keselarasan tepi di area seam
- Perbandingan metode blending: no blend, feather, multi-band

Fungsi utama yang dipelajari:
- cv2.PSNR()           : Menghitung Peak Signal-to-Noise Ratio
- cv2.matchTemplate()   : Template matching untuk evaluasi alignment
- np.mean() / np.std()  : Statistik dasar untuk evaluasi kualitas
- cv2.absdiff()         : Menghitung selisih absolut antar gambar
- cv2.cvtColor()        : Konversi untuk evaluasi luminance
- cv2.Canny()           : Deteksi tepi untuk edge alignment error
- cv2.calcHist()        : Histogram untuk analisis distribusi error
- cv2.GaussianBlur()    : Blur untuk komponen SSIM
- cv2.applyColorMap()   : Heatmap visualization perbedaan
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan computer vision
import cv2

# Mengimpor library NumPy untuk operasi array dan matriks numerik
import numpy as np

# Mengimpor library os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi grafik dan grid perbandingan
import matplotlib.pyplot as plt

# Mengimpor modul time untuk mengukur waktu eksekusi setiap tahap
import time

# Mengimpor math untuk perhitungan logaritma pada PSNR manual
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
print("PERCOBAAN 16: EVALUASI KUALITAS STITCHING (PSNR, SSIM)")
print("=" * 65)


# ============================================================
# FUNGSI HELPER: Homography dan Stitching Dasar
# ============================================================

def hitung_homography(img_src, img_dst, label=""):
    """
    Menghitung homography dari img_src ke img_dst menggunakan SIFT + FLANN + RANSAC.

    Parameter:
    - img_src : Gambar sumber yang akan di-warp
    - img_dst : Gambar tujuan (referensi)
    - label   : Label untuk logging

    Returns:
    - H        : Matriks homography 3x3
    - n_inlier : Jumlah inlier dari RANSAC
    - good     : List match yang lolos ratio test
    """
    # Mengkonversi kedua gambar ke grayscale untuk deteksi fitur
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    # Membuat detektor SIFT (Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create()

    # Mendeteksi keypoints dan menghitung deskriptor pada kedua gambar
    kp_src, desc_src = sift.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = sift.detectAndCompute(gray_dst, None)

    # Validasi bahwa deskriptor berhasil dihitung
    if (desc_src is None or desc_dst is None or
            len(desc_src) < 4 or len(desc_dst) < 4):
        if label:
            print(f"    {label}: Tidak cukup fitur terdeteksi")
        return np.eye(3, dtype=np.float64), 0, []

    # Membuat FLANN matcher untuk pencocokan fitur cepat
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Melakukan KNN matching (k=2 untuk ratio test)
    matches = flann.knnMatch(desc_src, desc_dst, k=2)

    # Menerapkan Lowe's ratio test untuk filter match yang baik
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Memerlukan minimal 10 match untuk homography yang reliable
    if len(good) < 10:
        if label:
            print(f"    {label}: Hanya {len(good)} matches (minimal 10)")
        return np.eye(3, dtype=np.float64), 0, good

    # Mengekstrak koordinat titik yang cocok
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Menghitung homography menggunakan RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    n_inlier = int(mask.ravel().sum()) if mask is not None else 0

    if label:
        print(f"    {label}: {len(good)} matches, {n_inlier} inliers")

    return H, n_inlier, good


def stitch_pair_simple(img_left, img_right, label="simple"):
    """
    Stitching sederhana dua gambar tanpa blending (overwrite langsung).

    Parameter:
    - img_left   : Gambar kiri
    - img_right  : Gambar kanan
    - label      : Label logging

    Returns:
    - panorama   : Hasil stitching
    - overlap_mask : Mask area overlap
    """
    # Menghitung homography dari gambar kiri ke gambar kanan
    H, n_inlier, _ = hitung_homography(img_left, img_right, label)

    # Mendapatkan dimensi kedua gambar
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    # Menghitung batas canvas menggunakan perspectiveTransform
    corners_left = np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]]).reshape(-1, 1, 2)
    corners_right = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
    corners_left_t = cv2.perspectiveTransform(corners_left, H)
    all_corners = np.concatenate([corners_left_t, corners_right], axis=0)

    # Menentukan batas canvas
    x_min = int(np.floor(all_corners[:, :, 0].min()))
    y_min = int(np.floor(all_corners[:, :, 1].min()))
    x_max = int(np.ceil(all_corners[:, :, 0].max()))
    y_max = int(np.ceil(all_corners[:, :, 1].max()))
    x_min, y_min = min(x_min, 0), min(y_min, 0)

    # Membuat matriks translasi untuk menggeser ke koordinat positif
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Membatasi ukuran canvas agar tidak terlalu besar
    canvas_w = min(canvas_w, 5000)
    canvas_h = min(canvas_h, 3000)

    # Warping gambar kiri ke canvas menggunakan homography
    warped_left = cv2.warpPerspective(img_left, T @ H, (canvas_w, canvas_h))

    # Membuat mask untuk gambar kiri yang sudah di-warp
    mask_left = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    warped_gray = cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY)
    mask_left[warped_gray > 0] = 255

    # Menempatkan gambar kanan pada canvas
    canvas = warped_left.copy()
    ox, oy = -x_min, -y_min
    ye = min(oy + h_r, canvas_h)
    xe = min(ox + w_r, canvas_w)

    # Membuat mask untuk gambar kanan
    mask_right = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_right[oy:ye, ox:xe] = 255

    # Menimpa area gambar kanan langsung (no blend)
    canvas[oy:ye, ox:xe] = img_right[:ye - oy, :xe - ox]

    # Menghitung mask overlap (area dimana kedua gambar ada)
    overlap_mask = cv2.bitwise_and(mask_left, mask_right)

    return canvas, overlap_mask, warped_left, T, H


def stitch_pair_feather(img_left, img_right, label="feather"):
    """
    Stitching dua gambar dengan feather blending (gradien transisi halus).

    Feather blending membuat transisi gradual di area overlap sehingga
    batas antar gambar tidak terlihat tajam.

    Returns:
    - panorama    : Hasil stitching dengan feather blend
    - overlap_mask: Mask area overlap
    """
    # Menghitung homography
    H, _, _ = hitung_homography(img_left, img_right, label)

    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    # Menghitung batas canvas
    corners_left = np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]]).reshape(-1, 1, 2)
    corners_right = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
    corners_left_t = cv2.perspectiveTransform(corners_left, H)
    all_c = np.concatenate([corners_left_t, corners_right], axis=0)

    x_min = int(np.floor(all_c[:, :, 0].min()))
    y_min = int(np.floor(all_c[:, :, 1].min()))
    x_max = int(np.ceil(all_c[:, :, 0].max()))
    y_max = int(np.ceil(all_c[:, :, 1].max()))
    x_min, y_min = min(x_min, 0), min(y_min, 0)

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_w = min(x_max - x_min, 5000)
    canvas_h = min(y_max - y_min, 3000)

    # Warping gambar kiri
    warped_left = cv2.warpPerspective(img_left, T @ H, (canvas_w, canvas_h))

    # Membuat mask gambar kiri
    mask_l = (cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

    # Menempatkan gambar kanan
    canvas_right = np.zeros_like(warped_left)
    ox, oy = -x_min, -y_min
    ye = min(oy + h_r, canvas_h)
    xe = min(ox + w_r, canvas_w)
    canvas_right[oy:ye, ox:xe] = img_right[:ye - oy, :xe - ox]

    # Membuat mask gambar kanan
    mask_r = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    mask_r[oy:ye, ox:xe] = 1.0

    # Menentukan area overlap
    overlap = (mask_l > 0) & (mask_r > 0)
    overlap_mask = (overlap * 255).astype(np.uint8)

    # Membuat weight map menggunakan distance transform (feather)
    # Distance transform memberikan jarak setiap piksel dari tepi mask
    dist_l = cv2.distanceTransform(mask_l.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    dist_r = cv2.distanceTransform(mask_r.astype(np.uint8) * 255, cv2.DIST_L2, 5)

    # Normalisasi weight di area overlap
    total = dist_l + dist_r + 1e-10
    weight_l = dist_l / total
    weight_r = dist_r / total

    # Menerapkan feather blending
    result = np.zeros_like(warped_left, dtype=np.float64)
    for c in range(3):
        result[:, :, c] = (warped_left[:, :, c].astype(np.float64) * weight_l +
                           canvas_right[:, :, c].astype(np.float64) * weight_r)

    return result.astype(np.uint8), overlap_mask


def stitch_pair_multiband(img_left, img_right, levels=4, label="multiband"):
    """
    Stitching dua gambar dengan multi-band blending (Laplacian pyramid).

    Multi-band blending menggunakan piramida Laplacian untuk memblend
    frekuensi rendah dan tinggi secara terpisah, menghasilkan transisi
    yang lebih natural dibandingkan feather blending.

    Returns:
    - panorama     : Hasil stitching dengan multi-band blend
    - overlap_mask : Mask area overlap
    """
    # Menghitung homography dari gambar kiri ke kanan
    H, _, _ = hitung_homography(img_left, img_right, label)

    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    # Menghitung batas canvas
    corners_left = np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]]).reshape(-1, 1, 2)
    corners_right = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
    corners_left_t = cv2.perspectiveTransform(corners_left, H)
    all_c = np.concatenate([corners_left_t, corners_right], axis=0)

    x_min = int(np.floor(all_c[:, :, 0].min()))
    y_min = int(np.floor(all_c[:, :, 1].min()))
    x_max = int(np.ceil(all_c[:, :, 0].max()))
    y_max = int(np.ceil(all_c[:, :, 1].max()))
    x_min, y_min = min(x_min, 0), min(y_min, 0)

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_w = min(x_max - x_min, 5000)
    canvas_h = min(y_max - y_min, 3000)

    # Memastikan ukuran canvas habis dibagi 2^levels untuk piramida
    factor = 2 ** levels
    canvas_w = (canvas_w // factor) * factor
    canvas_h = (canvas_h // factor) * factor

    # Warping gambar kiri
    warped_left = cv2.warpPerspective(img_left, T @ H, (canvas_w, canvas_h))

    # Menempatkan gambar kanan pada canvas
    canvas_right = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ox, oy = -x_min, -y_min
    ye = min(oy + h_r, canvas_h)
    xe = min(ox + w_r, canvas_w)
    canvas_right[oy:ye, ox:xe] = img_right[:ye - oy, :xe - ox]

    # Membuat mask untuk kedua gambar
    mask_l = (cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    mask_r = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    mask_r[oy:ye, ox:xe] = 1.0

    # Menentukan overlap mask
    overlap = (mask_l > 0) & (mask_r > 0)
    overlap_mask_out = (overlap * 255).astype(np.uint8)

    # Membuat blend mask (0 = kiri, 1 = kanan) menggunakan distance transform
    dist_l = cv2.distanceTransform((mask_l > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
    dist_r = cv2.distanceTransform((mask_r > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
    blend_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    total = dist_l + dist_r + 1e-10
    blend_mask = dist_r / total

    # Membangun Gaussian pyramid untuk blend mask
    gp_mask = [blend_mask]
    for _ in range(levels):
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    # Membangun Laplacian pyramid untuk kedua gambar
    def build_laplacian_pyramid(img, lvl):
        """Membuat piramida Laplacian dari gambar."""
        gp = [img.astype(np.float64)]
        for _ in range(lvl):
            gp.append(cv2.pyrDown(gp[-1]))
        lp = []
        for i in range(lvl):
            expanded = cv2.pyrUp(gp[i + 1],
                                 dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - expanded)
        lp.append(gp[lvl])
        return lp

    # Membuat Laplacian pyramid untuk gambar kiri dan kanan
    lp_left = build_laplacian_pyramid(warped_left, levels)
    lp_right = build_laplacian_pyramid(canvas_right, levels)

    # Menggabungkan piramida menggunakan blend mask
    lp_blend = []
    for i in range(levels + 1):
        m = gp_mask[i]
        if len(lp_left[i].shape) == 3:
            m3 = np.stack([m] * 3, axis=-1)
        else:
            m3 = m
        blended = lp_left[i] * (1 - m3) + lp_right[i] * m3
        lp_blend.append(blended)

    # Merekonstruksi gambar dari piramida yang sudah di-blend
    result = lp_blend[levels]
    for i in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(lp_blend[i].shape[1], lp_blend[i].shape[0]))
        result = result + lp_blend[i]

    # Memastikan range piksel valid [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result, overlap_mask_out


# ============================================================
# FUNGSI METRIK EVALUASI KUALITAS
# ============================================================

def hitung_psnr_manual(img1, img2):
    """
    Menghitung PSNR (Peak Signal-to-Noise Ratio) secara manual.

    PSNR = 20 * log10(MAX_I / RMSE)
    dimana RMSE = sqrt(mean((img1 - img2)^2))

    Semakin tinggi PSNR berarti semakin sedikit noise/perbedaan.
    - PSNR > 40 dB : Sangat baik (hampir identik)
    - PSNR 30-40 dB : Baik
    - PSNR 20-30 dB : Cukup
    - PSNR < 20 dB : Buruk

    Parameter:
    - img1, img2 : Dua gambar dengan ukuran yang sama

    Returns:
    - psnr_value : Nilai PSNR dalam dB
    """
    # Mengkonversi ke float64 untuk presisi perhitungan
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)

    # Menghitung MSE (Mean Squared Error)
    mse = np.mean((i1 - i2) ** 2)

    # Jika MSE = 0, gambar identik → PSNR tak hingga
    if mse == 0:
        return float('inf')

    # Nilai maksimum piksel (untuk gambar 8-bit = 255)
    MAX_I = 255.0

    # Menghitung RMSE (Root Mean Squared Error)
    rmse = math.sqrt(mse)

    # Menghitung PSNR menggunakan rumus: 20 * log10(MAX / RMSE)
    psnr = 20.0 * math.log10(MAX_I / rmse)

    return psnr


def hitung_ssim_manual(img1, img2, k1=0.01, k2=0.03, L=255, window_size=11):
    """
    Menghitung SSIM (Structural Similarity Index) secara manual.

    SSIM mengukur kemiripan struktural berdasarkan tiga komponen:
    - Luminance: perbandingan kecerahan rata-rata
    - Contrast: perbandingan standar deviasi (kontras)
    - Structure: perbandingan korelasi (struktur)

    Formula: SSIM = (2*mu1*mu2 + C1)(2*sigma12 + C2) /
                    (mu1^2 + mu2^2 + C1)(sigma1^2 + sigma2^2 + C2)

    Parameter:
    - img1, img2    : Dua gambar grayscale (atau akan dikonversi)
    - k1, k2        : Konstanta stabilisasi
    - L             : Range dinamis piksel (255 untuk 8-bit)
    - window_size   : Ukuran window Gaussian

    Returns:
    - ssim_value    : Nilai SSIM (range 0-1, 1 = identik)
    - ssim_map      : Map SSIM per piksel
    """
    # Mengkonversi ke grayscale jika gambar berwarna
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1.copy()

    if len(img2.shape) == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g2 = img2.copy()

    # Mengkonversi ke float64 untuk presisi
    g1 = g1.astype(np.float64)
    g2 = g2.astype(np.float64)

    # Menghitung konstanta stabilisasi C1 dan C2
    C1 = (k1 * L) ** 2  # C1 = (0.01 * 255)^2 ≈ 6.5
    C2 = (k2 * L) ** 2  # C2 = (0.03 * 255)^2 ≈ 58.5

    # Menghitung mean lokal menggunakan Gaussian blur
    # GaussianBlur berfungsi sebagai windowed averaging
    mu1 = cv2.GaussianBlur(g1, (window_size, window_size), 1.5)
    mu2 = cv2.GaussianBlur(g2, (window_size, window_size), 1.5)

    # Menghitung mu1^2, mu2^2, dan mu1*mu2
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Menghitung sigma (variance dan covariance lokal)
    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (window_size, window_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (window_size, window_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (window_size, window_size), 1.5) - mu1_mu2

    # Memastikan variance tidak negatif (numerical stability)
    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    # Menghitung SSIM map per piksel
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-10)

    # Menghitung SSIM rata-rata
    ssim_value = np.mean(ssim_map)

    return ssim_value, ssim_map


def hitung_metrik_komponen_ssim(img1, img2, k1=0.01, k2=0.03, L=255):
    """
    Menghitung komponen SSIM secara terpisah:
    luminance, contrast, dan structure.

    Returns:
    - luminance  : Skor perbandingan kecerahan
    - contrast   : Skor perbandingan kontras
    - structure  : Skor perbandingan struktur
    """
    # Konversi ke grayscale float
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g1 = img1.astype(np.float64)

    if len(img2.shape) == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g2 = img2.astype(np.float64)

    # Konstanta stabilisasi
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2

    # Statistik global
    mu1 = np.mean(g1)
    mu2 = np.mean(g2)
    sigma1 = np.std(g1)
    sigma2 = np.std(g2)
    sigma12 = np.mean((g1 - mu1) * (g2 - mu2))

    # Luminance comparison: l(x,y) = (2*mu1*mu2 + C1) / (mu1^2 + mu2^2 + C1)
    luminance = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)

    # Contrast comparison: c(x,y) = (2*sigma1*sigma2 + C2) / (sigma1^2 + sigma2^2 + C2)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)

    # Structure comparison: s(x,y) = (sigma12 + C3) / (sigma1*sigma2 + C3)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    return luminance, contrast, structure


def hitung_edge_alignment_error(img1, img2, overlap_mask):
    """
    Menghitung edge alignment error di area overlap menggunakan Canny.

    Membandingkan tepi (edge) dari kedua gambar di area overlap.
    Error yang rendah menunjukkan alignment yang baik.

    Parameter:
    - img1, img2      : Dua gambar yang sudah di-align
    - overlap_mask    : Mask area overlap

    Returns:
    - error           : Rata-rata perbedaan edge response
    - edge_diff_img   : Visualisasi perbedaan edge
    """
    # Konversi ke grayscale
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1.copy()

    if len(img2.shape) == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g2 = img2.copy()

    # Deteksi tepi menggunakan Canny edge detector
    # Threshold 50 (low) dan 150 (high) untuk sensitifitas sedang
    edges1 = cv2.Canny(g1, 50, 150)
    edges2 = cv2.Canny(g2, 50, 150)

    # Menghitung perbedaan edge hanya di area overlap
    if overlap_mask is not None:
        # Resize mask jika ukuran berbeda
        if overlap_mask.shape != edges1.shape:
            overlap_mask = cv2.resize(overlap_mask, (edges1.shape[1], edges1.shape[0]))
        edges1 = cv2.bitwise_and(edges1, overlap_mask)
        edges2 = cv2.bitwise_and(edges2, overlap_mask)

    # Menghitung absolute difference antara kedua edge maps
    edge_diff = cv2.absdiff(edges1, edges2)

    # Menghitung rata-rata error (persentase piksel edge yang berbeda)
    if overlap_mask is not None:
        overlap_pixels = np.sum(overlap_mask > 0)
        if overlap_pixels > 0:
            error = np.sum(edge_diff > 0) / overlap_pixels * 100
        else:
            error = 0
    else:
        error = np.mean(edge_diff > 0) * 100

    return error, edge_diff


# ============================================================
# LANGKAH 1: Memuat Gambar dan Melakukan Stitching
# ============================================================
print("\n[LANGKAH 1] Memuat gambar pasangan untuk stitching...")

# Membaca gambar kiri dan kanan
img_left = cv2.imread(os.path.join(IMAGE_DIR, "pair_left.jpg"))
img_right = cv2.imread(os.path.join(IMAGE_DIR, "pair_right.jpg"))

# Memvalidasi bahwa gambar berhasil dimuat
if img_left is None or img_right is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi gambar
print(f"  Gambar kiri  : {img_left.shape[1]}x{img_left.shape[0]} piksel")
print(f"  Gambar kanan : {img_right.shape[1]}x{img_right.shape[0]} piksel")


# ============================================================
# LANGKAH 2: Stitching dengan 3 Metode Blending Berbeda
# ============================================================
print("\n[LANGKAH 2] Melakukan stitching dengan 3 metode blending...")

# Metode 1: No blend (overwrite langsung)
print("\n  --- Metode 1: No Blend (Direct) ---")
t0 = time.time()
try:
    pano_noblend, overlap_noblend, warped_left, T_mat, H_mat = stitch_pair_simple(
        img_left, img_right, "no-blend"
    )
    t_noblend = time.time() - t0
    print(f"  Waktu: {t_noblend:.3f} detik")
    print(f"  Ukuran panorama: {pano_noblend.shape[1]}x{pano_noblend.shape[0]}")
except Exception as e:
    print(f"  [ERROR] No blend gagal: {e}")
    pano_noblend = None
    t_noblend = 0

# Metode 2: Feather blending
print("\n  --- Metode 2: Feather Blending ---")
t0 = time.time()
try:
    pano_feather, overlap_feather = stitch_pair_feather(img_left, img_right, "feather")
    t_feather = time.time() - t0
    print(f"  Waktu: {t_feather:.3f} detik")
    print(f"  Ukuran panorama: {pano_feather.shape[1]}x{pano_feather.shape[0]}")
except Exception as e:
    print(f"  [ERROR] Feather blend gagal: {e}")
    pano_feather = None
    t_feather = 0

# Metode 3: Multi-band blending
print("\n  --- Metode 3: Multi-Band Blending ---")
t0 = time.time()
try:
    pano_multiband, overlap_multiband = stitch_pair_multiband(
        img_left, img_right, levels=4, label="multiband"
    )
    t_multiband = time.time() - t0
    print(f"  Waktu: {t_multiband:.3f} detik")
    print(f"  Ukuran panorama: {pano_multiband.shape[1]}x{pano_multiband.shape[0]}")
except Exception as e:
    print(f"  [ERROR] Multi-band blend gagal: {e}")
    pano_multiband = None
    t_multiband = 0

# Menyimpan hasil stitching ke file
print("\n  Menyimpan hasil stitching...")
if pano_noblend is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "16_pano_noblend.jpg"), pano_noblend)
if pano_feather is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "16_pano_feather.jpg"), pano_feather)
if pano_multiband is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "16_pano_multiband.jpg"), pano_multiband)


# ============================================================
# LANGKAH 3: Menghitung PSNR Manual di Area Overlap
# ============================================================
print("\n[LANGKAH 3] Menghitung PSNR manual pada area overlap...")

# Mengumpulkan metrik untuk setiap metode
metrik_all = {}
panoramas = {
    "No Blend": pano_noblend,
    "Feather": pano_feather,
    "Multi-Band": pano_multiband
}
overlaps = {
    "No Blend": overlap_noblend if pano_noblend is not None else None,
    "Feather": overlap_feather if pano_feather is not None else None,
    "Multi-Band": overlap_multiband if pano_multiband is not None else None
}

# Untuk setiap metode, hitung metrik terhadap gambar kanan sebagai referensi
for nama_metode, pano in panoramas.items():
    if pano is None:
        continue

    print(f"\n  --- Metrik untuk: {nama_metode} ---")

    # Mengambil region yang sesuai dengan gambar kanan dari panorama
    h_r, w_r = img_right.shape[:2]
    h_p, w_p = pano.shape[:2]

    # Crop region kanan dari panorama (area yang seharusnya = img_right)
    # Menggunakan area yang overlap sebagai dasar perbandingan
    min_h = min(h_r, h_p)
    min_w = min(w_r, w_p)
    crop_pano = pano[:min_h, :min_w]
    crop_ref = img_right[:min_h, :min_w]

    # Menghitung PSNR manual
    psnr_manual = hitung_psnr_manual(crop_pano, crop_ref)
    print(f"    PSNR (manual)  : {psnr_manual:.2f} dB")

    # Menghitung PSNR menggunakan fungsi OpenCV bawaan untuk validasi
    psnr_cv = cv2.PSNR(crop_pano, crop_ref)
    print(f"    PSNR (OpenCV)  : {psnr_cv:.2f} dB")

    # Menghitung SSIM manual
    ssim_val, ssim_map = hitung_ssim_manual(crop_pano, crop_ref)
    print(f"    SSIM (manual)  : {ssim_val:.4f}")

    # Menghitung komponen SSIM secara terpisah
    lum, con, struc = hitung_metrik_komponen_ssim(crop_pano, crop_ref)
    print(f"    SSIM Luminance : {lum:.4f}")
    print(f"    SSIM Contrast  : {con:.4f}")
    print(f"    SSIM Structure : {struc:.4f}")

    # Menghitung MAE (Mean Absolute Error)
    mae = np.mean(cv2.absdiff(crop_pano, crop_ref))
    print(f"    MAE            : {mae:.2f}")

    # Menyimpan metrik
    metrik_all[nama_metode] = {
        'psnr_manual': psnr_manual,
        'psnr_cv': psnr_cv,
        'ssim': ssim_val,
        'ssim_luminance': lum,
        'ssim_contrast': con,
        'ssim_structure': struc,
        'mae': mae,
        'ssim_map': ssim_map
    }


# ============================================================
# LANGKAH 4: Menghitung dan Visualisasi Difference Map
# ============================================================
print("\n[LANGKAH 4] Menghitung difference map antar metode...")

# Untuk setiap panorama, hitung difference map terhadap referensi
diff_maps = {}
for nama_metode, pano in panoramas.items():
    if pano is None:
        continue

    h_r, w_r = img_right.shape[:2]
    h_p, w_p = pano.shape[:2]
    min_h = min(h_r, h_p)
    min_w = min(w_r, w_p)

    # Menghitung absolute difference antara panorama dan referensi
    diff = cv2.absdiff(pano[:min_h, :min_w], img_right[:min_h, :min_w])

    # Mengkonversi ke grayscale untuk heatmap
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Meningkatkan kontras difference map agar perbedaan lebih jelas
    diff_enhanced = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)

    # Membuat heatmap menggunakan colormap JET
    heatmap = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)

    diff_maps[nama_metode] = {
        'diff_gray': diff_gray,
        'diff_enhanced': diff_enhanced,
        'heatmap': heatmap
    }

    # Menyimpan heatmap ke file
    safe_name = nama_metode.lower().replace(" ", "_").replace("-", "")
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"16_diff_heatmap_{safe_name}.jpg"), heatmap)
    print(f"  Heatmap {nama_metode} disimpan.")

# Membuat visualisasi perbandingan difference maps
print("\n  Membuat grid perbandingan difference maps...")
try:
    names_list = [k for k in diff_maps.keys()]
    n_methods = len(names_list)

    if n_methods > 0:
        fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(2, 1)

        for idx, nama in enumerate(names_list):
            # Baris atas: difference map grayscale
            axes[0, idx].imshow(diff_maps[nama]['diff_enhanced'], cmap='gray')
            axes[0, idx].set_title(f"{nama}\nDifference Map", fontsize=11)
            axes[0, idx].axis('off')

            # Baris bawah: heatmap berwarna
            hm_rgb = cv2.cvtColor(diff_maps[nama]['heatmap'], cv2.COLOR_BGR2RGB)
            axes[1, idx].imshow(hm_rgb)
            axes[1, idx].set_title(f"{nama}\nHeatmap", fontsize=11)
            axes[1, idx].axis('off')

        plt.suptitle("Perbandingan Difference Maps Antar Metode Blending",
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "16_grid_difference_maps.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Grid difference maps disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat grid difference maps: {e}")


# ============================================================
# LANGKAH 5: Histogram Perbedaan Piksel di Area Overlap
# ============================================================
print("\n[LANGKAH 5] Menganalisis histogram perbedaan piksel...")

try:
    fig, axes = plt.subplots(1, len(diff_maps), figsize=(6 * len(diff_maps), 5))
    if len(diff_maps) == 1:
        axes = [axes]

    for idx, (nama, data) in enumerate(diff_maps.items()):
        # Menghitung histogram dari difference map
        hist_values = data['diff_gray'].ravel()

        # Memplot histogram
        axes[idx].hist(hist_values, bins=50, color='steelblue', edgecolor='black',
                       alpha=0.7, density=True)
        axes[idx].set_title(f"{nama}\nMean={np.mean(hist_values):.1f}, "
                            f"Std={np.std(hist_values):.1f}", fontsize=11)
        axes[idx].set_xlabel("Nilai Perbedaan Piksel")
        axes[idx].set_ylabel("Densitas")
        axes[idx].set_xlim(0, 255)

        # Menambahkan garis vertikal untuk mean
        axes[idx].axvline(np.mean(hist_values), color='red', linestyle='--',
                          label=f'Mean={np.mean(hist_values):.1f}')
        axes[idx].legend()

    plt.suptitle("Histogram Perbedaan Piksel di Area Overlap",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_histogram_perbedaan_piksel.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Histogram perbedaan piksel disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat histogram: {e}")


# ============================================================
# LANGKAH 6: Edge Alignment Error Analysis
# ============================================================
print("\n[LANGKAH 6] Menghitung edge alignment error...")

edge_errors = {}
for nama_metode, pano in panoramas.items():
    if pano is None:
        continue

    h_r, w_r = img_right.shape[:2]
    h_p, w_p = pano.shape[:2]
    min_h = min(h_r, h_p)
    min_w = min(w_r, w_p)

    # Membuat mask overlap sederhana (area yang valid di kedua gambar)
    mask_overlap = np.ones((min_h, min_w), dtype=np.uint8) * 255

    # Menghitung edge alignment error
    error, edge_diff = hitung_edge_alignment_error(
        pano[:min_h, :min_w], img_right[:min_h, :min_w], mask_overlap
    )

    edge_errors[nama_metode] = {
        'error': error,
        'edge_diff': edge_diff
    }

    print(f"  {nama_metode}: Edge alignment error = {error:.2f}%")

# Visualisasi edge difference
try:
    names_list = [k for k in edge_errors.keys()]
    n = len(names_list)
    if n > 0:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for idx, nama in enumerate(names_list):
            axes[idx].imshow(edge_errors[nama]['edge_diff'], cmap='hot')
            axes[idx].set_title(f"{nama}\nEdge Error: {edge_errors[nama]['error']:.2f}%",
                                fontsize=11)
            axes[idx].axis('off')

        plt.suptitle("Edge Alignment Error di Area Overlap",
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "16_edge_alignment_error.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Visualisasi edge alignment error disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat visualisasi edge error: {e}")


# ============================================================
# LANGKAH 7: SSIM Map Visualization
# ============================================================
print("\n[LANGKAH 7] Membuat visualisasi SSIM map...")

try:
    ssim_names = [k for k in metrik_all.keys() if 'ssim_map' in metrik_all[k]]
    n = len(ssim_names)
    if n > 0:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for idx, nama in enumerate(ssim_names):
            sm = metrik_all[nama]['ssim_map']
            im_plot = axes[idx].imshow(sm, cmap='RdYlGn', vmin=0, vmax=1)
            axes[idx].set_title(f"{nama}\nSSIM: {metrik_all[nama]['ssim']:.4f}",
                                fontsize=11)
            axes[idx].axis('off')
            plt.colorbar(im_plot, ax=axes[idx], fraction=0.046, pad=0.04)

        plt.suptitle("SSIM Map per Metode Blending (Hijau=Baik, Merah=Buruk)",
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "16_ssim_maps.png"), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  SSIM maps disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat SSIM maps: {e}")


# ============================================================
# LANGKAH 8: Evaluasi Outdoor vs Indoor Panorama
# ============================================================
print("\n[LANGKAH 8] Membandingkan kualitas outdoor vs indoor panorama...")

# Memuat gambar outdoor dan indoor
outdoor_files = ["panorama_outdoor_1.jpg", "panorama_outdoor_2.jpg", "panorama_outdoor_3.jpg"]
indoor_files = ["panorama_indoor_1.jpg", "panorama_indoor_2.jpg", "panorama_indoor_3.jpg"]

scene_metrics = {}

for scene_name, files in [("Outdoor", outdoor_files), ("Indoor", indoor_files)]:
    print(f"\n  --- Evaluasi Scene {scene_name} ---")

    # Memuat gambar scene
    images = []
    for f in files:
        img = cv2.imread(os.path.join(IMAGE_DIR, f))
        if img is not None:
            images.append(img)

    if len(images) < 2:
        print(f"    [WARNING] Tidak cukup gambar untuk scene {scene_name}")
        continue

    # Stitching pasangan pertama (gambar 1-2) untuk evaluasi
    try:
        pano_scene, overlap_scene = stitch_pair_feather(images[0], images[1],
                                                         f"{scene_name}")

        # Ukuran perbandingan
        h1, w1 = images[1].shape[:2]
        hp, wp = pano_scene.shape[:2]
        min_h = min(h1, hp)
        min_w = min(w1, wp)

        # Menghitung PSNR dan SSIM
        psnr_scene = hitung_psnr_manual(pano_scene[:min_h, :min_w],
                                         images[1][:min_h, :min_w])
        ssim_scene, _ = hitung_ssim_manual(pano_scene[:min_h, :min_w],
                                            images[1][:min_h, :min_w])

        scene_metrics[scene_name] = {
            'psnr': psnr_scene,
            'ssim': ssim_scene,
            'panorama': pano_scene
        }

        print(f"    PSNR: {psnr_scene:.2f} dB")
        print(f"    SSIM: {ssim_scene:.4f}")

        # Menyimpan panorama scene
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"16_pano_{scene_name.lower()}.jpg"),
                    pano_scene)
    except Exception as e:
        print(f"    [ERROR] Gagal memproses scene {scene_name}: {e}")


# ============================================================
# LANGKAH 9: Perbandingan Bar Chart Metrik
# ============================================================
print("\n[LANGKAH 9] Membuat bar chart perbandingan metrik...")

try:
    # Data untuk bar chart
    methods = list(metrik_all.keys())
    psnr_values = [metrik_all[m]['psnr_manual'] for m in methods]
    ssim_values = [metrik_all[m]['ssim'] for m in methods]
    mae_values = [metrik_all[m]['mae'] for m in methods]

    # Membuat figure dengan 3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Bar chart PSNR
    bars1 = axes[0].bar(methods, psnr_values, color=['#e74c3c', '#2ecc71', '#3498db'],
                         edgecolor='black')
    axes[0].set_title("PSNR (dB)\n(Semakin tinggi = lebih baik)", fontsize=12)
    axes[0].set_ylabel("PSNR (dB)")
    for bar, val in zip(bars1, psnr_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha='center', fontsize=10)

    # Bar chart SSIM
    bars2 = axes[1].bar(methods, ssim_values, color=['#e74c3c', '#2ecc71', '#3498db'],
                         edgecolor='black')
    axes[1].set_title("SSIM\n(Semakin tinggi = lebih baik)", fontsize=12)
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0, 1.1)
    for bar, val in zip(bars2, ssim_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.4f}", ha='center', fontsize=10)

    # Bar chart MAE
    bars3 = axes[2].bar(methods, mae_values, color=['#e74c3c', '#2ecc71', '#3498db'],
                         edgecolor='black')
    axes[2].set_title("MAE (Mean Absolute Error)\n(Semakin rendah = lebih baik)", fontsize=12)
    axes[2].set_ylabel("MAE")
    for bar, val in zip(bars3, mae_values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha='center', fontsize=10)

    plt.suptitle("Perbandingan Metrik Kualitas Antar Metode Blending",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_barchart_metrik.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Bar chart metrik disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat bar chart: {e}")


# ============================================================
# LANGKAH 10: Comprehensive Quality Dashboard
# ============================================================
print("\n[LANGKAH 10] Membuat comprehensive quality dashboard...")

try:
    fig = plt.figure(figsize=(24, 16))

    # Baris 1: Panorama results
    total_methods = len([p for p in panoramas.values() if p is not None])
    col_idx = 0

    for i, (nama, pano) in enumerate(panoramas.items()):
        if pano is None:
            continue
        ax = fig.add_subplot(4, total_methods, col_idx + 1)
        pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
        ax.imshow(pano_rgb)
        ax.set_title(f"{nama}", fontsize=10)
        ax.axis('off')
        col_idx += 1

    # Baris 2: Difference heatmaps
    col_idx = 0
    for i, (nama, data) in enumerate(diff_maps.items()):
        ax = fig.add_subplot(4, total_methods, total_methods + col_idx + 1)
        hm_rgb = cv2.cvtColor(data['heatmap'], cv2.COLOR_BGR2RGB)
        ax.imshow(hm_rgb)
        ax.set_title(f"{nama} - Heatmap", fontsize=10)
        ax.axis('off')
        col_idx += 1

    # Baris 3: SSIM maps
    col_idx = 0
    for nama in metrik_all:
        if 'ssim_map' in metrik_all[nama]:
            ax = fig.add_subplot(4, total_methods, 2 * total_methods + col_idx + 1)
            ax.imshow(metrik_all[nama]['ssim_map'], cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title(f"{nama} - SSIM Map\n({metrik_all[nama]['ssim']:.4f})",
                          fontsize=10)
            ax.axis('off')
            col_idx += 1

    # Baris 4: Edge error maps
    col_idx = 0
    for nama, data in edge_errors.items():
        ax = fig.add_subplot(4, total_methods, 3 * total_methods + col_idx + 1)
        ax.imshow(data['edge_diff'], cmap='hot')
        ax.set_title(f"{nama} - Edge Error\n({data['error']:.2f}%)", fontsize=10)
        ax.axis('off')
        col_idx += 1

    plt.suptitle("Dashboard Evaluasi Kualitas Stitching Lengkap",
                  fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_quality_dashboard.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Quality dashboard disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat dashboard: {e}")


# ============================================================
# LANGKAH 11: Tabel Metrik Lengkap
# ============================================================
print("\n[LANGKAH 11] Tabel Metrik Kualitas Lengkap")
print("=" * 75)
print(f"{'Metode':<15} {'PSNR(dB)':<12} {'SSIM':<10} {'MAE':<10} "
      f"{'Luminance':<12} {'Contrast':<12} {'Structure':<12}")
print("-" * 75)

for nama, m in metrik_all.items():
    print(f"{nama:<15} {m['psnr_manual']:>8.2f}    {m['ssim']:>8.4f}  "
          f"{m['mae']:>8.2f}  {m['ssim_luminance']:>8.4f}    "
          f"{m['ssim_contrast']:>8.4f}    {m['ssim_structure']:>8.4f}")

print("-" * 75)

# Menampilkan edge alignment error
print(f"\n{'Metode':<15} {'Edge Error (%)':<15}")
print("-" * 30)
for nama, data in edge_errors.items():
    print(f"{nama:<15} {data['error']:>10.2f}%")

# Menampilkan scene comparison
if scene_metrics:
    print(f"\n{'Scene':<15} {'PSNR(dB)':<12} {'SSIM':<10}")
    print("-" * 35)
    for scene, m in scene_metrics.items():
        print(f"{scene:<15} {m['psnr']:>8.2f}    {m['ssim']:>8.4f}")

# Menentukan metode terbaik
if metrik_all:
    best_psnr = max(metrik_all, key=lambda x: metrik_all[x]['psnr_manual'])
    best_ssim = max(metrik_all, key=lambda x: metrik_all[x]['ssim'])
    best_mae = min(metrik_all, key=lambda x: metrik_all[x]['mae'])

    print(f"\n  Metode terbaik berdasarkan PSNR : {best_psnr}")
    print(f"  Metode terbaik berdasarkan SSIM : {best_ssim}")
    print(f"  Metode terbaik berdasarkan MAE  : {best_mae}")


# ============================================================
# LANGKAH 12: Template Matching untuk Evaluasi Alignment
# ============================================================
print("\n[LANGKAH 12] Evaluasi alignment menggunakan template matching...")

try:
    # Mengambil sebuah patch dari gambar kanan sebagai template
    h_r, w_r = img_right.shape[:2]
    patch_size = min(100, h_r // 3, w_r // 3)
    cy, cx = h_r // 2, w_r // 3
    template = img_right[cy:cy + patch_size, cx:cx + patch_size]

    print(f"  Template ukuran: {patch_size}x{patch_size} piksel")
    print(f"  Posisi template pada gambar kanan: ({cx}, {cy})")

    # Template matching pada setiap panorama
    for nama, pano in panoramas.items():
        if pano is None:
            continue

        # Menjalankan template matching dengan metode TM_CCOEFF_NORMED
        result = cv2.matchTemplate(pano, template, cv2.TM_CCOEFF_NORMED)

        # Menemukan lokasi match terbaik
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        print(f"  {nama}: Match confidence = {max_val:.4f}, Lokasi = {max_loc}")

        # Menggambar rectangle pada lokasi match
        marked = pano.copy()
        cv2.rectangle(marked, max_loc,
                      (max_loc[0] + patch_size, max_loc[1] + patch_size),
                      (0, 255, 0), 3)

        safe_name = nama.lower().replace(" ", "_").replace("-", "")
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"16_template_match_{safe_name}.jpg"), marked)

    print("  Template matching visualisasi disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal melakukan template matching: {e}")


# ============================================================
# LANGKAH 13: Membuat Visualisasi Perbandingan Akhir
# ============================================================
print("\n[LANGKAH 13] Membuat visualisasi perbandingan akhir...")

try:
    valid_panos = {k: v for k, v in panoramas.items() if v is not None}
    n = len(valid_panos)

    if n > 0:
        fig, axes = plt.subplots(2, n, figsize=(7 * n, 10))
        if n == 1:
            axes = axes.reshape(2, 1)

        for idx, (nama, pano) in enumerate(valid_panos.items()):
            # Baris atas: panorama
            pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
            axes[0, idx].imshow(pano_rgb)
            if nama in metrik_all:
                m = metrik_all[nama]
                axes[0, idx].set_title(
                    f"{nama}\nPSNR={m['psnr_manual']:.1f}dB | SSIM={m['ssim']:.4f}",
                    fontsize=11
                )
            else:
                axes[0, idx].set_title(nama, fontsize=11)
            axes[0, idx].axis('off')

            # Baris bawah: zoom pada seam area (crop tengah)
            hp, wp = pano.shape[:2]
            cx_pano = wp // 2
            cy_pano = hp // 2
            crop_rad = min(hp // 4, wp // 6, 150)
            zoom = pano[cy_pano - crop_rad:cy_pano + crop_rad,
                        cx_pano - crop_rad:cx_pano + crop_rad]
            if zoom.size > 0:
                zoom_rgb = cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB)
                axes[1, idx].imshow(zoom_rgb)
            axes[1, idx].set_title(f"{nama}\nZoom Area Seam", fontsize=11)
            axes[1, idx].axis('off')

        plt.suptitle("Perbandingan Hasil Stitching dan Zoom Area Seam",
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "16_perbandingan_akhir.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Visualisasi perbandingan akhir disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat visualisasi perbandingan: {e}")


# ============================================================
# RINGKASAN PROGRAM
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 16")
print("=" * 65)
print("""
Apa yang telah dipelajari:
1. PSNR (Peak Signal-to-Noise Ratio):
   - Mengukur rasio sinyal terhadap noise dalam dB
   - PSNR tinggi = kualitas lebih baik
   - Implementasi manual: 20 * log10(MAX / RMSE)

2. SSIM (Structural Similarity Index):
   - Mengukur kemiripan struktural (luminance, contrast, structure)
   - Range 0-1, dimana 1 = identik
   - Lebih sesuai dengan persepsi manusia dibanding PSNR

3. Difference Map & Heatmap:
   - Visualisasi perbedaan piksel antar gambar
   - Heatmap mempermudah identifikasi area problematik

4. Edge Alignment Error:
   - Menggunakan Canny edge detection di area overlap
   - Mengukur seberapa baik tepi gambar selaras

5. Perbandingan Metode Blending:
   - No blend: cepat tapi terlihat seam
   - Feather: transisi halus, cukup baik
   - Multi-band: terbaik untuk natural blending

6. Template Matching untuk validasi alignment:
   - cv2.matchTemplate() untuk mengukur confidence alignment

File output disimpan di folder: output/
""")

print("Program selesai dijalankan.")
print("=" * 65)
