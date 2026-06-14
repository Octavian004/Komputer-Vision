"""
==========================================================================
PERCOBAAN 15: GAIN COMPENSATION MANUAL
==========================================================================
Program ini mengimplementasikan exposure/gain compensation secara manual
untuk panorama yang terdiri dari gambar dengan exposure berbeda-beda.
Perbedaan exposure menyebabkan seam brightness yang terlihat jelas
pada hasil stitching panorama.

Konsep yang dipelajari:
- Analisis perbedaan exposure antar gambar
- Global gain compensation (faktor skala per gambar)
- Per-channel gain compensation (equalize BGR terpisah)
- LAB color space compensation (adjust L channel saja)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Histogram matching (match ke gambar referensi)
- Evaluasi kualitas compensation (histogram overlap, std dev)

Fungsi utama yang dipelajari:
- cv2.cvtColor()    : Konversi BGR ke HSV/LAB untuk analisis brightness
- cv2.calcHist()    : Menghitung histogram per channel
- np.mean()         : Menghitung rata-rata intensitas area overlap
- cv2.createCLAHE() : Adaptive histogram equalization
- cv2.normalize()   : Normalisasi gambar
- cv2.equalizeHist(): Ekualisasi histogram global
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
print("PERCOBAAN 15: GAIN COMPENSATION MANUAL")
print("=" * 65)


# ============================================================
# FUNGSI HELPER: Homography dan Stitching
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
    kp_src, desc_src = sift.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = sift.detectAndCompute(gray_dst, None)

    # Validasi deskriptor
    if (desc_src is None or desc_dst is None or
            len(desc_src) < 4 or len(desc_dst) < 4):
        if label:
            print(f"    {label}: Tidak cukup fitur")
        return np.eye(3, dtype=np.float64), 0

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_src, desc_dst, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        if label:
            print(f"    {label}: Hanya {len(good)} matches")
        return np.eye(3, dtype=np.float64), 0

    # Homography (RANSAC)
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    n_inlier = int(mask.ravel().sum()) if mask is not None else 0

    if label:
        print(f"    {label}: {len(good)} matches, {n_inlier} inliers")

    return H, n_inlier


def stitch_gambar_list(images, label=""):
    """
    Melakukan stitching daftar gambar secara berurutan (kiri ke kanan).

    Parameter:
    - images : List gambar BGR
    - label  : Label logging

    Returns:
    - panorama : Hasil stitching
    """
    if len(images) < 2:
        return images[0].copy() if len(images) == 1 else None

    # Memulai dengan gambar kedua sebagai basis (referensi)
    result = images[1].copy()

    # Menggabungkan gambar pertama (kiri) dengan basis
    H01, _ = hitung_homography(images[0], images[1], f"{label} 0→1")

    h_r, w_r = result.shape[:2]
    h_0, w_0 = images[0].shape[:2]

    # Menghitung batas canvas
    corners0 = np.float32([[0, 0], [w_0, 0], [w_0, h_0], [0, h_0]]).reshape(-1, 1, 2)
    corners1 = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
    corners0_t = cv2.perspectiveTransform(corners0, H01)
    all_c = np.concatenate([corners0_t, corners1], axis=0)

    x_min = int(np.floor(all_c[:, :, 0].min()))
    y_min = int(np.floor(all_c[:, :, 1].min()))
    x_max = int(np.ceil(all_c[:, :, 0].max()))
    y_max = int(np.ceil(all_c[:, :, 1].max()))
    x_min, y_min = min(x_min, 0), min(y_min, 0)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

    # Warping gambar kiri
    warped = cv2.warpPerspective(images[0], T @ H01, (canvas_w, canvas_h))

    # Menempatkan gambar referensi
    ox, oy = -x_min, -y_min
    ye = min(oy + h_r, canvas_h)
    xe = min(ox + w_r, canvas_w)
    warped[oy:ye, ox:xe] = result[:ye - oy, :xe - ox]

    result = warped

    # Menggabungkan gambar-gambar berikutnya (jika ada)
    for i in range(2, len(images)):
        H_i, _ = hitung_homography(result, images[i], f"{label} res→{i}")
        # Karena result sudah besar, letakkan images[i] di atas result
        h_i, w_i = images[i].shape[:2]
        h_res, w_res = result.shape[:2]

        # Tempatkan langsung jika homography identitas
        if np.allclose(H_i, np.eye(3)):
            continue

        # Hitung canvas baru
        corners_res = np.float32([[0, 0], [w_res, 0], [w_res, h_res], [0, h_res]]).reshape(-1, 1, 2)
        corners_i = np.float32([[0, 0], [w_i, 0], [w_i, h_i], [0, h_i]]).reshape(-1, 1, 2)

        # result sudah di-warp, jadi kita perlu warp images[i] ke result
        H_inv = np.linalg.inv(H_i) if np.linalg.det(H_i) != 0 else np.eye(3)
        corners_i_t = cv2.perspectiveTransform(corners_i, H_inv)
        all_ci = np.concatenate([corners_res, corners_i_t], axis=0)

        xi_min = int(np.floor(all_ci[:, :, 0].min()))
        yi_min = int(np.floor(all_ci[:, :, 1].min()))
        xi_max = int(np.ceil(all_ci[:, :, 0].max()))
        yi_max = int(np.ceil(all_ci[:, :, 1].max()))

        # Batas aman
        cw_i = min(xi_max - min(xi_min, 0), w_res + w_i)
        ch_i = min(yi_max - min(yi_min, 0), h_res + h_i)

        # Letakkan images[i] langsung di area overlap result
        # (simplifikasi: extend canvas ke kanan)
        if result.shape[1] < w_res + w_i // 2:
            extended = np.zeros((max(h_res, h_i), w_res + w_i // 2, 3), dtype=np.uint8)
            extended[:h_res, :w_res] = result
            result = extended

        # Place pada area kanan result
        start_x = max(w_res - w_i // 3, 0)
        end_x = min(start_x + w_i, result.shape[1])
        end_y = min(h_i, result.shape[0])
        w_place = end_x - start_x
        result[:end_y, start_x:end_x] = images[i][:end_y, :w_place]

    return result


# ============================================================
# LANGKAH 1: Memuat Gambar dengan Exposure Berbeda
# ============================================================
print("\n[LANGKAH 1] Memuat gambar dengan exposure berbeda...")

# Memuat set gambar exposure (dark, normal, bright) untuk 3 bagian panorama
img_dark_1 = cv2.imread(os.path.join(IMAGE_DIR, "exposure_dark_1.jpg"))
img_normal_1 = cv2.imread(os.path.join(IMAGE_DIR, "exposure_normal_1.jpg"))
img_bright_1 = cv2.imread(os.path.join(IMAGE_DIR, "exposure_bright_1.jpg"))

img_dark_2 = cv2.imread(os.path.join(IMAGE_DIR, "exposure_dark_2.jpg"))
img_normal_2 = cv2.imread(os.path.join(IMAGE_DIR, "exposure_normal_2.jpg"))
img_bright_2 = cv2.imread(os.path.join(IMAGE_DIR, "exposure_bright_2.jpg"))

img_dark_3 = cv2.imread(os.path.join(IMAGE_DIR, "exposure_dark_3.jpg"))
img_normal_3 = cv2.imread(os.path.join(IMAGE_DIR, "exposure_normal_3.jpg"))
img_bright_3 = cv2.imread(os.path.join(IMAGE_DIR, "exposure_bright_3.jpg"))

# Memvalidasi gambar
semua_exposure = {
    "dark_1": img_dark_1, "normal_1": img_normal_1, "bright_1": img_bright_1,
    "dark_2": img_dark_2, "normal_2": img_normal_2, "bright_2": img_bright_2,
    "dark_3": img_dark_3, "normal_3": img_normal_3, "bright_3": img_bright_3
}

valid = True
for nama, img in semua_exposure.items():
    if img is None:
        print(f"  [ERROR] {nama} tidak ditemukan!")
        valid = False
    else:
        print(f"  {nama:>10}: {img.shape[1]}x{img.shape[0]}")

if not valid:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!")
    exit()

# Membuat set gambar dengan exposure campuran (simulasi real-world)
# Gambar 1 gelap, gambar 2 normal, gambar 3 terang
exposure_mix = [img_dark_1, img_normal_2, img_bright_3]
print("\n  Set exposure campuran: dark_1, normal_2, bright_3")


# ============================================================
# LANGKAH 2: Analisis Histogram Brightness per Gambar
# ============================================================
print("\n[LANGKAH 2] Menganalisis histogram brightness per gambar...")

def analisis_brightness(img, nama=""):
    """
    Menganalisis brightness gambar: mean, std, histogram channel L (LAB).

    Parameter:
    - img  : Gambar BGR
    - nama : Label

    Returns:
    - mean_intensity : Rata-rata brightness
    - std_intensity  : Standar deviasi brightness
    - hist_L         : Histogram channel L
    """
    # Mengkonversi ke LAB untuk mendapatkan channel Lightness
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]  # Channel L (brightness, 0-255)

    # Menghitung statistik brightness
    mean_intensity = np.mean(L)
    std_intensity = np.std(L)

    # Menghitung histogram channel L
    hist_L = cv2.calcHist([L], [0], None, [256], [0, 256]).ravel()

    return mean_intensity, std_intensity, hist_L


# Menganalisis setiap gambar dalam set campuran
print(f"\n  {'Gambar':<12} | {'Mean Brightness':>15} | {'Std Dev':>10}")
print(f"  {'-'*12}-+-{'-'*15}-+-{'-'*10}")

exposure_stats = []
for i, (img, nama) in enumerate(zip(exposure_mix, ["Dark (1)", "Normal (2)", "Bright (3)"])):
    mean_b, std_b, hist_b = analisis_brightness(img, nama)
    exposure_stats.append({
        "nama": nama,
        "mean": mean_b,
        "std": std_b,
        "hist": hist_b
    })
    print(f"  {nama:<12} | {mean_b:>15.2f} | {std_b:>10.2f}")

# Menghitung rasio exposure antar gambar
print(f"\n  Rasio brightness relatif (terhadap normal):")
ref_mean = exposure_stats[1]["mean"]  # Normal sebagai referensi
for stat in exposure_stats:
    ratio = stat["mean"] / ref_mean if ref_mean > 0 else 0
    print(f"    {stat['nama']:<12}: {ratio:.4f}x")


# ============================================================
# LANGKAH 3: Method 1 - Global Gain Compensation
# ============================================================
print("\n[LANGKAH 3] Method 1: Global gain compensation...")

def global_gain_compensation(images, ref_index=1):
    """
    Menerapkan kompensasi gain global: mengalikan setiap gambar dengan
    faktor agar rata-rata brightness-nya mendekati gambar referensi.

    Parameter:
    - images    : List gambar BGR
    - ref_index : Indeks gambar referensi (default: gambar tengah)

    Returns:
    - compensated : List gambar yang sudah dikompensasi
    - gains       : List faktor gain per gambar
    """
    # Menghitung mean brightness (grayscale) setiap gambar
    means = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        means.append(np.mean(gray))

    # Menghitung gain relatif terhadap referensi
    ref_mean = means[ref_index]
    gains = []
    compensated = []

    for i, img in enumerate(images):
        if means[i] > 0:
            # Faktor gain = rata-rata referensi / rata-rata gambar ini
            gain = ref_mean / means[i]
        else:
            gain = 1.0

        gains.append(gain)

        # Mengalikan gambar dengan faktor gain
        # np.clip memastikan nilai tetap dalam rentang 0-255
        comp = np.clip(img.astype(np.float64) * gain, 0, 255).astype(np.uint8)
        compensated.append(comp)

    return compensated, gains


waktu_start = time.time()
comp_global, gains_global = global_gain_compensation(exposure_mix, ref_index=1)
waktu_global = time.time() - waktu_start

# Menampilkan faktor gain
print(f"  Faktor gain per gambar:")
for i, (g, nama) in enumerate(zip(gains_global, ["Dark", "Normal", "Bright"])):
    print(f"    {nama}: gain = {g:.4f}")
print(f"  Waktu: {waktu_global*1000:.2f} ms")

# Menyimpan hasil
for i, img_c in enumerate(comp_global):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"15_global_gain_{i+1}.jpg"), img_c)
print("  [OK] Hasil global gain compensation disimpan.")


# ============================================================
# LANGKAH 4: Method 2 - Per-Channel Gain Compensation
# ============================================================
print("\n[LANGKAH 4] Method 2: Per-channel gain compensation...")

def per_channel_gain_compensation(images, ref_index=1):
    """
    Menerapkan kompensasi gain per channel BGR: setiap channel (B, G, R)
    diequalize secara terpisah terhadap gambar referensi.

    Parameter:
    - images    : List gambar BGR
    - ref_index : Indeks gambar referensi

    Returns:
    - compensated : List gambar yang sudah dikompensasi
    - gains       : List faktor gain per gambar per channel
    """
    # Menghitung mean per channel untuk setiap gambar
    channel_means = []
    for img in images:
        means = [np.mean(img[:, :, c]) for c in range(3)]
        channel_means.append(means)

    # Referensi channel means
    ref_means = channel_means[ref_index]

    gains = []
    compensated = []

    for i, img in enumerate(images):
        gain_bgr = []
        comp = np.zeros_like(img, dtype=np.float64)

        for c in range(3):
            if channel_means[i][c] > 0:
                # Faktor gain per channel
                g = ref_means[c] / channel_means[i][c]
            else:
                g = 1.0
            gain_bgr.append(g)

            # Mengalikan channel dengan gain masing-masing
            comp[:, :, c] = img[:, :, c].astype(np.float64) * g

        gains.append(gain_bgr)
        compensated.append(np.clip(comp, 0, 255).astype(np.uint8))

    return compensated, gains


waktu_start = time.time()
comp_channel, gains_channel = per_channel_gain_compensation(exposure_mix, ref_index=1)
waktu_channel = time.time() - waktu_start

# Menampilkan faktor gain per channel
print(f"  Faktor gain per channel:")
for i, (g, nama) in enumerate(zip(gains_channel, ["Dark", "Normal", "Bright"])):
    print(f"    {nama}: B={g[0]:.4f}, G={g[1]:.4f}, R={g[2]:.4f}")
print(f"  Waktu: {waktu_channel*1000:.2f} ms")

# Menyimpan hasil
for i, img_c in enumerate(comp_channel):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"15_channel_gain_{i+1}.jpg"), img_c)
print("  [OK] Hasil per-channel gain compensation disimpan.")


# ============================================================
# LANGKAH 5: Method 3 - LAB Space Compensation
# ============================================================
print("\n[LANGKAH 5] Method 3: LAB space compensation (adjust L channel)...")

def lab_compensation(images, ref_index=1):
    """
    Menerapkan kompensasi exposure pada ruang warna LAB: hanya
    menyesuaikan channel L (Lightness) agar brightness konsisten,
    tanpa mengubah warna (channel A dan B tetap).

    Parameter:
    - images    : List gambar BGR
    - ref_index : Indeks gambar referensi

    Returns:
    - compensated : List gambar yang sudah dikompensasi (BGR)
    """
    # Mengkonversi semua gambar ke LAB
    labs = [cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for img in images]

    # Menghitung mean L channel referensi
    ref_L_mean = np.mean(labs[ref_index][:, :, 0])

    compensated = []
    for i, lab in enumerate(labs):
        lab_comp = lab.copy().astype(np.float64)

        # Menghitung faktor skala untuk channel L
        L_mean = np.mean(lab[:, :, 0])
        if L_mean > 0:
            gain_L = ref_L_mean / L_mean
        else:
            gain_L = 1.0

        # Mengalikan channel L dengan gain
        lab_comp[:, :, 0] = np.clip(lab_comp[:, :, 0] * gain_L, 0, 255)

        # Mengkonversi kembali ke BGR
        lab_comp = lab_comp.astype(np.uint8)
        bgr_comp = cv2.cvtColor(lab_comp, cv2.COLOR_LAB2BGR)
        compensated.append(bgr_comp)

        print(f"    Gambar {i+1}: L_mean={L_mean:.2f} → L_gain={gain_L:.4f}")

    return compensated


waktu_start = time.time()
comp_lab = lab_compensation(exposure_mix, ref_index=1)
waktu_lab = time.time() - waktu_start

print(f"  Waktu: {waktu_lab*1000:.2f} ms")

# Menyimpan hasil
for i, img_c in enumerate(comp_lab):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"15_lab_comp_{i+1}.jpg"), img_c)
print("  [OK] Hasil LAB compensation disimpan.")


# ============================================================
# LANGKAH 6: Method 4 - CLAHE pada Setiap Gambar
# ============================================================
print("\n[LANGKAH 6] Method 4: CLAHE (Adaptive Histogram Equalization)...")

def clahe_compensation(images, clip_limit=2.0, tile_size=(8, 8)):
    """
    Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    pada setiap gambar secara independen. CLAHE menyeimbangkan kontras
    secara lokal, menghindari over-amplification noise.

    Parameter:
    - images     : List gambar BGR
    - clip_limit : Batas kontras (default 2.0)
    - tile_size  : Ukuran grid untuk adaptive equalization

    Returns:
    - compensated : List gambar yang sudah dikompensasi
    """
    # Membuat objek CLAHE dengan parameter yang ditentukan
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    compensated = []
    for i, img in enumerate(images):
        # Mengkonversi ke LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Menerapkan CLAHE hanya pada channel L (brightness)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Mengkonversi kembali ke BGR
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        compensated.append(result)

        # Menampilkan statistik
        mean_before = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        mean_after = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
        print(f"    Gambar {i+1}: mean before={mean_before:.2f}, after={mean_after:.2f}")

    return compensated


waktu_start = time.time()
comp_clahe = clahe_compensation(exposure_mix)
waktu_clahe = time.time() - waktu_start

print(f"  Waktu: {waktu_clahe*1000:.2f} ms")

# Menyimpan hasil
for i, img_c in enumerate(comp_clahe):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"15_clahe_{i+1}.jpg"), img_c)
print("  [OK] Hasil CLAHE compensation disimpan.")


# ============================================================
# LANGKAH 7: Method 5 - Histogram Matching
# ============================================================
print("\n[LANGKAH 7] Method 5: Histogram matching (match ke referensi)...")

def histogram_matching(source, reference):
    """
    Melakukan histogram matching: mengubah distribusi intensitas
    gambar sumber agar menyerupai distribusi gambar referensi.

    Parameter:
    - source    : Gambar sumber (yang akan diubah)
    - reference : Gambar referensi (target distribusi)

    Returns:
    - matched   : Gambar sumber yang sudah di-match
    """
    matched = np.zeros_like(source)

    # Melakukan matching per channel BGR secara terpisah
    for c in range(3):
        # Menghitung histogram dan CDF gambar sumber
        hist_src, _ = np.histogram(source[:, :, c].ravel(), bins=256, range=(0, 256))
        cdf_src = np.cumsum(hist_src).astype(np.float64)
        cdf_src = cdf_src / cdf_src[-1]  # Normalisasi ke [0, 1]

        # Menghitung histogram dan CDF gambar referensi
        hist_ref, _ = np.histogram(reference[:, :, c].ravel(), bins=256, range=(0, 256))
        cdf_ref = np.cumsum(hist_ref).astype(np.float64)
        cdf_ref = cdf_ref / cdf_ref[-1]  # Normalisasi ke [0, 1]

        # Membuat lookup table: untuk setiap intensitas di source,
        # cari intensitas di reference yang memiliki CDF terdekat
        lookup = np.zeros(256, dtype=np.uint8)
        for src_val in range(256):
            # Mencari nilai di reference CDF yang paling dekat
            diff = np.abs(cdf_ref - cdf_src[src_val])
            lookup[src_val] = np.argmin(diff)

        # Menerapkan lookup table pada channel saat ini
        matched[:, :, c] = lookup[source[:, :, c]]

    return matched


def histogram_matching_compensation(images, ref_index=1):
    """
    Menerapkan histogram matching pada semua gambar terhadap referensi.

    Parameter:
    - images    : List gambar BGR
    - ref_index : Indeks gambar referensi

    Returns:
    - compensated : List gambar yang sudah di-match
    """
    reference = images[ref_index]
    compensated = []

    for i, img in enumerate(images):
        if i == ref_index:
            # Gambar referensi tidak perlu diubah
            compensated.append(img.copy())
            print(f"    Gambar {i+1}: referensi (tidak diubah)")
        else:
            matched = histogram_matching(img, reference)
            compensated.append(matched)
            # Menampilkan perubahan mean
            mean_before = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            mean_after = np.mean(cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY))
            print(f"    Gambar {i+1}: mean {mean_before:.2f} → {mean_after:.2f}")

    return compensated


waktu_start = time.time()
comp_histmatch = histogram_matching_compensation(exposure_mix)
waktu_histmatch = time.time() - waktu_start

print(f"  Waktu: {waktu_histmatch*1000:.2f} ms")

# Menyimpan hasil
for i, img_c in enumerate(comp_histmatch):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"15_histmatch_{i+1}.jpg"), img_c)
print("  [OK] Hasil histogram matching disimpan.")


# ============================================================
# LANGKAH 8: Stitching - Tanpa Compensation vs Setiap Method
# ============================================================
print("\n[LANGKAH 8] Melakukan stitching tanpa dan dengan compensation...")

# Stitching tanpa compensation (gambar exposure campuran langsung)
print("\n  Stitching TANPA compensation...")
pano_tanpa = stitch_gambar_list(exposure_mix, "Tanpa comp")
if pano_tanpa is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "15_pano_tanpa_compensation.jpg"), pano_tanpa)
    print(f"  [OK] Panorama tanpa compensation: {pano_tanpa.shape[1]}x{pano_tanpa.shape[0]}")

# Stitching dengan global gain
print("\n  Stitching dengan GLOBAL GAIN...")
pano_global = stitch_gambar_list(comp_global, "Global gain")
if pano_global is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "15_pano_global_gain.jpg"), pano_global)
    print(f"  [OK] Panorama global gain: {pano_global.shape[1]}x{pano_global.shape[0]}")

# Stitching dengan per-channel gain
print("\n  Stitching dengan PER-CHANNEL GAIN...")
pano_channel = stitch_gambar_list(comp_channel, "Channel gain")
if pano_channel is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "15_pano_channel_gain.jpg"), pano_channel)
    print(f"  [OK] Panorama per-channel gain: {pano_channel.shape[1]}x{pano_channel.shape[0]}")

# Stitching dengan LAB compensation
print("\n  Stitching dengan LAB COMPENSATION...")
pano_lab = stitch_gambar_list(comp_lab, "LAB comp")
if pano_lab is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "15_pano_lab_comp.jpg"), pano_lab)
    print(f"  [OK] Panorama LAB comp: {pano_lab.shape[1]}x{pano_lab.shape[0]}")

# Stitching dengan CLAHE
print("\n  Stitching dengan CLAHE...")
pano_clahe = stitch_gambar_list(comp_clahe, "CLAHE")
if pano_clahe is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "15_pano_clahe.jpg"), pano_clahe)
    print(f"  [OK] Panorama CLAHE: {pano_clahe.shape[1]}x{pano_clahe.shape[0]}")

# Stitching dengan histogram matching
print("\n  Stitching dengan HISTOGRAM MATCHING...")
pano_histmatch = stitch_gambar_list(comp_histmatch, "HistMatch")
if pano_histmatch is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "15_pano_histmatch.jpg"), pano_histmatch)
    print(f"  [OK] Panorama histogram matching: {pano_histmatch.shape[1]}x{pano_histmatch.shape[0]}")


# ============================================================
# LANGKAH 9: Histogram Sebelum dan Sesudah Compensation
# ============================================================
print("\n[LANGKAH 9] Membuat histogram sebelum dan sesudah compensation...")

# Membuat figure histogram: 5 metode × 2 baris (before/after)
fig_hist, axes_hist = plt.subplots(2, 3, figsize=(18, 8))

# Warna untuk setiap gambar
colors_plot = ['#1565C0', '#2E7D32', '#E65100']
labels_gambar = ['Dark (1)', 'Normal (2)', 'Bright (3)']

# Baris 1: Histogram SEBELUM compensation
axes_hist[0, 0].set_title("Sebelum Compensation\n(L channel - LAB)", fontsize=11)
for i, img in enumerate(exposure_mix):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0], None, [256], [0, 256]).ravel()
    axes_hist[0, 0].plot(hist, color=colors_plot[i], label=labels_gambar[i], linewidth=1.5)
axes_hist[0, 0].legend(fontsize=8)
axes_hist[0, 0].set_xlabel("Intensitas L")
axes_hist[0, 0].set_ylabel("Jumlah Piksel")
axes_hist[0, 0].grid(True, alpha=0.3)

# Histogram sesudah Global Gain
axes_hist[0, 1].set_title("Sesudah Global Gain", fontsize=11)
for i, img in enumerate(comp_global):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0], None, [256], [0, 256]).ravel()
    axes_hist[0, 1].plot(hist, color=colors_plot[i], label=labels_gambar[i], linewidth=1.5)
axes_hist[0, 1].legend(fontsize=8)
axes_hist[0, 1].set_xlabel("Intensitas L")
axes_hist[0, 1].grid(True, alpha=0.3)

# Histogram sesudah Per-Channel
axes_hist[0, 2].set_title("Sesudah Per-Channel Gain", fontsize=11)
for i, img in enumerate(comp_channel):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0], None, [256], [0, 256]).ravel()
    axes_hist[0, 2].plot(hist, color=colors_plot[i], label=labels_gambar[i], linewidth=1.5)
axes_hist[0, 2].legend(fontsize=8)
axes_hist[0, 2].set_xlabel("Intensitas L")
axes_hist[0, 2].grid(True, alpha=0.3)

# Histogram sesudah LAB Compensation
axes_hist[1, 0].set_title("Sesudah LAB Compensation", fontsize=11)
for i, img in enumerate(comp_lab):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0], None, [256], [0, 256]).ravel()
    axes_hist[1, 0].plot(hist, color=colors_plot[i], label=labels_gambar[i], linewidth=1.5)
axes_hist[1, 0].legend(fontsize=8)
axes_hist[1, 0].set_xlabel("Intensitas L")
axes_hist[1, 0].set_ylabel("Jumlah Piksel")
axes_hist[1, 0].grid(True, alpha=0.3)

# Histogram sesudah CLAHE
axes_hist[1, 1].set_title("Sesudah CLAHE", fontsize=11)
for i, img in enumerate(comp_clahe):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0], None, [256], [0, 256]).ravel()
    axes_hist[1, 1].plot(hist, color=colors_plot[i], label=labels_gambar[i], linewidth=1.5)
axes_hist[1, 1].legend(fontsize=8)
axes_hist[1, 1].set_xlabel("Intensitas L")
axes_hist[1, 1].grid(True, alpha=0.3)

# Histogram sesudah Histogram Matching
axes_hist[1, 2].set_title("Sesudah Histogram Matching", fontsize=11)
for i, img in enumerate(comp_histmatch):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0], None, [256], [0, 256]).ravel()
    axes_hist[1, 2].plot(hist, color=colors_plot[i], label=labels_gambar[i], linewidth=1.5)
axes_hist[1, 2].legend(fontsize=8)
axes_hist[1, 2].set_xlabel("Intensitas L")
axes_hist[1, 2].grid(True, alpha=0.3)

plt.suptitle("Percobaan 15: Histogram Brightness Sebelum dan Sesudah Compensation",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "15_grid_histogram_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid histogram comparison disimpan.")
plt.close()


# ============================================================
# LANGKAH 10: Evaluasi - Standar Deviasi Brightness Antar Gambar
# ============================================================
print("\n[LANGKAH 10] Mengevaluasi konsistensi brightness antar gambar...")

def evaluasi_konsistensi(images, nama_metode=""):
    """
    Menghitung standar deviasi rata-rata brightness antar gambar.
    Semakin rendah std dev → semakin konsisten exposure → lebih baik.

    Parameter:
    - images      : List gambar BGR
    - nama_metode : Label metode

    Returns:
    - std_brightness : Standar deviasi mean brightness antar gambar
    - means          : List mean brightness per gambar
    """
    means = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        means.append(np.mean(gray))

    std_brightness = np.std(means)
    return std_brightness, means


# Mengevaluasi setiap metode
metode_eval = {
    "Tanpa Comp": exposure_mix,
    "Global Gain": comp_global,
    "Per-Channel": comp_channel,
    "LAB Comp": comp_lab,
    "CLAHE": comp_clahe,
    "Hist Match": comp_histmatch
}

eval_results = {}
print(f"\n  {'Metode':<15} | {'Std Dev':>10} | {'Mean per gambar':>30}")
print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*30}")

for nama, imgs in metode_eval.items():
    std_b, means = evaluasi_konsistensi(imgs, nama)
    eval_results[nama] = {"std": std_b, "means": means}

    means_str = ", ".join([f"{m:.1f}" for m in means])
    print(f"  {nama:<15} | {std_b:>10.2f} | {means_str:>30}")


# ============================================================
# LANGKAH 11: Membuat Grid Perbandingan Side-by-Side
# ============================================================
print("\n[LANGKAH 11] Membuat grid perbandingan visual side-by-side...")

# --- Grid 1: Input gambar vs setiap metode compensation ---
fig1, axes1 = plt.subplots(3, 6, figsize=(26, 12))

metode_imgs = {
    "Original": exposure_mix,
    "Global Gain": comp_global,
    "Per-Channel": comp_channel,
    "LAB Comp": comp_lab,
    "CLAHE": comp_clahe,
    "Hist Match": comp_histmatch
}

for col, (nama_m, imgs) in enumerate(metode_imgs.items()):
    for row in range(3):
        axes1[row, col].imshow(cv2.cvtColor(imgs[row], cv2.COLOR_BGR2RGB))
        if row == 0:
            axes1[row, col].set_title(nama_m, fontsize=10, fontweight='bold')
        axes1[row, col].axis("off")

        # Menambahkan label brightness di sudut
        mean_val = np.mean(cv2.cvtColor(imgs[row], cv2.COLOR_BGR2GRAY))
        axes1[row, col].text(5, 20, f"μ={mean_val:.1f}",
                              fontsize=8, color='white',
                              bbox=dict(facecolor='black', alpha=0.6))

# Label baris
for row, label in enumerate(["Gambar 1\n(Dark)", "Gambar 2\n(Normal)", "Gambar 3\n(Bright)"]):
    axes1[row, 0].set_ylabel(label, fontsize=10, fontweight='bold')

plt.suptitle("Percobaan 15: Perbandingan 5 Metode Gain Compensation\n"
             "(μ = rata-rata brightness)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "15_grid_comparison_all_methods.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid perbandingan semua metode disimpan.")
plt.close()

# --- Grid 2: Panorama stitching comparison ---
pano_list = [
    ("Tanpa\nCompensation", pano_tanpa),
    ("Global\nGain", pano_global),
    ("Per-Channel\nGain", pano_channel),
    ("LAB\nCompensation", pano_lab),
    ("CLAHE", pano_clahe),
    ("Histogram\nMatching", pano_histmatch)
]

# Filter hanya panorama yang berhasil dibuat
pano_valid = [(n, p) for n, p in pano_list if p is not None]

if len(pano_valid) > 0:
    fig2, axes2 = plt.subplots(len(pano_valid), 1, figsize=(18, 3 * len(pano_valid)))
    if len(pano_valid) == 1:
        axes2 = [axes2]

    for i, (nama_p, pano) in enumerate(pano_valid):
        axes2[i].imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
        axes2[i].set_title(nama_p, fontsize=11)
        axes2[i].axis("off")

    plt.suptitle("Percobaan 15: Panorama Stitching dengan Berbagai Compensation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_grid_panorama_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Grid panorama comparison disimpan.")
    plt.close()

# --- Grid 3: Bar chart evaluasi ---
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: Std Dev brightness
nama_metode_list = list(eval_results.keys())
std_list = [eval_results[n]["std"] for n in nama_metode_list]
warna_bar = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

bars = ax3a.bar(range(len(nama_metode_list)), std_list,
                tick_label=nama_metode_list,
                color=warna_bar[:len(nama_metode_list)],
                edgecolor='black', linewidth=0.5)
ax3a.set_ylabel("Std Dev Brightness")
ax3a.set_title("Konsistensi Brightness Antar Gambar\n(lebih rendah = lebih konsisten)")
ax3a.grid(True, alpha=0.3, axis='y')

# Menambahkan label
for bar, val in zip(bars, std_list):
    ax3a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
              f'{val:.2f}', ha='center', fontsize=9)

# Bar chart: Waktu eksekusi
waktu_all = [0, waktu_global * 1000, waktu_channel * 1000,
             waktu_lab * 1000, waktu_clahe * 1000, waktu_histmatch * 1000]
bars2 = ax3b.bar(range(len(nama_metode_list)), waktu_all,
                 tick_label=nama_metode_list,
                 color=warna_bar[:len(nama_metode_list)],
                 edgecolor='black', linewidth=0.5)
ax3b.set_ylabel("Waktu (ms)")
ax3b.set_title("Waktu Eksekusi per Metode")
ax3b.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars2, waktu_all):
    ax3b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
              f'{val:.1f}', ha='center', fontsize=9)

plt.suptitle("Percobaan 15: Evaluasi Metode Gain Compensation",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "15_grid_evaluasi.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid evaluasi disimpan.")
plt.close()


# ============================================================
# LANGKAH 12: Ringkasan dan Kesimpulan
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 15: GAIN COMPENSATION MANUAL")
print("=" * 65)

# Tabel ringkasan
print(f"\n  Tabel Perbandingan Metode Compensation:")
print(f"  {'Metode':<15} | {'Std Dev':>8} | {'Waktu (ms)':>10} | {'Keterangan':>25}")
print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*10}-+-{'-'*25}")

keterangan = {
    "Tanpa Comp": "Baseline (tanpa koreksi)",
    "Global Gain": "Skala global per gambar",
    "Per-Channel": "Skala per channel BGR",
    "LAB Comp": "Adjust L channel saja",
    "CLAHE": "Adaptive equalization",
    "Hist Match": "Match distribusi ke ref"
}

waktu_dict = {
    "Tanpa Comp": 0,
    "Global Gain": waktu_global * 1000,
    "Per-Channel": waktu_channel * 1000,
    "LAB Comp": waktu_lab * 1000,
    "CLAHE": waktu_clahe * 1000,
    "Hist Match": waktu_histmatch * 1000
}

for nama in nama_metode_list:
    print(f"  {nama:<15} | {eval_results[nama]['std']:>8.2f} | "
          f"{waktu_dict[nama]:>8.1f}ms | {keterangan[nama]:>25}")

# Metode terbaik (std dev terendah, selain baseline)
best_method = min(
    [(n, d) for n, d in eval_results.items() if n != "Tanpa Comp"],
    key=lambda x: x[1]["std"]
)
print(f"\n  Metode terbaik: {best_method[0]} (std dev = {best_method[1]['std']:.2f})")

# Daftar file output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("15_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

# Kesimpulan
print("\n  Kesimpulan:")
print("    - Global gain: sederhana dan cepat, cukup untuk perbedaan exposure kecil")
print("    - Per-channel gain: memperbaiki color cast, lebih akurat dari global")
print("    - LAB compensation: mempertahankan warna, hanya menyesuaikan brightness")
print("    - CLAHE: meningkatkan kontras lokal, independen antar gambar")
print("    - Histogram matching: paling kuat untuk menyelaraskan distribusi")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.cvtColor()    → Konversi BGR ke LAB/HSV")
print("    cv2.calcHist()    → Menghitung histogram")
print("    np.mean()         → Rata-rata intensitas (gain reference)")
print("    cv2.createCLAHE() → Adaptive histogram equalization")
print("    cv2.normalize()   → Normalisasi gambar")
print("    cv2.equalizeHist()→ Ekualisasi histogram global")
print("=" * 65)
