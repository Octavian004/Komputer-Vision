"""
==========================================================================
PERCOBAAN 17: PANORAMA LOOP CLOSURE
==========================================================================
Program ini menangani panorama 360° dimana gambar pertama dan terakhir
saling overlap (loop). Tanpa loop closure, akumulasi error pada chain
homography menyebabkan drift/misalignment saat loop kembali.

Konsep yang dipelajari:
- Loop detection: mendeteksi bahwa gambar pertama dan terakhir overlap
- Chain homography: mengalikan homography secara berurutan
- Accumulated drift: error yang terakumulasi sepanjang rantai
- Loop closure correction: distribusi error secara merata ke semua frame
- Interpolasi koreksi: menyebarkan residual secara gradual

Fungsi utama yang dipelajari:
- cv2.findHomography()      : Estimasi homography antar pasangan gambar
- np.matmul()               : Perkalian rantai matriks homography
- np.linalg.inv()           : Invers matriks untuk referensi
- cv2.warpPerspective()     : Warping panorama ke canvas besar
- cv2.addWeighted()         : Blending area overlap loop
- cv2.SIFT_create()         : Deteksi fitur SIFT
- cv2.FlannBasedMatcher()   : Pencocokan fitur cepat
- cv2.perspectiveTransform(): Transformasi titik menggunakan homography
- cv2.drawMatches()         : Visualisasi feature matches
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

# Mengimpor copy untuk deep copy objek
import copy

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
print("PERCOBAAN 17: PANORAMA LOOP CLOSURE")
print("=" * 65)


# ============================================================
# FUNGSI HELPER: Feature Detection dan Matching
# ============================================================

def deteksi_dan_match(img1, img2, label=""):
    """
    Mendeteksi fitur SIFT dan mencocokkan antar dua gambar.
    Menggunakan FLANN matcher dan Lowe's ratio test.

    Parameter:
    - img1   : Gambar pertama (BGR)
    - img2   : Gambar kedua (BGR)
    - label  : Label untuk logging

    Returns:
    - kp1, kp2     : Keypoints dari kedua gambar
    - good_matches  : List match yang lolos ratio test
    - desc1, desc2  : Deskriptor SIFT
    """
    # Mengkonversi ke grayscale karena SIFT bekerja pada intensitas
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Membuat detektor SIFT (Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create(nfeatures=2000)

    # Mendeteksi keypoints dan menghitung deskriptor 128-D
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    # Validasi deskriptor
    if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
        if label:
            print(f"    {label}: Tidak cukup fitur ({len(kp1) if kp1 else 0}, "
                  f"{len(kp2) if kp2 else 0})")
        return kp1, kp2, [], desc1, desc2

    # Membuat FLANN matcher (Fast Library for Approximate Nearest Neighbors)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Melakukan KNN matching (k=2 untuk Lowe's ratio test)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Menerapkan Lowe's ratio test
    # Match dianggap baik jika jarak terbaik < 0.75 * jarak kedua terbaik
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if label:
        print(f"    {label}: {len(kp1)} kp1, {len(kp2)} kp2, {len(good)} good matches")

    return kp1, kp2, good, desc1, desc2


def hitung_homography(img1, img2, label=""):
    """
    Menghitung matriks homography dari img1 ke img2 menggunakan RANSAC.

    Returns:
    - H        : Matriks homography 3x3
    - n_inlier : Jumlah inlier RANSAC
    - n_match  : Jumlah match yang baik
    """
    # Mendeteksi dan mencocokkan fitur
    kp1, kp2, good, _, _ = deteksi_dan_match(img1, img2, label)

    # Memerlukan minimal 10 match untuk homography yang reliable
    if len(good) < 10:
        return np.eye(3, dtype=np.float64), 0, len(good)

    # Mengekstrak koordinat titik yang cocok
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Menghitung homography menggunakan RANSAC (threshold = 5.0 piksel)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Menghitung jumlah inlier (titik yang sesuai dengan model)
    n_inlier = int(mask.ravel().sum()) if mask is not None else 0

    # Jika homography gagal, kembalikan identitas
    if H is None:
        H = np.eye(3, dtype=np.float64)

    return H, n_inlier, len(good)


# ============================================================
# LANGKAH 1: Memuat Gambar Loop Panorama
# ============================================================
print("\n[LANGKAH 1] Memuat 7 gambar loop panorama...")

# Mendefinisikan nama file gambar loop panorama (1-7)
# Gambar ini membentuk lingkaran 360° dimana gambar 7 overlap dengan gambar 1
loop_files = [f"panorama_loop_{i}.jpg" for i in range(1, 8)]

# Memuat semua gambar
images = []
for f in loop_files:
    path = os.path.join(IMAGE_DIR, f)
    img = cv2.imread(path)
    if img is not None:
        images.append(img)
        print(f"  Loaded: {f} ({img.shape[1]}x{img.shape[0]})")
    else:
        print(f"  [WARNING] Gagal memuat: {f}")

# Memvalidasi jumlah gambar minimum
n_images = len(images)
print(f"\n  Total gambar berhasil dimuat: {n_images}")

if n_images < 3:
    print("[ERROR] Memerlukan minimal 3 gambar. Jalankan download_image.py terlebih dahulu.")
    exit()

# Menyimpan grid input images
print("\n  Membuat grid gambar input...")
try:
    cols = min(n_images, 4)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        if i < n_images:
            axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Gambar {i + 1}", fontsize=11)
        axes[i].axis('off')

    plt.suptitle(f"Input: {n_images} Gambar Loop Panorama",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_input_images.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Grid input images disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat grid: {e}")


# ============================================================
# LANGKAH 2: Matching Semua Pasangan Bersebelahan + First-Last
# ============================================================
print("\n[LANGKAH 2] Mencocokkan fitur antar pasangan bersebelahan...")

# Menyimpan homography untuk semua pasangan bersebelahan
# H_pairs[i] = homography dari gambar i ke gambar i+1
H_pairs = []
match_info = []

for i in range(n_images - 1):
    print(f"\n  Pasangan {i + 1} → {i + 2}:")
    H, n_inlier, n_match = hitung_homography(images[i], images[i + 1],
                                               f"img{i + 1}→img{i + 2}")
    H_pairs.append(H)
    match_info.append({
        'pair': f"{i + 1}→{i + 2}",
        'matches': n_match,
        'inliers': n_inlier
    })
    print(f"    Matches: {n_match}, Inliers: {n_inlier}")

# ============================================================
# LANGKAH 3: Loop Detection - Matching First & Last Image
# ============================================================
print("\n[LANGKAH 3] Loop detection: matching gambar pertama dan terakhir...")

# Mencocokkan gambar pertama (index 0) dengan gambar terakhir
H_loop, n_inlier_loop, n_match_loop = hitung_homography(
    images[-1], images[0], f"img{n_images}→img1 (LOOP)"
)

print(f"  Loop match: {n_match_loop} matches, {n_inlier_loop} inliers")

# Visualisasi loop match
try:
    kp1, kp2, good_loop, _, _ = deteksi_dan_match(images[-1], images[0], "")

    if len(good_loop) > 0:
        # Menggambar feature matches antara gambar pertama dan terakhir
        match_vis = cv2.drawMatches(
            images[-1], kp1, images[0], kp2,
            good_loop[:50], None,  # Hanya tampilkan 50 match terbaik
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0)
        )
        cv2.imwrite(os.path.join(OUTPUT_DIR, "17_loop_detection_matches.jpg"), match_vis)

        # Menampilkan loop match menggunakan matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Loop Detection: Gambar {n_images} ↔ Gambar 1\n"
                     f"{len(good_loop)} matches, {n_inlier_loop} inliers",
                     fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "17_loop_detection_vis.png"),
            dpi=150,
            bbox_inches='tight'
        )
        plt.show()
        plt.close()
        print("  Visualisasi loop match disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat visualisasi loop match: {e}")

# Menentukan apakah loop terdeteksi
loop_detected = n_inlier_loop >= 10
print(f"\n  Loop terdeteksi: {'YA' if loop_detected else 'TIDAK'}")
if loop_detected:
    print(f"  Gambar {n_images} dan Gambar 1 memiliki area overlap!")


# ============================================================
# LANGKAH 4: Chain Homography TANPA Loop Closure
# ============================================================
print("\n[LANGKAH 4] Menghitung chain homography tanpa loop closure...")

# Memilih gambar referensi (tengah) untuk meminimalkan distorsi
ref_idx = n_images // 2
print(f"  Gambar referensi: Gambar {ref_idx + 1} (tengah)")

# Menghitung homography kumulatif dari setiap gambar ke referensi
# H_cumulative[i] = homography dari gambar i ke gambar referensi
H_cumulative_no_closure = [None] * n_images
H_cumulative_no_closure[ref_idx] = np.eye(3, dtype=np.float64)

# Menghitung homography ke kiri dari referensi (ref → ref-1 → ref-2 → ...)
for i in range(ref_idx - 1, -1, -1):
    # H_pairs[i] = homography dari gambar i ke gambar i+1
    # Kita butuh: gambar i ke referensi = H_pair[i] @ H_cumulative[i+1]
    # Karena H_pairs[i] mentransformasi i→i+1, maka:
    H_cumulative_no_closure[i] = H_cumulative_no_closure[i + 1] @ H_pairs[i]

# Menghitung homography ke kanan dari referensi (ref → ref+1 → ref+2 → ...)
for i in range(ref_idx + 1, n_images):
    # H_pairs[i-1] = homography dari gambar i-1 ke gambar i
    # Kita butuh: gambar i ke referensi = inv(H_pairs[i-1]) @ H_cumulative[i-1]
    H_inv = np.linalg.inv(H_pairs[i - 1])
    H_cumulative_no_closure[i] = H_cumulative_no_closure[i - 1] @ H_inv

# Menampilkan accumulated homography
print("\n  Accumulated homography (tanpa closure):")
for i, H in enumerate(H_cumulative_no_closure):
    if H is not None:
        # Menghitung translasi dan rotasi dari homography
        tx = H[0, 2]
        ty = H[1, 2]
        angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
        print(f"    Gambar {i + 1}: tx={tx:>8.1f}, ty={ty:>8.1f}, rot={angle:>6.2f}°")


# ============================================================
# LANGKAH 5: Warping dan Stitching Tanpa Loop Closure
# ============================================================
print("\n[LANGKAH 5] Melakukan warping dan stitching tanpa loop closure...")

def warp_and_stitch(images, H_list, label=""):
    """
    Melakukan warping semua gambar ke canvas tunggal menggunakan
    homography kumulatif, kemudian menggabungkannya.

    Parameter:
    - images : List gambar BGR
    - H_list : List homography kumulatif (gambar i ke referensi)
    - label  : Label untuk logging

    Returns:
    - panorama : Hasil panorama
    """
    n = len(images)

    # Menghitung batas canvas dari semua sudut gambar yang di-transformasi
    all_corners = []
    for i in range(n):
        h, w = images[i].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        if H_list[i] is not None:
            corners_t = cv2.perspectiveTransform(corners, H_list[i])
            all_corners.append(corners_t)

    if len(all_corners) == 0:
        return None

    # Menggabungkan semua sudut untuk menentukan batas canvas
    all_corners = np.concatenate(all_corners, axis=0)
    x_min = int(np.floor(all_corners[:, :, 0].min()))
    y_min = int(np.floor(all_corners[:, :, 1].min()))
    x_max = int(np.ceil(all_corners[:, :, 0].max()))
    y_max = int(np.ceil(all_corners[:, :, 1].max()))

    # Membatasi ukuran canvas untuk menghindari out of memory
    canvas_w = min(x_max - x_min, 8000)
    canvas_h = min(y_max - y_min, 4000)

    # Matriks translasi untuk menggeser koordinat negatif ke positif
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

    if label:
        print(f"  {label}: Canvas = {canvas_w}x{canvas_h}")

    # Membuat canvas kosong dan weight accumulator untuk blending
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    # Warping setiap gambar ke canvas
    for i in range(n):
        if H_list[i] is None:
            continue

        # Warping gambar i ke canvas menggunakan homography + translasi
        warped = cv2.warpPerspective(images[i], T @ H_list[i],
                                      (canvas_w, canvas_h))

        # Membuat mask untuk area yang valid (piksel > 0)
        mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float64)

        # Membuat weight menggunakan distance transform
        # Piksel di tengah gambar mendapat weight lebih tinggi
        mask_u8 = (mask * 255).astype(np.uint8)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float64)
        dist = dist / (dist.max() + 1e-10)

        # Akumulasi weighted image
        for c in range(3):
            canvas[:, :, c] += warped[:, :, c].astype(np.float64) * dist
        weight_sum += dist

    # Normalisasi untuk mendapatkan rata-rata weighted
    weight_sum = np.maximum(weight_sum, 1e-10)
    for c in range(3):
        canvas[:, :, c] /= weight_sum

    return np.clip(canvas, 0, 255).astype(np.uint8)


# Stitching tanpa loop closure
t0 = time.time()
pano_no_closure = warp_and_stitch(images, H_cumulative_no_closure, "Tanpa closure")
t_no_closure = time.time() - t0
print(f"  Waktu stitching tanpa closure: {t_no_closure:.3f} detik")

if pano_no_closure is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "17_pano_tanpa_closure.jpg"), pano_no_closure)
    print(f"  Ukuran panorama: {pano_no_closure.shape[1]}x{pano_no_closure.shape[0]}")


# ============================================================
# LANGKAH 6: Analisis Accumulated Error
# ============================================================
print("\n[LANGKAH 6] Menganalisis accumulated error pada chain homography...")

# Menghitung accumulated homography dari gambar 1 ke gambar N melalui chain
# chain_H = H_{N-1→N} @ H_{N-2→N-1} @ ... @ H_{1→2}
H_chain = np.eye(3, dtype=np.float64)
accumulated_errors = [0.0]

# Titik referensi: sudut gambar pertama
h0, w0 = images[0].shape[:2]
ref_corners = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2)

for i in range(n_images - 1):
    # Mengalikan chain homography
    H_chain = H_pairs[i] @ H_chain

    # Menghitung posisi sudut gambar pertama setelah chain transform
    corners_transformed = cv2.perspectiveTransform(ref_corners, H_chain)

    # Menghitung deviasi dari posisi awal (error akumulasi)
    deviation = np.sqrt(np.sum((corners_transformed - ref_corners) ** 2, axis=-1))
    mean_dev = np.mean(deviation)
    accumulated_errors.append(mean_dev)

print("  Accumulated error per gambar:")
for i, err in enumerate(accumulated_errors):
    print(f"    Gambar {i + 1}: error = {err:.2f} piksel")

# Jika loop terdeteksi, hitung loop residual
if loop_detected:
    # Chain homography lengkap: gambar 1 → 2 → ... → N → 1 (seharusnya = Identity)
    # H_loop mentransformasi gambar N ke gambar 1
    H_full_loop = H_loop @ H_chain

    # Residual = seberapa jauh H_full_loop dari Identity matrix
    loop_residual = H_full_loop - np.eye(3)
    residual_norm = np.linalg.norm(loop_residual)
    print(f"\n  Loop residual (norm): {residual_norm:.4f}")
    print(f"  Matriks loop residual:")
    for row in loop_residual:
        print(f"    [{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f}]")

# Membuat grafik accumulated error
try:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_vals = list(range(1, n_images + 1))
    ax.plot(x_vals, accumulated_errors, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(x_vals, accumulated_errors, alpha=0.2, color='blue')
    ax.set_xlabel("Nomor Gambar", fontsize=12)
    ax.set_ylabel("Accumulated Error (piksel)", fontsize=12)
    ax.set_title("Akumulasi Error pada Chain Homography\n(Tanpa Loop Closure)",
                  fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_vals)

    # Menandai gambar referensi
    ax.axvline(ref_idx + 1, color='red', linestyle='--',
               label=f'Referensi (Gambar {ref_idx + 1})')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_accumulated_error.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Grafik accumulated error disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat grafik: {e}")


# ============================================================
# LANGKAH 7: Implementasi Loop Closure Correction
# ============================================================
print("\n[LANGKAH 7] Mengimplementasikan loop closure correction...")

if loop_detected:
    # === METODE: Distribusi Error Merata ===
    # Ide: jika accumulate chain dari 1→N→1 memberikan residual R,
    # distribusikan koreksi secara gradual ke setiap homography

    # Langkah 1: Hitung total accumulated homography (chain + loop)
    H_total = H_full_loop  # Seharusnya = Identity jika perfect

    # Langkah 2: Dekomposisi koreksi
    # Kita akan menginterpolasi antara Identity dan inv(H_total)
    # dan mendistribusikan ke setiap frame secara proporsional

    # Koreksi total yang diperlukan
    H_correction_total = np.linalg.inv(H_total) if np.linalg.det(H_total) != 0 else np.eye(3)

    # Langkah 3: Membuat interpolated corrections
    # Setiap gambar i mendapat koreksi proporsional: (i/N) * correction
    H_cumulative_with_closure = [None] * n_images

    # Gambar referensi tetap di posisi Identity
    H_cumulative_with_closure[ref_idx] = np.eye(3, dtype=np.float64)

    # Menghitung correction factor per gambar
    print("  Menghitung koreksi per gambar...")

    # Pertama, hitung chain homography dari referensi ke setiap gambar (tanpa closure)
    H_from_ref = [None] * n_images
    H_from_ref[ref_idx] = np.eye(3, dtype=np.float64)

    for i in range(ref_idx - 1, -1, -1):
        H_from_ref[i] = H_from_ref[i + 1] @ H_pairs[i]
    for i in range(ref_idx + 1, n_images):
        H_from_ref[i] = H_from_ref[i - 1] @ np.linalg.inv(H_pairs[i - 1])

    # Menghitung jarak chain dari referensi (jumlah step)
    chain_distances = [abs(i - ref_idx) for i in range(n_images)]
    max_dist = max(chain_distances) if max(chain_distances) > 0 else 1

    # Mendistribusikan koreksi secara proporsional terhadap jarak dari referensi
    for i in range(n_images):
        # Faktor koreksi: semakin jauh dari referensi, semakin besar koreksi
        alpha = chain_distances[i] / max_dist

        # Interpolasi logaritmik matriks (aproksimasi linear sederhana)
        # H_corrected = H_original @ (Identity + alpha * (H_correction - Identity))
        correction_interp = np.eye(3) + alpha * (H_correction_total - np.eye(3))

        # Menerapkan koreksi
        H_cumulative_with_closure[i] = H_from_ref[i] @ correction_interp

    # Menampilkan accumulated homography setelah loop closure
    print("\n  Accumulated homography (dengan closure):")
    for i, H in enumerate(H_cumulative_with_closure):
        if H is not None:
            tx = H[0, 2]
            ty = H[1, 2]
            angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
            print(f"    Gambar {i + 1}: tx={tx:>8.1f}, ty={ty:>8.1f}, rot={angle:>6.2f}°")

else:
    print("  Loop tidak terdeteksi, menggunakan homography tanpa koreksi.")
    H_cumulative_with_closure = copy.deepcopy(H_cumulative_no_closure)


# ============================================================
# LANGKAH 8: Stitching dengan Loop Closure
# ============================================================
print("\n[LANGKAH 8] Melakukan stitching dengan loop closure...")

t0 = time.time()
pano_with_closure = warp_and_stitch(images, H_cumulative_with_closure, "Dengan closure")
t_with_closure = time.time() - t0
print(f"  Waktu stitching dengan closure: {t_with_closure:.3f} detik")

if pano_with_closure is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "17_pano_dengan_closure.jpg"), pano_with_closure)
    print(f"  Ukuran panorama: {pano_with_closure.shape[1]}x{pano_with_closure.shape[0]}")


# ============================================================
# LANGKAH 9: Perbandingan Tanpa vs Dengan Loop Closure
# ============================================================
print("\n[LANGKAH 9] Membandingkan hasil tanpa vs dengan loop closure...")

try:
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    # Panorama tanpa loop closure
    if pano_no_closure is not None:
        pano_nc_rgb = cv2.cvtColor(pano_no_closure, cv2.COLOR_BGR2RGB)
        axes[0].imshow(pano_nc_rgb)
        axes[0].set_title("Tanpa Loop Closure\n(Perhatikan drift/misalignment di ujung)",
                          fontsize=13)
    axes[0].axis('off')

    # Panorama dengan loop closure
    if pano_with_closure is not None:
        pano_wc_rgb = cv2.cvtColor(pano_with_closure, cv2.COLOR_BGR2RGB)
        axes[1].imshow(pano_wc_rgb)
        axes[1].set_title("Dengan Loop Closure\n(Error didistribusikan merata)",
                          fontsize=13)
    axes[1].axis('off')

    plt.suptitle("Perbandingan: Tanpa vs Dengan Loop Closure",
                  fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_perbandingan_closure.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Perbandingan panorama disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat perbandingan: {e}")


# ============================================================
# LANGKAH 10: Zoom pada Area Loop Closure
# ============================================================
print("\n[LANGKAH 10] Zoom pada area loop closure...")

try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (label, pano) in enumerate([
        ("Tanpa Closure", pano_no_closure),
        ("Dengan Closure", pano_with_closure)
    ]):
        if pano is None:
            axes[idx].set_title(f"{label}\n(Tidak tersedia)")
            axes[idx].axis('off')
            continue

        hp, wp = pano.shape[:2]

        # Zoom pada area kanan atas (area dimana loop bertemu)
        # Mengambil 1/4 dari gambar di bagian kanan
        x_start = max(0, wp * 3 // 4 - 100)
        x_end = min(wp, wp)
        y_start = 0
        y_end = min(hp, hp // 2)

        zoom_region = pano[y_start:y_end, x_start:x_end]

        if zoom_region.size > 0:
            zoom_rgb = cv2.cvtColor(zoom_region, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(zoom_rgb)
        axes[idx].set_title(f"{label}\nZoom Area Loop", fontsize=12)
        axes[idx].axis('off')

    plt.suptitle("Detail Area Loop Closure", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_zoom_loop_area.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Zoom area loop disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat zoom: {e}")


# ============================================================
# LANGKAH 11: Analisis Error Per Gambar (Setelah Closure)
# ============================================================
print("\n[LANGKAH 11] Menganalisis error per gambar setelah loop closure...")

# Menghitung error setelah correction
corrected_errors = []
if loop_detected:
    # Hitung chain baru menggunakan homography yang sudah dikoreksi
    for i in range(n_images):
        H_corr = H_cumulative_with_closure[i]
        if H_corr is not None:
            # Menghitung deviasi dari homography ideal (tanpa drift)
            # Referensi: gambar i seharusnya di posisi yang benar
            deviation = np.linalg.norm(H_corr - H_cumulative_no_closure[i])
            corrected_errors.append(deviation)
        else:
            corrected_errors.append(0)

    print("  Koreksi yang diterapkan per gambar:")
    for i, err in enumerate(corrected_errors):
        print(f"    Gambar {i + 1}: koreksi magnitude = {err:.4f}")

# Membuat grafik perbandingan error sebelum dan sesudah closure
try:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x_vals = list(range(1, n_images + 1))

    # Error sebelum closure
    ax.plot(x_vals, accumulated_errors, 'ro-', linewidth=2, markersize=8,
            label='Tanpa Closure (accumulated)')

    # Error setelah closure correction amount
    if corrected_errors:
        ax.plot(x_vals, corrected_errors, 'gs-', linewidth=2, markersize=8,
                label='Magnitude Koreksi Closure')

    ax.set_xlabel("Nomor Gambar", fontsize=12)
    ax.set_ylabel("Error / Koreksi", fontsize=12)
    ax.set_title("Analisis Error Per Gambar: Sebelum vs Sesudah Loop Closure",
                  fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_vals)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_error_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Grafik perbandingan error disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat grafik: {e}")


# ============================================================
# LANGKAH 12: Visualisasi Pipeline Komprehensif
# ============================================================
print("\n[LANGKAH 12] Membuat visualisasi pipeline komprehensif...")

try:
    fig = plt.figure(figsize=(24, 18))

    # Baris 1: Input images (max 7)
    n_show = min(n_images, 7)
    for i in range(n_show):
        ax = fig.add_subplot(4, n_show, i + 1)
        ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Img {i + 1}", fontsize=9)
        ax.axis('off')

    # Baris 2: Panorama tanpa closure
    ax_nc = fig.add_subplot(4, 1, 2)
    if pano_no_closure is not None:
        ax_nc.imshow(cv2.cvtColor(pano_no_closure, cv2.COLOR_BGR2RGB))
    ax_nc.set_title("Panorama TANPA Loop Closure (drift terlihat di ujung)",
                     fontsize=12)
    ax_nc.axis('off')

    # Baris 3: Panorama dengan closure
    ax_wc = fig.add_subplot(4, 1, 3)
    if pano_with_closure is not None:
        ax_wc.imshow(cv2.cvtColor(pano_with_closure, cv2.COLOR_BGR2RGB))
    ax_wc.set_title("Panorama DENGAN Loop Closure (error didistribusikan merata)",
                     fontsize=12)
    ax_wc.axis('off')

    # Baris 4: Grafik error
    ax_err = fig.add_subplot(4, 1, 4)
    x_vals = list(range(1, n_images + 1))
    ax_err.plot(x_vals, accumulated_errors, 'ro-', linewidth=2, markersize=8,
                label='Accumulated Error')
    if corrected_errors:
        ax_err.plot(x_vals, corrected_errors, 'gs-', linewidth=2, markersize=8,
                    label='Correction Applied')
    ax_err.set_xlabel("Nomor Gambar")
    ax_err.set_ylabel("Error (piksel)")
    ax_err.set_title("Analisis Error", fontsize=12)
    ax_err.legend()
    ax_err.grid(True, alpha=0.3)

    plt.suptitle("Pipeline Panorama Loop Closure (Percobaan 17)",
                  fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_pipeline_komprehensif.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Pipeline komprehensif disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat pipeline komprehensif: {e}")


# ============================================================
# LANGKAH 13: Match Statistics Table
# ============================================================
print("\n[LANGKAH 13] Statistik pencocokan fitur:")
print("=" * 55)
print(f"{'Pasangan':<15} {'Matches':<12} {'Inliers':<12} {'Rasio':<10}")
print("-" * 55)

for info in match_info:
    ratio = info['inliers'] / info['matches'] * 100 if info['matches'] > 0 else 0
    print(f"{info['pair']:<15} {info['matches']:<12} {info['inliers']:<12} {ratio:>6.1f}%")

# Loop match
if loop_detected:
    ratio_loop = n_inlier_loop / n_match_loop * 100 if n_match_loop > 0 else 0
    print(f"{f'{n_images}→1 (LOOP)':<15} {n_match_loop:<12} {n_inlier_loop:<12} "
          f"{ratio_loop:>6.1f}%")

print("-" * 55)

# Total statistics
total_matches = sum(info['matches'] for info in match_info)
total_inliers = sum(info['inliers'] for info in match_info)
print(f"{'TOTAL':<15} {total_matches:<12} {total_inliers:<12}")


# ============================================================
# RINGKASAN PROGRAM
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 17")
print("=" * 65)
print(f"""
Apa yang telah dipelajari:
1. Loop Detection:
   - Mencocokkan fitur antara gambar pertama dan terakhir
   - {n_match_loop} matches dan {n_inlier_loop} inliers pada loop connection
   - Loop {'terdeteksi' if loop_detected else 'tidak terdeteksi'}

2. Chain Homography:
   - Mengalikan homography secara berurutan untuk setiap pasangan
   - Error terakumulasi sepanjang rantai
   - Accumulated error mencapai {max(accumulated_errors):.2f} piksel

3. Loop Closure Correction:
   - Distribusi error secara merata ke semua frame
   - Mengurangi drift di titik loop
   - Koreksi proporsional terhadap jarak dari frame referensi

4. Referensi Frame:
   - Gambar {ref_idx + 1} dipilih sebagai referensi (tengah)
   - Meminimalkan distorsi perspektif secara keseluruhan

5. Warping dan Blending:
   - Semua gambar di-warp ke canvas tunggal
   - Distance-based weighted blending untuk transisi halus

File hasil:
- 17_input_images.png              : Grid gambar input
- 17_loop_detection_matches.jpg    : Visualisasi loop match
- 17_pano_tanpa_closure.jpg        : Panorama tanpa loop closure
- 17_pano_dengan_closure.jpg       : Panorama dengan loop closure
- 17_perbandingan_closure.png      : Perbandingan side-by-side
- 17_zoom_loop_area.png            : Detail area loop
- 17_accumulated_error.png         : Grafik accumulated error
- 17_error_comparison.png          : Perbandingan error
- 17_pipeline_komprehensif.png     : Pipeline visualisasi lengkap
""")

print("Program selesai dijalankan.")
print("=" * 65)
