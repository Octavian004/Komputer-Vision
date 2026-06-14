"""
==========================================================================
PERCOBAAN 9: SEAM FINDING
==========================================================================
Program ini memahami dan membandingkan berbagai metode seam finding
untuk menentukan garis pemisah optimal di area overlap pada panorama.
Seam finding menentukan di mana dua gambar akan "dipotong" dan
digabungkan agar perbedaan antar gambar tidak terlihat.

Konsep yang dipelajari:
- Seam finding: menentukan garis potong optimal di area overlap
- Voronoi seam: garis tengah (midline) area overlap
- Minimum difference seam: jalur dengan |I1 - I2| minimal
- Dynamic programming untuk menemukan seam optimal
- Pengaruh seam terhadap kualitas visual panorama
- Kombinasi seam finding + blending untuk hasil terbaik

Fungsi utama yang dipelajari:
- cv2.findContours()      : Menemukan kontur garis seam
- np.gradient()           : Menghitung gradien untuk cost function
- cv2.distanceTransform() : Distance transform untuk Voronoi seam
- cv2.line() / polylines(): Menggambar garis seam pada gambar
- np.argmin()             : Menemukan indeks minimum (DP seam)
- cv2.GaussianBlur()      : Smoothing untuk cost map
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan computer vision
import cv2

# Mengimpor library NumPy untuk operasi array, matriks, dan DP
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
print("PERCOBAAN 9: SEAM FINDING")
print("=" * 65)


# ============================================================
# FUNGSI HELPER: Homography dan Warping
# ============================================================

def hitung_homography(img_src, img_dst, label=""):
    """
    Menghitung homography dari img_src ke img_dst.
    Pipeline: SIFT → FLANN → ratio test → RANSAC.

    Returns:
    - H        : Matriks homography 3x3
    - n_inlier : Jumlah inlier
    """
    # Konversi ke grayscale
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    # Detektor SIFT
    sift = cv2.SIFT_create()
    kp_src, desc_src = sift.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = sift.detectAndCompute(gray_dst, None)

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

    # Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        if label:
            print(f"    {label}: Terlalu sedikit matches ({len(good)})")
        return np.eye(3, dtype=np.float64), 0

    # Titik korespondensi
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Homography RANSAC
    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return np.eye(3, dtype=np.float64), 0

    n_inlier = int(mask_h.ravel().sum()) if mask_h is not None else 0
    if label:
        print(f"    {label}: matches={len(good)}, inliers={n_inlier}")

    return H, n_inlier


# ============================================================
# LANGKAH 1: Memuat Gambar Pasangan
# ============================================================
print("\n[LANGKAH 1] Memuat gambar pasangan untuk seam finding...")

# Membaca gambar kiri dan kanan
img_left = cv2.imread(os.path.join(IMAGE_DIR, "pair_left.jpg"))
img_right = cv2.imread(os.path.join(IMAGE_DIR, "pair_right.jpg"))

if img_left is None or img_right is None:
    print("  [ERROR] pair_left.jpg atau pair_right.jpg tidak ditemukan!")
    print("  Jalankan download_image.py terlebih dahulu.")
    exit()

print(f"  pair_left.jpg:  {img_left.shape[1]}x{img_left.shape[0]} piksel")
print(f"  pair_right.jpg: {img_right.shape[1]}x{img_right.shape[0]} piksel")


# ============================================================
# LANGKAH 2: Warping dan Alignment menggunakan Homography
# ============================================================
print("\n[LANGKAH 2] Menghitung homography dan melakukan warping...")

# Menghitung homography dari gambar kanan ke kiri
H_right_to_left, n_inlier = hitung_homography(img_right, img_left,
                                                label="H(right→left)")

# Menentukan ukuran canvas
h_l, w_l = img_left.shape[:2]
h_r, w_r = img_right.shape[:2]

# Menghitung posisi sudut gambar kanan setelah warping
corners_right = np.float32([[0, 0], [w_r, 0], [w_r, h_r],
                             [0, h_r]]).reshape(-1, 1, 2)
corners_trans = cv2.perspectiveTransform(corners_right, H_right_to_left)

# Menggabungkan semua sudut
all_corners = np.concatenate([
    np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]]).reshape(-1, 1, 2),
    corners_trans
])

x_min = int(np.floor(all_corners[:, 0, 0].min()))
y_min = int(np.floor(all_corners[:, 0, 1].min()))
x_max = int(np.ceil(all_corners[:, 0, 0].max()))
y_max = int(np.ceil(all_corners[:, 0, 1].max()))

canvas_w = min(x_max - x_min, 4000)
canvas_h = min(y_max - y_min, 2000)

# Matriks translasi
H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]],
                         dtype=np.float64)

# Warping gambar kanan ke koordinat gambar kiri
warped_right = cv2.warpPerspective(img_right, H_translate @ H_right_to_left,
                                    (canvas_w, canvas_h))

# Menempatkan gambar kiri pada canvas
warped_left = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
ox = -x_min
oy = -y_min
y1 = max(0, oy)
y2 = min(canvas_h, oy + h_l)
x1 = max(0, ox)
x2 = min(canvas_w, ox + w_l)
sy1 = max(0, -oy)
sx1 = max(0, -ox)
ah = y2 - y1
aw = x2 - x1
warped_left[y1:y1 + ah, x1:x1 + aw] = img_left[sy1:sy1 + ah, sx1:sx1 + aw]

print(f"  Canvas: {canvas_w}x{canvas_h} piksel")

# Menyimpan gambar warped
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_warped_left.jpg"), warped_left)
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_warped_right.jpg"), warped_right)
print("  [OK] Gambar warped disimpan.")


# ============================================================
# LANGKAH 3: Membuat Mask Overlap Region
# ============================================================
print("\n[LANGKAH 3] Membuat mask area overlap...")

# Membuat mask non-hitam untuk kedua gambar
mask_left = (cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
mask_right = (cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

# Area overlap: piksel yang ada di kedua gambar
overlap_mask = (mask_left & mask_right).astype(np.uint8) * 255

# Area only-left dan only-right
only_left_mask = ((mask_left == 1) & (mask_right == 0)).astype(np.uint8) * 255
only_right_mask = ((mask_left == 0) & (mask_right == 1)).astype(np.uint8) * 255

# Menghitung statistik area overlap
total_overlap = np.sum(overlap_mask > 0)
total_left = np.sum(mask_left > 0)
total_right = np.sum(mask_right > 0)
overlap_pct = total_overlap / min(total_left, total_right) * 100 if min(total_left, total_right) > 0 else 0

print(f"  Area gambar kiri:  {total_left} piksel")
print(f"  Area gambar kanan: {total_right} piksel")
print(f"  Area overlap:      {total_overlap} piksel ({overlap_pct:.1f}%)")

# Menyimpan mask overlap
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_overlap_mask.jpg"), overlap_mask)
print("  [OK] Mask overlap disimpan.")


# ============================================================
# LANGKAH 4: Metode 1 - Voronoi Seam (Midline)
# ============================================================
print("\n[LANGKAH 4] Menerapkan Voronoi seam (midline)...")

t_start = time.time()

# Voronoi seam: garis pemisah berdasarkan jarak ke tepi masing-masing gambar
# Piksel diassign ke gambar yang tepinya paling dekat

# Menghitung distance transform untuk kedua mask
dist_left = cv2.distanceTransform(mask_left, cv2.DIST_L2, 5)
dist_right = cv2.distanceTransform(mask_right, cv2.DIST_L2, 5)

# Seam Voronoi: garis di mana dist_left == dist_right (dalam area overlap)
# Piksel dimasukkan ke gambar dengan distance transform lebih besar
voronoi_left_region = (dist_left >= dist_right).astype(np.uint8)
voronoi_right_region = (dist_left < dist_right).astype(np.uint8)

# Mask final Voronoi: hanya berlaku di area overlap
voronoi_seam_mask = voronoi_left_region.copy()
voronoi_seam_mask[mask_left == 0] = 0  # Harus ada gambar kiri
voronoi_seam_mask[mask_right == 0] = 1  # Jika hanya kanan, gunakan kanan? → sebaliknya

# Memperbaiki: area only-left → gunakan kiri, only-right → gunakan kanan
final_mask_voronoi = np.zeros_like(mask_left)
final_mask_voronoi[mask_left > 0] = 1  # Default: gambar kiri
# Di area overlap, gunakan Voronoi
overlap_region = (mask_left > 0) & (mask_right > 0)
final_mask_voronoi[overlap_region & (dist_right > dist_left)] = 0

# Blending menggunakan seam Voronoi (hard cut)
result_voronoi = np.where(
    final_mask_voronoi[:, :, np.newaxis] == 1,
    warped_left,
    warped_right
)

t_voronoi = time.time() - t_start

# Menemukan garis seam (batas antara region kiri dan kanan di area overlap)
seam_voronoi_line = np.zeros_like(mask_left)
# Tepi antara region kiri dan kanan
kernel = np.ones((3, 3), dtype=np.uint8)
dilated = cv2.dilate(final_mask_voronoi, kernel, iterations=1)
eroded = cv2.erode(final_mask_voronoi, kernel, iterations=1)
seam_voronoi_line = ((dilated - eroded) > 0).astype(np.uint8)
seam_voronoi_line = seam_voronoi_line & (overlap_mask > 0).astype(np.uint8)

# Visualisasi seam pada gambar
vis_voronoi = result_voronoi.copy()
vis_voronoi[seam_voronoi_line > 0] = [0, 0, 255]  # Garis merah

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_result_voronoi.jpg"), result_voronoi)
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_seam_voronoi.jpg"), vis_voronoi)
print(f"  [OK] Voronoi seam selesai ({t_voronoi:.3f} detik)")


# ============================================================
# LANGKAH 5: Metode 2 - Minimum Difference Seam (DP)
# ============================================================
print("\n[LANGKAH 5] Menerapkan minimum difference seam (Dynamic Programming)...")

t_start = time.time()

# Menghitung perbedaan absolut antara kedua gambar di area overlap
diff = cv2.absdiff(warped_left, warped_right)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Memastikan hanya area overlap yang dihitung
diff_gray[overlap_mask == 0] = 1e9  # Area non-overlap: cost sangat tinggi

# Menambahkan gradien sebagai komponen cost function
# Seam yang melewati area dengan gradien tinggi (tepi objek) kurang diinginkan
gray_left = cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY).astype(np.float64)
gray_right = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Menghitung gradien pada kedua gambar
grad_left = np.abs(np.gradient(gray_left, axis=1))
grad_right = np.abs(np.gradient(gray_right, axis=1))

# Cost function gabungan: perbedaan warna + penalti gradien
# Seam sebaiknya melewati area di mana kedua gambar mirip DAN gradien rendah
cost_map = diff_gray + 0.5 * (grad_left + grad_right)

# Smoothing cost map untuk menghindari noise
cost_map = cv2.GaussianBlur(cost_map, (5, 5), 0)

# Menemukan bounding box area overlap untuk DP
overlap_coords = np.where(overlap_mask > 0)
if len(overlap_coords[0]) > 0:
    ov_y_min = overlap_coords[0].min()
    ov_y_max = overlap_coords[0].max()
    ov_x_min = overlap_coords[1].min()
    ov_x_max = overlap_coords[1].max()

    # Ekstrak region overlap dari cost map
    cost_region = cost_map[ov_y_min:ov_y_max + 1, ov_x_min:ov_x_max + 1].copy()
    overlap_region_mask = (overlap_mask[ov_y_min:ov_y_max + 1,
                           ov_x_min:ov_x_max + 1] > 0)

    # Set area non-overlap ke cost sangat tinggi
    cost_region[~overlap_region_mask] = 1e9

    rh, rw = cost_region.shape

    # Dynamic Programming: mencari seam vertikal optimal
    # (dari atas ke bawah melalui area overlap)
    # dp[y][x] = cost minimum untuk mencapai piksel (y, x) dari baris atas
    dp = np.full((rh, rw), 1e18, dtype=np.float64)
    backtrack = np.zeros((rh, rw), dtype=np.int32)

    # Inisialisasi baris pertama
    dp[0, :] = cost_region[0, :]

    # Mengisi tabel DP dari atas ke bawah
    for y in range(1, rh):
        for x in range(rw):
            # Skip jika bukan area overlap
            if not overlap_region_mask[y, x]:
                continue

            # Cari minimum dari 3 tetangga atas (kiri-atas, atas, kanan-atas)
            best_cost = dp[y - 1, x]
            best_x = x

            if x > 0 and dp[y - 1, x - 1] < best_cost:
                best_cost = dp[y - 1, x - 1]
                best_x = x - 1

            if x < rw - 1 and dp[y - 1, x + 1] < best_cost:
                best_cost = dp[y - 1, x + 1]
                best_x = x + 1

            dp[y, x] = best_cost + cost_region[y, x]
            backtrack[y, x] = best_x

    # Backtracking: menemukan seam dari bawah ke atas
    # Mencari posisi x dengan cost minimum di baris terakhir
    last_row_valid = dp[rh - 1, :].copy()
    last_row_valid[~overlap_region_mask[rh - 1, :]] = 1e18
    seam_x = np.argmin(last_row_valid)

    # Menelusuri seam dari bawah ke atas
    seam_path = []
    for y in range(rh - 1, -1, -1):
        seam_path.append((y + ov_y_min, seam_x + ov_x_min))
        seam_x = backtrack[y, seam_x]

    seam_path.reverse()

    print(f"  Panjang seam DP: {len(seam_path)} piksel")

    # Membuat mask berdasarkan seam DP
    # Piksel di kiri seam → gambar kiri, di kanan seam → gambar kanan
    final_mask_dp = np.ones_like(mask_left)  # Default: gambar kiri
    final_mask_dp[mask_left == 0] = 0
    final_mask_dp[(mask_left == 0) & (mask_right > 0)] = 0

    # Di area overlap, gunakan seam DP
    for y, x in seam_path:
        # Semua piksel di kanan seam: gunakan gambar kanan
        final_mask_dp[y, x:] = np.where(
            (mask_right[y, x:] > 0),
            0,
            final_mask_dp[y, x:]
        )

    # Blending menggunakan seam DP (hard cut)
    result_dp = np.where(
        final_mask_dp[:, :, np.newaxis] == 1,
        warped_left,
        warped_right
    )

    # Menggambar garis seam pada gambar
    vis_dp = result_dp.copy()
    for y, x in seam_path:
        if 0 <= y < canvas_h and 0 <= x < canvas_w:
            cv2.circle(vis_dp, (x, y), 1, (0, 255, 0), -1)  # Garis hijau

else:
    # Fallback jika tidak ada area overlap
    result_dp = warped_left.copy()
    seam_path = []
    vis_dp = result_dp.copy()
    final_mask_dp = np.ones_like(mask_left)

t_dp = time.time() - t_start

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_result_dp_seam.jpg"), result_dp)
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_seam_dp.jpg"), vis_dp)
print(f"  [OK] DP seam selesai ({t_dp:.3f} detik)")


# ============================================================
# LANGKAH 6: Metode 3 - Graph Cut Seam (Konsep Sederhana)
# ============================================================
print("\n[LANGKAH 6] Menerapkan graph cut seam (konsep sederhana)...")

t_start = time.time()

# Implementasi sederhana graph cut menggunakan iterative min-cut concept
# Menggunakan pendekatan energy minimization:
# E(seam) = Σ data_cost + Σ smooth_cost

# Data cost: perbedaan warna antara kedua gambar
data_cost = diff_gray.copy()

# Smooth cost: perbedaan gradien (menghindari seam yang melompat)
smooth_cost_h = np.abs(np.gradient(diff_gray, axis=1))
smooth_cost_v = np.abs(np.gradient(diff_gray, axis=0))

# Total cost map untuk graph cut sederhana
gc_cost = data_cost + 0.3 * (smooth_cost_h + smooth_cost_v)
gc_cost = cv2.GaussianBlur(gc_cost, (7, 7), 0)

# Menggunakan pendekatan DP yang lebih halus sebagai approximasi graph cut
# Dengan cost function yang berbeda dari metode 2
if len(overlap_coords[0]) > 0:
    gc_region = gc_cost[ov_y_min:ov_y_max + 1, ov_x_min:ov_x_max + 1].copy()
    gc_region[~overlap_region_mask] = 1e9

    rh, rw = gc_region.shape

    # Multi-pass DP: forward dan backward untuk approximasi global minimum
    # Pass 1: atas ke bawah
    dp_fwd = np.full((rh, rw), 1e18, dtype=np.float64)
    bt_fwd = np.zeros((rh, rw), dtype=np.int32)
    dp_fwd[0, :] = gc_region[0, :]

    for y in range(1, rh):
        for x in range(rw):
            if not overlap_region_mask[y, x]:
                continue
            best = dp_fwd[y - 1, x]
            bx = x
            # Memperluas pencarian ke 5 tetangga untuk smoothness
            for dx in [-2, -1, 1, 2]:
                nx = x + dx
                if 0 <= nx < rw and dp_fwd[y - 1, nx] + abs(dx) * 0.5 < best:
                    best = dp_fwd[y - 1, nx] + abs(dx) * 0.5
                    bx = nx
            dp_fwd[y, x] = best + gc_region[y, x]
            bt_fwd[y, x] = bx

    # Backtrack
    last_valid = dp_fwd[rh - 1, :].copy()
    last_valid[~overlap_region_mask[rh - 1, :]] = 1e18
    gc_x = np.argmin(last_valid)

    gc_seam_path = []
    for y in range(rh - 1, -1, -1):
        gc_seam_path.append((y + ov_y_min, gc_x + ov_x_min))
        gc_x = bt_fwd[y, gc_x]

    gc_seam_path.reverse()

    print(f"  Panjang seam graph cut: {len(gc_seam_path)} piksel")

    # Membuat mask berdasarkan graph cut seam
    final_mask_gc = np.ones_like(mask_left)
    final_mask_gc[mask_left == 0] = 0
    final_mask_gc[(mask_left == 0) & (mask_right > 0)] = 0

    for y, x in gc_seam_path:
        final_mask_gc[y, x:] = np.where(
            (mask_right[y, x:] > 0),
            0,
            final_mask_gc[y, x:]
        )

    # Blending
    result_gc = np.where(
        final_mask_gc[:, :, np.newaxis] == 1,
        warped_left,
        warped_right
    )

    # Visualisasi seam
    vis_gc = result_gc.copy()
    for y, x in gc_seam_path:
        if 0 <= y < canvas_h and 0 <= x < canvas_w:
            cv2.circle(vis_gc, (x, y), 1, (255, 0, 0), -1)  # Garis biru

else:
    result_gc = warped_left.copy()
    gc_seam_path = []
    vis_gc = result_gc.copy()
    final_mask_gc = np.ones_like(mask_left)

t_gc = time.time() - t_start

# Menyimpan hasil
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_result_graphcut_seam.jpg"), result_gc)
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_seam_graphcut.jpg"), vis_gc)
print(f"  [OK] Graph cut seam selesai ({t_gc:.3f} detik)")


# ============================================================
# LANGKAH 7: Visualisasi Semua Seam Line pada Overlap
# ============================================================
print("\n[LANGKAH 7] Membuat visualisasi semua seam pada area overlap...")

# Membuat gambar overlap (rata-rata kedua gambar)
overlap_blend = np.zeros_like(warped_left, dtype=np.float64)
overlap_count = np.zeros((canvas_h, canvas_w), dtype=np.float64)

for warped in [warped_left, warped_right]:
    m = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float64)
    for c in range(3):
        overlap_blend[:, :, c] += warped[:, :, c].astype(np.float64) * m
    overlap_count += m

overlap_count[overlap_count == 0] = 1
for c in range(3):
    overlap_blend[:, :, c] /= overlap_count
overlap_vis = overlap_blend.astype(np.uint8)

# Menggambar ketiga garis seam pada gambar overlap
vis_all_seams = overlap_vis.copy()

# Seam Voronoi: merah
vis_all_seams[seam_voronoi_line > 0] = [0, 0, 255]

# Seam DP: hijau
for y, x in seam_path:
    if 0 <= y < canvas_h and 0 <= x < canvas_w:
        cv2.circle(vis_all_seams, (x, y), 1, (0, 255, 0), -1)

# Seam Graph Cut: biru
for y, x in gc_seam_path:
    if 0 <= y < canvas_h and 0 <= x < canvas_w:
        cv2.circle(vis_all_seams, (x, y), 1, (255, 0, 0), -1)

# Menyimpan visualisasi semua seam
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_all_seams_overlay.jpg"), vis_all_seams)
print("  [OK] Visualisasi semua seam disimpan.")
print("    Merah  = Voronoi seam (midline)")
print("    Hijau  = Minimum difference seam (DP)")
print("    Biru   = Graph cut seam (approx.)")


# ============================================================
# LANGKAH 8: Seam + Feather Blending
# ============================================================
print("\n[LANGKAH 8] Menerapkan seam + feather blending...")


def apply_feather_with_seam(warped_l, warped_r, seam_mask, feather_width=30):
    """
    Menerapkan feather blending di sekitar garis seam.
    Transisi gradual di sekitar seam untuk menghilangkan hard edge.

    Parameter:
    - warped_l      : Gambar kiri (warped)
    - warped_r      : Gambar kanan (warped)
    - seam_mask     : Mask (1 = kiri, 0 = kanan)
    - feather_width : Lebar area feathering (piksel)

    Returns:
    - result : Gambar blended
    """
    # Menghitung distance transform dari tepi seam mask
    # Jarak positif ke area kiri, negatif ke area kanan
    dist_to_seam = cv2.distanceTransform(
        seam_mask.astype(np.uint8), cv2.DIST_L2, 5
    )
    dist_from_seam = cv2.distanceTransform(
        (1 - seam_mask).astype(np.uint8), cv2.DIST_L2, 5
    )

    # Alpha: 1.0 di area kiri jauh dari seam, 0.0 di area kanan jauh dari seam
    # Transisi gradual di sekitar seam
    alpha = np.zeros_like(dist_to_seam)
    total_dist = dist_to_seam + dist_from_seam
    total_dist[total_dist == 0] = 1

    # Normalized distance: 1.0 = jauh ke kiri, 0.0 = jauh ke kanan
    alpha = dist_to_seam / total_dist

    # Clamp dan smooth
    alpha = np.clip(alpha, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (feather_width * 2 + 1, feather_width * 2 + 1), 0)

    # Blending
    result = np.zeros_like(warped_l, dtype=np.float64)
    mask_l = (cv2.cvtColor(warped_l, cv2.COLOR_BGR2GRAY) > 0).astype(np.float64)
    mask_r = (cv2.cvtColor(warped_r, cv2.COLOR_BGR2GRAY) > 0).astype(np.float64)

    for c in range(3):
        blended = (warped_l[:, :, c].astype(np.float64) * alpha +
                   warped_r[:, :, c].astype(np.float64) * (1 - alpha))

        # Di area hanya kiri atau hanya kanan, gunakan gambar tersebut
        result[:, :, c] = np.where(
            (mask_l > 0) & (mask_r > 0),
            blended,
            np.where(mask_l > 0,
                     warped_l[:, :, c].astype(np.float64),
                     warped_r[:, :, c].astype(np.float64))
        )

    return np.clip(result, 0, 255).astype(np.uint8)


# Menerapkan feather blending dengan seam Voronoi
result_voronoi_feather = apply_feather_with_seam(
    warped_left, warped_right, final_mask_voronoi, feather_width=20
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_result_voronoi_feather.jpg"),
            result_voronoi_feather)
print("  [OK] Voronoi + feather blending disimpan.")

# Menerapkan feather blending dengan seam DP
result_dp_feather = apply_feather_with_seam(
    warped_left, warped_right, final_mask_dp, feather_width=20
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_result_dp_feather.jpg"),
            result_dp_feather)
print("  [OK] DP + feather blending disimpan.")

# Menerapkan feather blending dengan seam graph cut
result_gc_feather = apply_feather_with_seam(
    warped_left, warped_right, final_mask_gc, feather_width=20
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_result_gc_feather.jpg"),
            result_gc_feather)
print("  [OK] Graph cut + feather blending disimpan.")


# ============================================================
# LANGKAH 9: Seam + Multi-Band Blending
# ============================================================
print("\n[LANGKAH 9] Menerapkan seam + multi-band blending...")


def multiband_blend_with_seam(img1, img2, seam_mask, levels=4):
    """
    Multi-band blending menggunakan Laplacian pyramid
    dengan seam mask sebagai panduan blending.

    Parameter:
    - img1      : Gambar kiri
    - img2      : Gambar kanan
    - seam_mask : Mask (1=kiri, 0=kanan)
    - levels    : Jumlah level pyramid

    Returns:
    - result : Gambar hasil multi-band blending
    """
    # Menyiapkan mask sebagai float
    mask_float = seam_mask.astype(np.float64)

    # Membangun Gaussian pyramid untuk mask
    gp_mask = [mask_float]
    current = mask_float.copy()
    for _ in range(levels):
        current = cv2.pyrDown(current)
        gp_mask.append(current)

    # Membangun Laplacian pyramid untuk img1
    gp1 = [img1.astype(np.float64)]
    current = img1.astype(np.float64)
    for _ in range(levels):
        current = cv2.pyrDown(current)
        gp1.append(current)

    lp1 = [gp1[levels]]
    for i in range(levels, 0, -1):
        expanded = cv2.pyrUp(gp1[i])
        # Menyesuaikan ukuran jika berbeda
        h_target, w_target = gp1[i - 1].shape[:2]
        expanded = expanded[:h_target, :w_target]
        laplacian = gp1[i - 1] - expanded
        lp1.append(laplacian)
    lp1.reverse()

    # Membangun Laplacian pyramid untuk img2
    gp2 = [img2.astype(np.float64)]
    current = img2.astype(np.float64)
    for _ in range(levels):
        current = cv2.pyrDown(current)
        gp2.append(current)

    lp2 = [gp2[levels]]
    for i in range(levels, 0, -1):
        expanded = cv2.pyrUp(gp2[i])
        h_target, w_target = gp2[i - 1].shape[:2]
        expanded = expanded[:h_target, :w_target]
        laplacian = gp2[i - 1] - expanded
        lp2.append(laplacian)
    lp2.reverse()

    # Membangun blended Laplacian pyramid
    lp_blend = []
    for i in range(levels + 1):
        m = gp_mask[i] if i < len(gp_mask) else gp_mask[-1]

        # Menyesuaikan ukuran mask
        l1 = lp1[i]
        l2 = lp2[i]

        h_l = min(l1.shape[0], l2.shape[0])
        w_l = min(l1.shape[1], l2.shape[1])
        l1 = l1[:h_l, :w_l]
        l2 = l2[:h_l, :w_l]

        m_resized = cv2.resize(m, (w_l, h_l))
        if len(m_resized.shape) == 2:
            m_resized = m_resized[:, :, np.newaxis]

        blended = l1 * m_resized + l2 * (1 - m_resized)
        lp_blend.append(blended)

    # Merekonstruksi gambar dari blended pyramid
    result = lp_blend[levels]
    for i in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result)
        h_target, w_target = lp_blend[i].shape[:2]
        result = result[:h_target, :w_target]
        result = result + lp_blend[i]

    return np.clip(result, 0, 255).astype(np.uint8)


# Menerapkan multi-band blending dengan seam DP (terbaik)
try:
    result_dp_multiband = multiband_blend_with_seam(
        warped_left, warped_right, final_mask_dp, levels=4
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "09_result_dp_multiband.jpg"),
                result_dp_multiband)
    print("  [OK] DP + multi-band blending disimpan.")
except Exception as e:
    print(f"  [WARNING] Multi-band blending gagal: {e}")
    result_dp_multiband = result_dp_feather


# ============================================================
# LANGKAH 10: Visualisasi Cost Map
# ============================================================
print("\n[LANGKAH 10] Membuat visualisasi cost map...")

# Normalisasi cost map untuk visualisasi
cost_vis = diff_gray.copy()
cost_vis[overlap_mask == 0] = 0
max_cost = cost_vis[overlap_mask > 0].max() if np.any(overlap_mask > 0) else 1
cost_vis_norm = (cost_vis / max_cost * 255).astype(np.uint8)

# Colormap untuk cost map
cost_colormap = cv2.applyColorMap(cost_vis_norm, cv2.COLORMAP_JET)

# Menyimpan cost map
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_cost_map.jpg"), cost_colormap)
print("  [OK] Cost map disimpan.")


# ============================================================
# LANGKAH 11: Membuat Grid Perbandingan Seam Methods
# ============================================================
print("\n[LANGKAH 11] Membuat grid perbandingan metode seam...")

# Grid 2x3: (hard seam) dan (seam + feather blending)
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))

# Baris atas: Hard seam (tanpa blending di seam)
axes1[0, 0].imshow(cv2.cvtColor(vis_voronoi, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title(f"Voronoi Seam (hard) - {t_voronoi:.3f}s", fontsize=11)
axes1[0, 0].axis("off")

axes1[0, 1].imshow(cv2.cvtColor(vis_dp, cv2.COLOR_BGR2RGB))
axes1[0, 1].set_title(f"DP Min-Diff Seam (hard) - {t_dp:.3f}s", fontsize=11)
axes1[0, 1].axis("off")

axes1[0, 2].imshow(cv2.cvtColor(vis_gc, cv2.COLOR_BGR2RGB))
axes1[0, 2].set_title(f"Graph Cut Seam (hard) - {t_gc:.3f}s", fontsize=11)
axes1[0, 2].axis("off")

# Baris bawah: Seam + feather blending
axes1[1, 0].imshow(cv2.cvtColor(result_voronoi_feather, cv2.COLOR_BGR2RGB))
axes1[1, 0].set_title("Voronoi + Feather Blend", fontsize=11)
axes1[1, 0].axis("off")

axes1[1, 1].imshow(cv2.cvtColor(result_dp_feather, cv2.COLOR_BGR2RGB))
axes1[1, 1].set_title("DP Min-Diff + Feather Blend", fontsize=11)
axes1[1, 1].axis("off")

axes1[1, 2].imshow(cv2.cvtColor(result_gc_feather, cv2.COLOR_BGR2RGB))
axes1[1, 2].set_title("Graph Cut + Feather Blend", fontsize=11)
axes1[1, 2].axis("off")

plt.suptitle("Percobaan 9: Perbandingan Metode Seam Finding",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_grid_seam_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid perbandingan seam disimpan.")
plt.close()


# ============================================================
# LANGKAH 12: Grid Visualisasi Seam + Cost Map
# ============================================================
print("\n[LANGKAH 12] Membuat grid visualisasi seam & cost map...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Cost map
axes2[0, 0].imshow(cv2.cvtColor(cost_colormap, cv2.COLOR_BGR2RGB))
axes2[0, 0].set_title("Cost Map (|I_left - I_right|)", fontsize=11)
axes2[0, 0].axis("off")

# Semua seam overlay
axes2[0, 1].imshow(cv2.cvtColor(vis_all_seams, cv2.COLOR_BGR2RGB))
axes2[0, 1].set_title("Semua Seam (R=Voronoi, G=DP, B=GraphCut)", fontsize=11)
axes2[0, 1].axis("off")

# DP + multi-band blending
axes2[1, 0].imshow(cv2.cvtColor(result_dp_multiband, cv2.COLOR_BGR2RGB))
axes2[1, 0].set_title("DP Seam + Multi-Band Blending (Terbaik)", fontsize=11)
axes2[1, 0].axis("off")

# Overlap mask
axes2[1, 1].imshow(overlap_mask, cmap='gray')
axes2[1, 1].set_title(f"Overlap Mask ({overlap_pct:.1f}% overlap)", fontsize=11)
axes2[1, 1].axis("off")

plt.suptitle("Percobaan 9: Cost Map dan Seam Analysis",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_grid_cost_and_seams.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid cost & seams disimpan.")
plt.close()


# ============================================================
# LANGKAH 13: Ringkasan dan Statistik
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 9: SEAM FINDING")
print("=" * 65)

# Tabel perbandingan metode seam
print("\n  Tabel Perbandingan Metode Seam Finding:")
print(f"  {'Metode':<20} | {'Waktu (s)':>10} | {'Pjg Seam':>9} | {'Konsep':<30}")
print(f"  {'-' * 20}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 30}")
print(f"  {'Voronoi':<20} | {t_voronoi:>10.3f} | "
      f"{np.sum(seam_voronoi_line > 0):>9} | Midline (equidistant)")

seam_dp_len = len(seam_path) if seam_path else 0
print(f"  {'Min-Diff DP':<20} | {t_dp:>10.3f} | "
      f"{seam_dp_len:>9} | Min |I1-I2| path (DP)")

seam_gc_len = len(gc_seam_path) if gc_seam_path else 0
print(f"  {'Graph Cut (approx)':<20} | {t_gc:>10.3f} | "
      f"{seam_gc_len:>9} | Energy minimization")

# Tabel kualitas blending
print(f"\n  Tabel Kualitas Blending:")
print(f"  {'Kombinasi':<30} | {'Catatan':<35}")
print(f"  {'-' * 30}-+-{'-' * 35}")
print(f"  {'Seam saja (hard cut)':<30} | Tepi terlihat jelas di seam")
print(f"  {'Seam + Feather':<30} | Transisi halus, sedikit blur")
print(f"  {'Seam + Multi-Band':<30} | Terbaik, detail+transisi halus")

# Penjelasan konsep
print("\n  Konsep Seam Finding:")
print("  - Voronoi: cepat, midline dari overlap (equidistant)")
print("  - Min-Diff DP: mencari jalur |I1-I2| minimal via DP")
print("  - Graph Cut: energy minimization (data + smoothness)")
print("  - Cost function = |I1-I2| + λ * gradient penalty")
print("  - Seam alone menghasilkan hard edge → perlu blending tambahan")
print("  - Combinasi seam + multi-band blending = hasil terbaik")

# Daftar output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("09_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.distanceTransform()  → Distance transform (Voronoi)")
print("    np.gradient()            → Gradien untuk cost function")
print("    np.argmin()              → Menemukan min cost (DP)")
print("    cv2.GaussianBlur()       → Smoothing cost map")
print("    cv2.pyrDown()/pyrUp()    → Pyramid untuk multi-band blend")
print("    cv2.absdiff()            → Perbedaan absolut antar gambar")
print("=" * 65)
