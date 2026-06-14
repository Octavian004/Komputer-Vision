"""
==========================================================================
PERCOBAAN 4: MULTI-IMAGE PANORAMA
==========================================================================
Program ini membangun panorama lebar dari 5 gambar menggunakan chain
homography. Gambar tengah (ke-3) dijadikan sebagai referensi untuk
meminimalkan distorsi. Homography antar pasangan dihitung lalu di-chain
untuk memetakan semua gambar ke koordinat referensi.

Konsep yang dipelajari:
- Chain homography: menggabungkan transformasi berurutan
- Pemilihan gambar referensi (tengah) untuk meminimalkan distorsi
- Warping multi-gambar ke satu canvas besar
- Perbandingan referensi kiri vs tengah vs kanan
- Analisis distorsi pada panorama lebar

Fungsi utama yang dipelajari:
- cv2.SIFT_create()        : Detektor fitur SIFT
- cv2.findHomography()     : Estimasi homography antar pasangan gambar
- np.linalg.inv()          : Menghitung invers matriks homography
- cv2.warpPerspective()    : Warping ke canvas referensi
- np.matmul() / operator @ : Chain multiplication homography
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan stitching
import cv2

# Mengimpor library NumPy untuk operasi matriks dan array
import numpy as np

# Mengimpor library os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi grid perbandingan
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
print("PERCOBAAN 4: MULTI-IMAGE PANORAMA")
print("=" * 65)

# ============================================================
# LANGKAH 1: Memuat 5 Gambar Panorama Wide
# ============================================================
print("\n[LANGKAH 1] Memuat 5 gambar panorama wide...")

# List untuk menyimpan gambar-gambar panorama wide
images = []
for i in range(1, 6):
    # Membaca setiap gambar panorama wide
    path = os.path.join(IMAGE_DIR, f"panorama_wide_{i}.jpg")
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] panorama_wide_{i}.jpg tidak ditemukan!")
        exit()
    images.append(img)
    print(f"  panorama_wide_{i}.jpg: {img.shape[1]}x{img.shape[0]} piksel")

# Menampilkan jumlah gambar yang berhasil dimuat
n_images = len(images)
print(f"  Total gambar: {n_images}")


# ============================================================
# FUNGSI HELPER: Mencocokkan Fitur dan Estimasi Homography
# ============================================================
def hitung_homography(img_src, img_dst, label=""):
    """
    Menghitung homography dari img_src ke img_dst.
    Tahapan: deteksi SIFT → FLANN match → ratio test → RANSAC homography.

    Parameter:
    - img_src : Gambar sumber yang akan di-warp
    - img_dst : Gambar tujuan/referensi
    - label   : Label deskriptif untuk logging

    Returns:
    - H       : Matriks homography 3x3 (src → dst)
    - n_inlier: Jumlah inlier RANSAC
    - n_match : Jumlah good matches
    """
    # Mengkonversi kedua gambar ke grayscale
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    # Membuat detektor SIFT
    sift = cv2.SIFT_create()

    # Mendeteksi keypoints dan deskriptor
    kp_src, desc_src = sift.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = sift.detectAndCompute(gray_dst, None)

    # Mengonfigurasi FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Memastikan deskriptor valid dan cukup untuk matching
    if desc_src is None or desc_dst is None or len(desc_src) < 4 or len(desc_dst) < 4:
        if label:
            print(f"    {label}: Tidak cukup fitur untuk matching")
        return np.eye(3, dtype=np.float64), 0, 0

    # Melakukan pencocokan k-NN (k=2)
    matches_knn = flann.knnMatch(desc_src, desc_dst, k=2)

    # Menerapkan Lowe's ratio test
    good = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Memastikan ada cukup kecocokan untuk homography (minimal 4)
    if len(good) < 4:
        if label:
            print(f"    {label}: {len(good)} matches (terlalu sedikit, gunakan identitas)")
        return np.eye(3, dtype=np.float64), 0, len(good)

    # Mengekstrak titik korespondensi
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Mengestimasi homography dengan RANSAC
    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Jika estimasi gagal, gunakan matriks identitas sebagai fallback
    if H is None:
        if label:
            print(f"    {label}: Homography gagal, gunakan identitas")
        return np.eye(3, dtype=np.float64), 0, len(good)

    # Menghitung jumlah inlier
    n_inlier = mask_h.ravel().sum() if mask_h is not None else 0
    n_match = len(good)

    # Menampilkan log
    if label:
        print(f"    {label}: {n_match} matches, {n_inlier} inliers")

    return H, n_inlier, n_match


# ============================================================
# LANGKAH 2: Memilih Gambar Tengah sebagai Referensi
# ============================================================
print("\n[LANGKAH 2] Memilih gambar tengah (ke-3) sebagai referensi...")

# Indeks gambar referensi (gambar ke-3, indeks 2)
ref_idx = 2
print(f"  Referensi: Gambar ke-{ref_idx + 1} (indeks {ref_idx})")
print(f"  Alasan: Gambar tengah meminimalkan distorsi akumulatif")

# ============================================================
# LANGKAH 3: Deteksi Fitur pada Semua Gambar
# ============================================================
print("\n[LANGKAH 3] Mendeteksi fitur SIFT pada semua gambar...")

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# List untuk menyimpan keypoints dan deskriptor semua gambar
all_kp = []
all_desc = []

for i, img in enumerate(images):
    # Mengkonversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mendeteksi fitur
    kp, desc = sift.detectAndCompute(gray, None)
    all_kp.append(kp)
    all_desc.append(desc)
    print(f"  Gambar {i + 1}: {len(kp)} keypoints")

# ============================================================
# LANGKAH 4: Mencocokkan Pasangan Bersebelahan
# ============================================================
print("\n[LANGKAH 4] Menghitung homography pasangan bersebelahan...")
print("  Pasangan: (1,2), (2,3), (3,4), (4,5)")

# Dictionary untuk menyimpan homography antar pasangan
# H_pair[i] = homography dari gambar i ke gambar i+1
H_pair = {}

for i in range(n_images - 1):
    # Menghitung homography dari gambar i ke gambar i+1
    label = f"H({i + 1}→{i + 2})"
    H, n_inlier, n_match = hitung_homography(images[i], images[i + 1], label)
    H_pair[i] = H

# ============================================================
# LANGKAH 5: Chain Homography ke Referensi (Gambar Tengah)
# ============================================================
print("\n[LANGKAH 5] Menghitung chain homography ke referensi...")

# Dictionary untuk menyimpan homography dari setiap gambar ke referensi
# H_to_ref[i] = homography dari gambar i ke gambar referensi
H_to_ref = {}

# Gambar referensi: identitas (tidak perlu transformasi)
H_to_ref[ref_idx] = np.eye(3, dtype=np.float64)
print(f"  H({ref_idx + 1}→ref) = Identitas (gambar referensi)")

# Chain homography untuk gambar di KIRI referensi (indeks < ref_idx)
# Logika: H(i→ref) = inv(H(i→i+1)) @ ... @ inv(H(ref-1→ref))
# Dalam kasus kita, H_pair[i] = H(i→i+1), jadi inv adalah H(i+1→i)
for i in range(ref_idx - 1, -1, -1):
    # H(i→ref) = H(i+1→ref) @ inv(H_pair[i])
    # Karena H_pair[i] = H(i→i+1), maka inv = H(i+1→i)
    # Kita butuh H(i→ref) = ??? ... sebenarnya:
    # H_pair[i] memetakan dari gambar i ke gambar i+1
    # Jadi untuk memetakan gambar i ke ref:
    # H(i→ref) = H(i+1→ref) @ H_pair[i]  -- BUKAN inv, karena H_pair sudah dari i ke i+1
    # Namun kita perlu hati-hati: H_pair[i] menggambar i→i+1
    # Jadi H(i→ref) = H((i+1)→ref) @ H(i→i+1)
    H_to_ref[i] = H_to_ref[i + 1] @ H_pair[i]
    print(f"  H({i + 1}→ref) = H({i + 2}→ref) @ H({i + 1}→{i + 2})")

# Chain homography untuk gambar di KANAN referensi (indeks > ref_idx)
# H(i→ref) = H((i-1)→ref) @ inv(H_pair[i-1])
# H_pair[i-1] = H(i-1→i), inv = H(i→i-1)
# H(i→ref) = H((i-1)→ref) @ inv(H(i-1→i))
for i in range(ref_idx + 1, n_images):
    # H_pair[i-1] memetakan gambar i-1 ke gambar i
    # Kita butuh gambar i ke ref, jadi: H(i→ref) = H((i-1)→ref) @ inv(H(i-1→i))
    H_inv = np.linalg.inv(H_pair[i - 1])
    H_to_ref[i] = H_to_ref[i - 1] @ H_inv
    print(f"  H({i + 1}→ref) = H({i}→ref) @ inv(H({i}→{i + 1}))")

# ============================================================
# LANGKAH 6: Menentukan Ukuran Canvas
# ============================================================
print("\n[LANGKAH 6] Menentukan ukuran canvas untuk panorama...")

# Mengumpulkan semua sudut yang sudah ditransformasi
all_corners = []
for i in range(n_images):
    h_img, w_img = images[i].shape[:2]

    # 4 sudut gambar i
    corners = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]).reshape(-1, 1, 2)

    # Mentransformasi sudut menggunakan homography ke referensi
    corners_t = cv2.perspectiveTransform(corners, H_to_ref[i])
    all_corners.append(corners_t)

# Menggabungkan semua sudut
all_corners = np.concatenate(all_corners, axis=0)

# Menentukan batas canvas
x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

# Menghitung ukuran canvas
canvas_w = x_max - x_min
canvas_h = y_max - y_min

# Membatasi ukuran canvas agar tidak terlalu besar
MAX_CANVAS = 8000
if canvas_w > MAX_CANVAS or canvas_h > MAX_CANVAS:
    scale = MAX_CANVAS / max(canvas_w, canvas_h)
    canvas_w = int(canvas_w * scale)
    canvas_h = int(canvas_h * scale)
    print(f"  [WARNING] Canvas terlalu besar, di-scale ke {canvas_w}x{canvas_h}")

# Menampilkan info canvas
print(f"  Batas: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
print(f"  Ukuran canvas: {canvas_w} x {canvas_h}")

# Membuat matriks translasi
T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

# ============================================================
# LANGKAH 7: Warping Semua Gambar ke Canvas
# ============================================================
print("\n[LANGKAH 7] Melakukan warping semua gambar ke canvas...")

# Mengukur waktu keseluruhan warping
waktu_mulai = time.time()

# Canvas utama untuk akumulasi hasil
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Counter untuk normalisasi blending (berapa gambar menumpuk)
count_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

# Akumulator floating point untuk blending
accumulator = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)

for i in range(n_images):
    # Menghitung homography gabungan: translasi + chain homography
    H_warp = T @ H_to_ref[i]

    # Melakukan warping gambar ke canvas
    warped = cv2.warpPerspective(images[i], H_warp, (canvas_w, canvas_h))

    # Membuat mask area yang terisi (non-hitam)
    mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

    # Menambahkan ke akumulator dan counter
    for c in range(3):
        accumulator[:, :, c] += warped[:, :, c].astype(np.float64) * mask
    count_map += mask

    # Menampilkan status warping gambar
    filled = np.sum(mask > 0)
    total = canvas_w * canvas_h
    print(f"  Gambar {i + 1}: {filled} piksel terisi ({filled / total * 100:.1f}%)")

# Melakukan rata-rata di area yang overlap (simple average blending)
# Menghindari pembagian dengan nol
count_map[count_map == 0] = 1

# Membuat hasil akhir dengan rata-rata
for c in range(3):
    canvas[:, :, c] = np.clip(accumulator[:, :, c] / count_map, 0, 255).astype(np.uint8)

# Menghitung waktu
waktu_warp = time.time() - waktu_mulai
print(f"  Waktu warping + blending: {waktu_warp:.3f} detik")

# Menyimpan hasil panorama
cv2.imwrite(os.path.join(OUTPUT_DIR, "04_panorama_5_gambar_ref_tengah.jpg"), canvas)
print("  [OK] Panorama 5 gambar (ref tengah) disimpan.")


# ============================================================
# LANGKAH 8: Crop Border Hitam
# ============================================================
print("\n[LANGKAH 8] Melakukan crop border hitam...")


def crop_border_hitam(img, threshold=5):
    """Menghilangkan border hitam dari hasil panorama."""
    # Mengkonversi ke grayscale dan threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological closing untuk mengisi lubang kecil
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Mencari kontur terbesar
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    # Mendapatkan bounding rect dari kontur terbesar
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return img[y:y + h, x:x + w]


# Melakukan crop pada hasil panorama
canvas_cropped = crop_border_hitam(canvas)
print(f"  Ukuran sebelum crop: {canvas.shape[1]}x{canvas.shape[0]}")
print(f"  Ukuran setelah crop: {canvas_cropped.shape[1]}x{canvas_cropped.shape[0]}")

# Menyimpan hasil yang sudah di-crop
cv2.imwrite(os.path.join(OUTPUT_DIR, "04_panorama_5_gambar_cropped.jpg"), canvas_cropped)
print("  [OK] Hasil cropped disimpan.")


# ============================================================
# FUNGSI HELPER: Build Panorama dengan Referensi Tertentu
# ============================================================
def bangun_panorama(images, H_pairs, ref_index, label=""):
    """
    Membangun panorama dari daftar gambar dengan referensi tertentu.

    Parameter:
    - images    : List gambar input
    - H_pairs   : Dictionary homography pasangan bersebelahan
    - ref_index : Indeks gambar referensi
    - label     : Label deskriptif untuk logging
    """
    n = len(images)

    # Menghitung chain homography ke referensi
    H_ref = {}
    H_ref[ref_index] = np.eye(3, dtype=np.float64)

    # Gambar di kiri referensi
    for i in range(ref_index - 1, -1, -1):
        H_ref[i] = H_ref[i + 1] @ H_pairs[i]

    # Gambar di kanan referensi
    for i in range(ref_index + 1, n):
        H_inv = np.linalg.inv(H_pairs[i - 1])
        H_ref[i] = H_ref[i - 1] @ H_inv

    # Menentukan ukuran canvas
    all_c = []
    for i in range(n):
        h_i, w_i = images[i].shape[:2]
        corners = np.float32([[0, 0], [w_i, 0], [w_i, h_i], [0, h_i]]).reshape(-1, 1, 2)
        corners_t = cv2.perspectiveTransform(corners, H_ref[i])
        all_c.append(corners_t)

    all_c = np.concatenate(all_c, axis=0)
    xmn, ymn = np.int32(all_c.min(axis=0).ravel())
    xmx, ymx = np.int32(all_c.max(axis=0).ravel())

    cw = min(xmx - xmn, MAX_CANVAS)
    ch = min(ymx - ymn, MAX_CANVAS)

    T_local = np.array([[1, 0, -xmn], [0, 1, -ymn], [0, 0, 1]], dtype=np.float64)

    # Warping dan blending
    acc = np.zeros((ch, cw, 3), dtype=np.float64)
    cnt = np.zeros((ch, cw), dtype=np.float32)

    for i in range(n):
        H_w = T_local @ H_ref[i]
        warped = cv2.warpPerspective(images[i], H_w, (cw, ch))
        m = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
        for c in range(3):
            acc[:, :, c] += warped[:, :, c].astype(np.float64) * m
        cnt += m

    cnt[cnt == 0] = 1
    result = np.zeros((ch, cw, 3), dtype=np.uint8)
    for c in range(3):
        result[:, :, c] = np.clip(acc[:, :, c] / cnt, 0, 255).astype(np.uint8)

    return crop_border_hitam(result)


# ============================================================
# LANGKAH 9: Perbandingan Referensi Kiri vs Tengah vs Kanan
# ============================================================
print("\n[LANGKAH 9] Membandingkan referensi kiri vs tengah vs kanan...")

# Daftar referensi yang akan diuji: kiri (0), tengah (2), kanan (4)
ref_options = [
    (0, "Referensi Kiri (Gambar 1)"),
    (2, "Referensi Tengah (Gambar 3)"),
    (4, "Referensi Kanan (Gambar 5)")
]

# Dictionary untuk menyimpan hasil setiap referensi
ref_results = {}

for ref_i, ref_label in ref_options:
    print(f"\n  Membangun panorama: {ref_label}...")
    waktu_mulai = time.time()

    # Membangun panorama dengan referensi tertentu
    result_ref = bangun_panorama(images, H_pair, ref_i, ref_label)

    waktu_ref = time.time() - waktu_mulai
    ref_results[ref_i] = {
        'result': result_ref,
        'label': ref_label,
        'waktu': waktu_ref,
        'ukuran': f"{result_ref.shape[1]}x{result_ref.shape[0]}"
    }

    # Menyimpan hasil
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"04_panorama_ref_{ref_i}.jpg"), result_ref)
    print(f"    Ukuran: {result_ref.shape[1]}x{result_ref.shape[0]}")
    print(f"    Waktu : {waktu_ref:.3f} detik")

# ============================================================
# LANGKAH 10: Analisis Distorsi pada Tepi
# ============================================================
print("\n[LANGKAH 10] Analisis distorsi pada tepi panorama...")

for ref_i, ref_label in ref_options:
    result = ref_results[ref_i]['result']
    h_r, w_r = result.shape[:2]

    # Mengambil strip kiri dan kanan panorama untuk analisis distorsi
    strip_width = min(100, w_r // 6)
    strip_left = result[:, :strip_width]
    strip_right = result[:, w_r - strip_width:]

    # Menghitung intensitas rata-rata sebagai indikator distorsi/warping
    mean_left = np.mean(strip_left)
    mean_right = np.mean(strip_right)

    # Menghitung area hitam/kosong (artefak warping)
    black_left = np.sum(cv2.cvtColor(strip_left, cv2.COLOR_BGR2GRAY) < 5)
    black_right = np.sum(cv2.cvtColor(strip_right, cv2.COLOR_BGR2GRAY) < 5)

    print(f"  {ref_label}:")
    print(f"    Strip kiri  - Mean: {mean_left:.1f}, Black pixels: {black_left}")
    print(f"    Strip kanan - Mean: {mean_right:.1f}, Black pixels: {black_right}")

# ============================================================
# LANGKAH 11: Membuat Grid Perbandingan
# ============================================================
print("\n[LANGKAH 11] Membuat grid perbandingan...")

# --- Grid 1: Input + Panorama utama ---
fig1, axes1 = plt.subplots(2, 5, figsize=(20, 7))

# Baris 1: 5 gambar input
for i in range(5):
    axes1[0, i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    axes1[0, i].set_title(f"Input {i + 1}", fontsize=10)
    axes1[0, i].axis("off")

# Baris 2: Panorama + info chain homography
# Subplot (1,0-1): Panorama referensi kiri
if 0 in ref_results:
    axes1[1, 0].imshow(cv2.cvtColor(ref_results[0]['result'], cv2.COLOR_BGR2RGB))
    axes1[1, 0].set_title("Ref: Gambar 1 (Kiri)", fontsize=10)
axes1[1, 0].axis("off")

# Subplot (1,1): kosong → informasi teks
axes1[1, 1].text(0.5, 0.5, "Chain Homography:\nH1→3 = H2→3 @ H1→2\nH2→3 = H2→3\n"
                 "H3→3 = I\nH4→3 = inv(H3→4)\nH5→3 = inv(H3→4) @ inv(H4→5)",
                 ha='center', va='center', fontsize=8,
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
axes1[1, 1].set_title("Chain Homography", fontsize=10)
axes1[1, 1].axis("off")

# Subplot (1,2): Panorama referensi tengah (utama)
if 2 in ref_results:
    axes1[1, 2].imshow(cv2.cvtColor(ref_results[2]['result'], cv2.COLOR_BGR2RGB))
    axes1[1, 2].set_title("Ref: Gambar 3 (Tengah) ★", fontsize=10, fontweight='bold')
axes1[1, 2].axis("off")

# Subplot (1,3): kosong → info distorsi
axes1[1, 3].text(0.5, 0.5, "Gambar tengah\nsebagai referensi\nmeminimalkan\ndistorsi di tepi\npanorama",
                 ha='center', va='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightcyan'))
axes1[1, 3].set_title("Info Distorsi", fontsize=10)
axes1[1, 3].axis("off")

# Subplot (1,4): Panorama referensi kanan
if 4 in ref_results:
    axes1[1, 4].imshow(cv2.cvtColor(ref_results[4]['result'], cv2.COLOR_BGR2RGB))
    axes1[1, 4].set_title("Ref: Gambar 5 (Kanan)", fontsize=10)
axes1[1, 4].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 4: Multi-Image Panorama (5 Gambar)",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid
plt.savefig(os.path.join(OUTPUT_DIR, "04_grid_input_dan_panorama.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid input dan panorama disimpan.")
plt.close()

# --- Grid 2: Perbandingan 3 referensi ---
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

for idx, (ref_i, ref_label) in enumerate(ref_options):
    if ref_i in ref_results:
        result = ref_results[ref_i]['result']
        axes2[idx].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes2[idx].set_title(
            f"{ref_label}\n{ref_results[ref_i]['ukuran']} | {ref_results[ref_i]['waktu']:.2f}s",
            fontsize=10
        )
    axes2[idx].axis("off")

# Judul dan layout
plt.suptitle("Perbandingan Referensi: Kiri vs Tengah vs Kanan",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid perbandingan referensi
plt.savefig(os.path.join(OUTPUT_DIR, "04_grid_perbandingan_referensi.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid perbandingan referensi disimpan.")
plt.close()

# ============================================================
# LANGKAH 12: Ringkasan
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 4: MULTI-IMAGE PANORAMA")
print("=" * 65)

# Tabel homography pasangan
print("\n  Homography Pasangan Bersebelahan:")
print(f"  {'Pasangan':<15} | {'Matches':>8} | {'Inliers':>8}")
print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}")
for i in range(n_images - 1):
    # Menghitung ulang untuk statistik (sudah dihitung sebelumnya)
    _, n_inl, n_mtch = hitung_homography(images[i], images[i + 1])
    print(f"  ({i + 1},{i + 2}){'':<10} | {n_mtch:>8} | {n_inl:>8}")

# Tabel perbandingan referensi
print(f"\n  Perbandingan Referensi:")
print(f"  {'Referensi':<30} | {'Ukuran':<15} | {'Waktu':>8}")
print(f"  {'-'*30}-+-{'-'*15}-+-{'-'*8}")
for ref_i, ref_label in ref_options:
    r = ref_results[ref_i]
    print(f"  {ref_label:<30} | {r['ukuran']:<15} | {r['waktu']:>7.3f}s")

# Menampilkan chain homography yang digunakan
print(f"\n  Chain Homography (ref=tengah):")
print(f"    H(1→3) = H(2→3) @ H(1→2)")
print(f"    H(2→3) = H(2→3)")
print(f"    H(3→3) = I (identitas)")
print(f"    H(4→3) = H(3→3) @ inv(H(3→4)) = inv(H(3→4))")
print(f"    H(5→3) = H(4→3) @ inv(H(4→5))")

# Daftar output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("04_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.SIFT_create()        → Detektor fitur SIFT")
print("    cv2.findHomography()     → Estimasi homography antar pasangan")
print("    np.linalg.inv()          → Invers matriks homography")
print("    cv2.warpPerspective()    → Warping ke canvas referensi")
print("    np.matmul() / @          → Chain multiplication homography")
print("    cv2.perspectiveTransform → Transformasi titik via homography")
print("=" * 65)
