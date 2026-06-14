"""
==========================================================================
PERCOBAAN 18: DOCUMENT ALIGNMENT DAN STITCHING
==========================================================================
Program ini mengimplementasikan stitching khusus untuk dokumen:
perspective correction, alignment berbasis kontur, dan penggabungan
dokumen secara vertikal maupun horizontal.

Dokumen stitching berbeda dari panorama biasa karena:
- Dokumen memiliki bentuk persegi panjang yang jelas
- Butuh koreksi perspektif sebelum digabungkan
- Fokus pada keterbacaan teks, bukan pemandangan

Konsep yang dipelajari:
- Deteksi sudut dokumen menggunakan contour + approxPolyDP
- Perspective correction (empat titik sudut)
- Vertical/horizontal stacking untuk dokumen
- Feature-based alignment untuk scan overlap
- Template matching untuk fine alignment
- Adaptive thresholding untuk peningkatan kualitas dokumen
- Binarisasi dokumen untuk keterbacaan optimal

Fungsi utama yang dipelajari:
- cv2.getPerspectiveTransform() : Perspektif transform dari 4 titik
- cv2.warpPerspective()         : Koreksi perspektif dokumen
- cv2.threshold()               : Binarisasi dokumen
- cv2.adaptiveThreshold()       : Binarisasi adaptif untuk teks
- cv2.findContours()            : Menemukan kontur dokumen
- cv2.approxPolyDP()            : Aproksimasi kontur ke poligon
- cv2.cvtColor()                : Konversi ruang warna
- np.vstack() / np.hstack()     : Menggabungkan gambar
- cv2.SIFT_create()             : Deteksi fitur untuk alignment
- cv2.matchTemplate()           : Template matching untuk refinement
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

# Mengimpor math untuk perhitungan geometri
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
print("PERCOBAAN 18: DOCUMENT ALIGNMENT DAN STITCHING")
print("=" * 65)


# ============================================================
# FUNGSI HELPER: Deteksi Sudut Dokumen
# ============================================================

def deteksi_sudut_dokumen(image, label=""):
    """
    Mendeteksi empat sudut dokumen menggunakan contour detection +
    polygon approximation (approxPolyDP).

    Pipeline:
    1. Konversi ke grayscale
    2. Gaussian blur untuk mengurangi noise
    3. Canny edge detection untuk menemukan tepi
    4. findContours untuk menemukan kontur
    5. approxPolyDP untuk aproksimasi ke poligon
    6. Memilih kontur terbesar dengan 4 sudut (dokumen)

    Parameter:
    - image : Gambar BGR input
    - label : Label logging

    Returns:
    - corners : Array 4 titik sudut (atau None jika gagal)
    - contour : Kontur dokumen yang terdeteksi
    """
    # Mengkonversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Menerapkan Gaussian blur untuk mengurangi noise
    # Kernel 5x5 memberikan smoothing yang cukup tanpa kehilangan detail tepi
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Mendeteksi tepi menggunakan Canny edge detector
    # Threshold 50 (low) dan 200 (high) untuk tepi yang jelas
    edges = cv2.Canny(blurred, 50, 200)

    # Dilatasi untuk menghubungkan tepi yang terputus
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Menemukan semua kontur pada gambar
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        if label:
            print(f"    {label}: Tidak ada kontur terdeteksi")
        return None, None

    # Mengurutkan kontur berdasarkan area (terbesar dulu)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Mencari kontur yang bisa diaproksimasikan menjadi 4 sudut (segi empat)
    doc_contour = None
    doc_corners = None

    for cnt in contours[:10]:  # Memeriksa 10 kontur terbesar
        # Menghitung keliling kontur
        perimeter = cv2.arcLength(cnt, True)

        # Mengaproksimasikan kontur ke poligon
        # Epsilon = 2% dari keliling (toleransi aproksimasi)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # Jika poligon memiliki 4 sudut, kemungkinan besar ini adalah dokumen
        if len(approx) == 4:
            # Memverifikasi bahwa area cukup besar (minimal 10% dari gambar)
            area = cv2.contourArea(approx)
            img_area = image.shape[0] * image.shape[1]

            if area > img_area * 0.1:
                doc_contour = cnt
                doc_corners = approx.reshape(4, 2)
                break

    # Jika tidak ditemukan poligon 4 sudut, gunakan bounding rect dari kontur terbesar
    if doc_corners is None:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        doc_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                                dtype=np.float32)
        doc_contour = cnt
        if label:
            print(f"    {label}: Menggunakan bounding rect (tidak ada 4-sudut)")
    else:
        if label:
            print(f"    {label}: Dokumen 4-sudut terdeteksi!")

    return doc_corners, doc_contour


def urutkan_sudut(pts):
    """
    Mengurutkan 4 titik sudut menjadi: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah.

    Algoritma:
    - Titik dengan sum(x+y) terkecil = kiri-atas
    - Titik dengan sum(x+y) terbesar = kanan-bawah
    - Titik dengan diff(y-x) terkecil = kanan-atas
    - Titik dengan diff(y-x) terbesar = kiri-bawah

    Parameter:
    - pts : Array 4 titik (4, 2)

    Returns:
    - ordered : Array 4 titik yang sudah diurutkan
    """
    pts = pts.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    # Menghitung jumlah dan selisih koordinat
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    # Kiri-atas memiliki sum terkecil
    ordered[0] = pts[np.argmin(s)]
    # Kanan-bawah memiliki sum terbesar
    ordered[2] = pts[np.argmax(s)]
    # Kanan-atas memiliki diff terkecil
    ordered[1] = pts[np.argmin(d)]
    # Kiri-bawah memiliki diff terbesar
    ordered[3] = pts[np.argmax(d)]

    return ordered


def koreksi_perspektif(image, corners, target_width=None, target_height=None, label=""):
    """
    Melakukan koreksi perspektif dokumen berdasarkan 4 titik sudut.

    Menggunakan getPerspectiveTransform untuk menghitung matriks transform
    dari sudut-sudut asli ke posisi persegi panjang sempurna.

    Parameter:
    - image         : Gambar input (BGR)
    - corners       : 4 titik sudut dokumen
    - target_width  : Lebar output (otomatis jika None)
    - target_height : Tinggi output (otomatis jika None)
    - label         : Label logging

    Returns:
    - corrected : Gambar dokumen yang sudah dikoreksi
    """
    # Mengurutkan sudut
    ordered = urutkan_sudut(corners)

    # Menghitung lebar dan tinggi target berdasarkan jarak sudut
    if target_width is None:
        # Lebar = rata-rata jarak atas dan bawah
        w1 = np.linalg.norm(ordered[1] - ordered[0])
        w2 = np.linalg.norm(ordered[2] - ordered[3])
        target_width = int(max(w1, w2))

    if target_height is None:
        # Tinggi = rata-rata jarak kiri dan kanan
        h1 = np.linalg.norm(ordered[3] - ordered[0])
        h2 = np.linalg.norm(ordered[2] - ordered[1])
        target_height = int(max(h1, h2))

    # Membatasi ukuran minimum
    target_width = max(target_width, 100)
    target_height = max(target_height, 100)

    # Mendefinisikan titik tujuan (persegi panjang sempurna)
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)

    # Menghitung matriks perspektif transform
    # getPerspectiveTransform menghitung 3x3 matrix dari 4 pasang titik
    M = cv2.getPerspectiveTransform(ordered, dst)

    # Menerapkan warping perspektif
    corrected = cv2.warpPerspective(image, M, (target_width, target_height))

    if label:
        print(f"    {label}: Koreksi perspektif → {target_width}x{target_height}")

    return corrected


def enhance_dokumen(image, method="adaptive", label=""):
    """
    Meningkatkan kualitas dokumen untuk keterbacaan teks.

    Metode yang tersedia:
    - 'binary'   : Binarisasi global (Otsu)
    - 'adaptive' : Adaptive thresholding
    - 'clahe'    : CLAHE untuk kontras adaptif

    Parameter:
    - image  : Gambar dokumen (BGR)
    - method : Metode enhancement
    - label  : Label logging

    Returns:
    - enhanced : Gambar yang sudah ditingkatkan
    """
    # Mengkonversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "binary":
        # Binarisasi Otsu: otomatis menentukan threshold optimal
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    elif method == "adaptive":
        # Adaptive threshold: threshold berbeda untuk setiap region
        # blockSize=15: ukuran region lokal
        # C=10: konstanta yang dikurangkan dari rata-rata
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 10)
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    elif method == "clahe":
        # CLAHE: Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        enhanced = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

    else:
        enhanced = image.copy()

    if label:
        print(f"    {label}: Enhancement '{method}' diterapkan")

    return enhanced


# ============================================================
# FUNGSI HELPER: Feature-Based Stitching
# ============================================================

def stitch_dokumen_feature(img1, img2, label=""):
    """
    Melakukan stitching dua dokumen menggunakan SIFT features.
    Cocok untuk dokumen yang memiliki area overlap.

    Parameter:
    - img1  : Dokumen pertama (BGR)
    - img2  : Dokumen kedua (BGR)
    - label : Label logging

    Returns:
    - result : Dokumen yang sudah di-stitch
    """
    # Mengkonversi ke grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Membuat detektor SIFT
    sift = cv2.SIFT_create()

    # Mendeteksi keypoints dan deskriptor
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
        if label:
            print(f"    {label}: Tidak cukup fitur, fallback ke vstack")
        return np.vstack([img1, img2])

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if label:
        print(f"    {label}: {len(good)} good matches")

    if len(good) < 10:
        if label:
            print(f"    {label}: Match kurang, fallback ke vstack")
        # Resize ke lebar yang sama sebelum vstack
        w_target = max(img1.shape[1], img2.shape[1])
        r1 = cv2.resize(img1, (w_target, img1.shape[0]))
        r2 = cv2.resize(img2, (w_target, img2.shape[0]))
        return np.vstack([r1, r2])

    # Menghitung homography dari img2 ke img1
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if H is None:
        if label:
            print(f"    {label}: Homography gagal, fallback ke vstack")
        w_target = max(img1.shape[1], img2.shape[1])
        r1 = cv2.resize(img1, (w_target, img1.shape[0]))
        r2 = cv2.resize(img2, (w_target, img2.shape[0]))
        return np.vstack([r1, r2])

    # Warping img2 ke frame img1
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Menghitung batas canvas
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners2_t = cv2.perspectiveTransform(corners2, H)
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_c = np.concatenate([corners1, corners2_t], axis=0)

    x_min = int(np.floor(all_c[:, :, 0].min()))
    y_min = int(np.floor(all_c[:, :, 1].min()))
    x_max = int(np.ceil(all_c[:, :, 0].max()))
    y_max = int(np.ceil(all_c[:, :, 1].max()))
    x_min, y_min = min(x_min, 0), min(y_min, 0)

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_w = min(x_max - x_min, 5000)
    canvas_h = min(y_max - y_min, 5000)

    # Warping img2
    warped2 = cv2.warpPerspective(img2, T @ H, (canvas_w, canvas_h))

    # Menempatkan img1
    ox, oy = -x_min, -y_min
    ye = min(oy + h1, canvas_h)
    xe = min(ox + w1, canvas_w)

    # Blending sederhana: img1 menulis di atas warped img2
    mask1 = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    mask1[oy:ye, ox:xe] = 1.0

    mask2 = (cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

    # Distance-based blending di area overlap
    dist1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
    total = dist1 + dist2 + 1e-10
    w1_map = dist1 / total
    w2_map = dist2 / total

    result = np.zeros_like(warped2, dtype=np.float64)

    canvas_img1 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_img1[oy:ye, ox:xe] = img1[:ye - oy, :xe - ox]

    for c in range(3):
        result[:, :, c] = (canvas_img1[:, :, c].astype(np.float64) * w1_map +
                           warped2[:, :, c].astype(np.float64) * w2_map)

    return np.clip(result, 0, 255).astype(np.uint8)


def stitch_dokumen_template(img1, img2, overlap_height=50, label=""):
    """
    Melakukan alignment dokumen menggunakan template matching untuk
    menemukan posisi overlap terbaik antara dua dokumen vertikal.

    Parameter:
    - img1            : Dokumen atas
    - img2            : Dokumen bawah
    - overlap_height  : Perkiraan tinggi overlap (piksel)
    - label           : Label logging

    Returns:
    - stitched : Dokumen yang sudah digabungkan
    """
    # Resize ke lebar yang sama
    w_target = max(img1.shape[1], img2.shape[1])
    if img1.shape[1] != w_target:
        scale = w_target / img1.shape[1]
        img1 = cv2.resize(img1, (w_target, int(img1.shape[0] * scale)))
    if img2.shape[1] != w_target:
        scale = w_target / img2.shape[1]
        img2 = cv2.resize(img2, (w_target, int(img2.shape[0] * scale)))

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Mengambil bagian bawah img1 sebagai template
    template_h = min(overlap_height, h1 // 3, h2 // 3)
    template = img1[h1 - template_h:h1, :]

    # Mengkonversi ke grayscale untuk template matching
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Melakukan template matching menggunakan TM_CCOEFF_NORMED
    # Metode ini memberikan nilai korelasi -1 (buruk) hingga 1 (sempurna)
    result = cv2.matchTemplate(img2_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Menemukan lokasi match terbaik
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    best_y = max_loc[1]

    if label:
        print(f"    {label}: Best match at y={best_y}, confidence={max_val:.4f}")

    # Jika confidence cukup tinggi, gunakan posisi template match
    if max_val > 0.3:
        # Overlap dimulai di best_y pada img2
        # Menggabungkan: img1 (sampai batas overlap) + img2 (setelah overlap)
        stitched = np.vstack([
            img1[:h1 - template_h + best_y // 2, :],
            img2[best_y + template_h // 2:, :]
        ])
    else:
        # Confidence rendah, lakukan simple stacking
        if label:
            print(f"    {label}: Confidence rendah, menggunakan simple vstack")
        stitched = np.vstack([img1, img2])

    return stitched


# ============================================================
# LANGKAH 1: Memuat Gambar Dokumen
# ============================================================
print("\n[LANGKAH 1] Memuat gambar dokumen...")

# Membaca 3 gambar dokumen
doc_files = ["dokumen_1.jpg", "dokumen_2.jpg", "dokumen_3.jpg"]
documents = []

for f in doc_files:
    path = os.path.join(IMAGE_DIR, f)
    img = cv2.imread(path)
    if img is not None:
        documents.append(img)
        print(f"  Loaded: {f} ({img.shape[1]}x{img.shape[0]})")
    else:
        print(f"  [WARNING] Gagal memuat: {f}")

n_docs = len(documents)
print(f"\n  Total dokumen dimuat: {n_docs}")

if n_docs == 0:
    print("[ERROR] Tidak ada dokumen yang berhasil dimuat!")
    print("Jalankan download_image.py terlebih dahulu.")
    exit()


# ============================================================
# LANGKAH 2: Deteksi Sudut Dokumen
# ============================================================
print("\n[LANGKAH 2] Mendeteksi sudut dokumen pada setiap gambar...")

doc_corners_list = []
doc_contours_list = []

for i, doc in enumerate(documents):
    print(f"\n  Dokumen {i + 1}:")
    corners, contour = deteksi_sudut_dokumen(doc, f"Dokumen {i + 1}")
    doc_corners_list.append(corners)
    doc_contours_list.append(contour)

    if corners is not None:
        print(f"    Sudut terdeteksi: {corners.tolist()}")

# Visualisasi deteksi sudut
try:
    n = min(n_docs, 3)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for i in range(n):
        vis = documents[i].copy()

        # Menggambar kontur dan sudut
        if doc_contours_list[i] is not None:
            cv2.drawContours(vis, [doc_contours_list[i]], -1, (0, 255, 0), 3)

        if doc_corners_list[i] is not None:
            for j, pt in enumerate(doc_corners_list[i]):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
                cv2.putText(vis, str(j + 1), (int(pt[0]) + 15, int(pt[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Dokumen {i + 1}\nDeteksi Sudut", fontsize=12)
        axes[i].axis('off')

    plt.suptitle("Deteksi Sudut Dokumen", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_deteksi_sudut.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("\n  Visualisasi deteksi sudut disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat visualisasi: {e}")


# ============================================================
# LANGKAH 3: Koreksi Perspektif Setiap Dokumen
# ============================================================
print("\n[LANGKAH 3] Melakukan koreksi perspektif pada setiap dokumen...")

corrected_docs = []

for i, doc in enumerate(documents):
    if doc_corners_list[i] is not None:
        # Melakukan koreksi perspektif menggunakan 4 sudut yang terdeteksi
        corrected = koreksi_perspektif(doc, doc_corners_list[i],
                                        label=f"Dokumen {i + 1}")
        corrected_docs.append(corrected)
    else:
        # Jika sudut tidak terdeteksi, gunakan gambar asli
        print(f"    Dokumen {i + 1}: Tidak ada sudut, menggunakan gambar asli")
        corrected_docs.append(doc.copy())

    # Menyimpan hasil koreksi
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"18_corrected_doc{i + 1}.jpg"),
                corrected_docs[-1])

# Visualisasi sebelum dan sesudah koreksi
try:
    n = min(n_docs, 3)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 12))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # Baris atas: gambar asli
        axes[0, i].imshow(cv2.cvtColor(documents[i], cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Dokumen {i + 1} - Asli\n"
                             f"({documents[i].shape[1]}x{documents[i].shape[0]})",
                             fontsize=11)
        axes[0, i].axis('off')

        # Baris bawah: gambar terkoreksi
        axes[1, i].imshow(cv2.cvtColor(corrected_docs[i], cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Dokumen {i + 1} - Terkoreksi\n"
                             f"({corrected_docs[i].shape[1]}x{corrected_docs[i].shape[0]})",
                             fontsize=11)
        axes[1, i].axis('off')

    plt.suptitle("Koreksi Perspektif Dokumen", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "18_koreksi_perspektif.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Visualisasi koreksi perspektif disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat visualisasi: {e}")


# ============================================================
# LANGKAH 4: Metode 1 - Simple Vertical Stacking (vstack)
# ============================================================
print("\n[LANGKAH 4] Metode 1: Simple vertical stacking...")

try:
    # Resize semua dokumen ke lebar yang sama
    target_width = max(doc.shape[1] for doc in corrected_docs)
    resized_docs = []

    for i, doc in enumerate(corrected_docs):
        if doc.shape[1] != target_width:
            # Menghitung skala dan me-resize
            scale = target_width / doc.shape[1]
            new_h = int(doc.shape[0] * scale)
            resized = cv2.resize(doc, (target_width, new_h))
            resized_docs.append(resized)
        else:
            resized_docs.append(doc.copy())

    # Menggabungkan secara vertikal menggunakan np.vstack
    stacked_vertical = np.vstack(resized_docs)

    # Menyimpan hasil
    cv2.imwrite(os.path.join(OUTPUT_DIR, "18_vstack_result.jpg"), stacked_vertical)
    print(f"  Hasil vstack: {stacked_vertical.shape[1]}x{stacked_vertical.shape[0]}")
    print(f"  Disimpan ke output/")

except Exception as e:
    print(f"  [ERROR] Vertical stacking gagal: {e}")
    stacked_vertical = None

# Juga membuat horizontal stacking
try:
    # Resize semua dokumen ke tinggi yang sama
    target_height = max(doc.shape[0] for doc in corrected_docs)
    resized_h_docs = []

    for i, doc in enumerate(corrected_docs):
        if doc.shape[0] != target_height:
            scale = target_height / doc.shape[0]
            new_w = int(doc.shape[1] * scale)
            resized = cv2.resize(doc, (new_w, target_height))
            resized_h_docs.append(resized)
        else:
            resized_h_docs.append(doc.copy())

    # Menggabungkan secara horizontal menggunakan np.hstack
    stacked_horizontal = np.hstack(resized_h_docs)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "18_hstack_result.jpg"), stacked_horizontal)
    print(f"  Hasil hstack: {stacked_horizontal.shape[1]}x{stacked_horizontal.shape[0]}")

except Exception as e:
    print(f"  [ERROR] Horizontal stacking gagal: {e}")
    stacked_horizontal = None


# ============================================================
# LANGKAH 5: Metode 2 - Feature-Based Stitching (SIFT)
# ============================================================
print("\n[LANGKAH 5] Metode 2: Feature-based stitching (SIFT)...")

try:
    if n_docs >= 2:
        # Stitching dokumen 1 dan 2 menggunakan SIFT features
        t0 = time.time()
        feature_stitch_12 = stitch_dokumen_feature(
            corrected_docs[0], corrected_docs[1], "Doc1+Doc2 SIFT"
        )
        t_feature = time.time() - t0
        print(f"  Waktu feature stitching Doc1+2: {t_feature:.3f} detik")

        cv2.imwrite(os.path.join(OUTPUT_DIR, "18_feature_stitch_12.jpg"), feature_stitch_12)

        # Jika ada dokumen 3, lanjutkan stitching
        if n_docs >= 3:
            feature_stitch_all = stitch_dokumen_feature(
                feature_stitch_12, corrected_docs[2], "Result+Doc3 SIFT"
            )
            cv2.imwrite(os.path.join(OUTPUT_DIR, "18_feature_stitch_all.jpg"),
                        feature_stitch_all)
            print(f"  Hasil feature stitch all: "
                  f"{feature_stitch_all.shape[1]}x{feature_stitch_all.shape[0]}")
        else:
            feature_stitch_all = feature_stitch_12
    else:
        feature_stitch_all = corrected_docs[0].copy()
        print("  Hanya 1 dokumen, tidak perlu stitching")

except Exception as e:
    print(f"  [ERROR] Feature stitching gagal: {e}")
    feature_stitch_all = None


# ============================================================
# LANGKAH 6: Metode 3 - Template Matching Alignment
# ============================================================
print("\n[LANGKAH 6] Metode 3: Template matching alignment...")

try:
    if n_docs >= 2:
        # Stitching dokumen 1 dan 2 menggunakan template matching
        t0 = time.time()
        template_stitch_12 = stitch_dokumen_template(
            corrected_docs[0], corrected_docs[1],
            overlap_height=60, label="Doc1+Doc2 Template"
        )
        t_template = time.time() - t0
        print(f"  Waktu template stitching Doc1+2: {t_template:.3f} detik")

        cv2.imwrite(os.path.join(OUTPUT_DIR, "18_template_stitch_12.jpg"), template_stitch_12)

        # Jika ada dokumen 3, lanjutkan stitching
        if n_docs >= 3:
            template_stitch_all = stitch_dokumen_template(
                template_stitch_12, corrected_docs[2],
                overlap_height=60, label="Result+Doc3 Template"
            )
            cv2.imwrite(os.path.join(OUTPUT_DIR, "18_template_stitch_all.jpg"),
                        template_stitch_all)
            print(f"  Hasil template stitch all: "
                  f"{template_stitch_all.shape[1]}x{template_stitch_all.shape[0]}")
        else:
            template_stitch_all = template_stitch_12
    else:
        template_stitch_all = corrected_docs[0].copy()

except Exception as e:
    print(f"  [ERROR] Template stitching gagal: {e}")
    template_stitch_all = None


# ============================================================
# LANGKAH 7: Enhancement Dokumen (Binarisasi & Contrast)
# ============================================================
print("\n[LANGKAH 7] Meningkatkan kualitas dokumen hasil stitching...")

# Menggunakan vstack result sebagai baseline
enhanced_results = {}

if stacked_vertical is not None:
    # Method 1: Otsu binarization
    enhanced_binary = enhance_dokumen(stacked_vertical, "binary", "Otsu Binary")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "18_enhanced_binary.jpg"), enhanced_binary)
    enhanced_results["Otsu Binary"] = enhanced_binary

    # Method 2: Adaptive threshold
    enhanced_adaptive = enhance_dokumen(stacked_vertical, "adaptive", "Adaptive")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "18_enhanced_adaptive.jpg"), enhanced_adaptive)
    enhanced_results["Adaptive"] = enhanced_adaptive

    # Method 3: CLAHE
    enhanced_clahe = enhance_dokumen(stacked_vertical, "clahe", "CLAHE")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "18_enhanced_clahe.jpg"), enhanced_clahe)
    enhanced_results["CLAHE"] = enhanced_clahe

# Visualisasi perbandingan enhancement
try:
    methods_vis = list(enhanced_results.keys())
    n_methods = len(methods_vis)

    if n_methods > 0 and stacked_vertical is not None:
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 8))

        # Original
        # Crop bagian atas saja untuk visualisasi
        crop_h = min(stacked_vertical.shape[0], 600)
        axes[0].imshow(cv2.cvtColor(stacked_vertical[:crop_h], cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis('off')

        for i, name in enumerate(methods_vis):
            crop = enhanced_results[name][:crop_h]
            axes[i + 1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(name, fontsize=11)
            axes[i + 1].axis('off')

        plt.suptitle("Perbandingan Enhancement Dokumen", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "18_enhancement_comparison.png"),
                dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Perbandingan enhancement disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat perbandingan: {e}")


# ============================================================
# LANGKAH 8: Demo Whiteboard-Style Stitching
# ============================================================
print("\n[LANGKAH 8] Demo whiteboard-style stitching...")

try:
    # Mensimulasikan gambar whiteboard dengan mengambil dokumen dan merotasinya
    if n_docs >= 1:
        doc_base = corrected_docs[0].copy()
        h_b, w_b = doc_base.shape[:2]

        # Membuat versi yang sedikit dirotasi dan digeser (simulasi tangkapan kamera berbeda)
        center = (w_b // 2, h_b // 2)

        # Rotasi sedikit (2 derajat) - simulasi kamera tidak sempurna
        M_rot1 = cv2.getRotationMatrix2D(center, 2, 1.0)
        wb_rotated1 = cv2.warpAffine(doc_base, M_rot1, (w_b, h_b),
                                      borderValue=(255, 255, 255))

        # Rotasi berlawanan (-3 derajat)
        M_rot2 = cv2.getRotationMatrix2D(center, -3, 1.0)
        wb_rotated2 = cv2.warpAffine(doc_base, M_rot2, (w_b, h_b),
                                      borderValue=(255, 255, 255))

        # Perspektif kecil
        src_pts = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]])
        dst_pts = np.float32([[15, 10], [w_b - 5, 20], [w_b - 10, h_b - 15], [8, h_b - 5]])
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        wb_perspective = cv2.warpPerspective(doc_base, M_persp, (w_b, h_b),
                                              borderValue=(255, 255, 255))

        print("  Gambar whiteboard simulasi dibuat:")
        print(f"    - Original: {w_b}x{h_b}")
        print(f"    - Rotated +2°: {wb_rotated1.shape[1]}x{wb_rotated1.shape[0]}")
        print(f"    - Rotated -3°: {wb_rotated2.shape[1]}x{wb_rotated2.shape[0]}")
        print(f"    - Perspective: {wb_perspective.shape[1]}x{wb_perspective.shape[0]}")

        # Koreksi masing-masing menggunakan feature matching ke original
        wb_images = [
            ("Original", doc_base),
            ("Rotated +2°", wb_rotated1),
            ("Rotated -3°", wb_rotated2),
            ("Perspective", wb_perspective)
        ]

        # Visualisasi whiteboard variasi
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        for j, (name, img) in enumerate(wb_images):
            crop_h = min(img.shape[0], 400)
            axes[j].imshow(cv2.cvtColor(img[:crop_h], cv2.COLOR_BGR2RGB))
            axes[j].set_title(name, fontsize=11)
            axes[j].axis('off')

        plt.suptitle("Simulasi Whiteboard dengan Variasi Sudut Kamera",
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "18_whiteboard_variasi.png"),
                dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Visualisasi whiteboard disimpan.")

        # Koreksi perspektif pada whiteboard
        wb_corrected = []
        for name, img in wb_images:
            corners, contour = deteksi_sudut_dokumen(img, f"WB {name}")
            if corners is not None:
                corr = koreksi_perspektif(img, corners, target_width=w_b,
                                           target_height=h_b, label=f"WB {name}")
            else:
                corr = img.copy()
            wb_corrected.append((name, corr))

        # Visualisasi koreksi whiteboard
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for j, (name, img) in enumerate(wb_images):
            crop_h = min(img.shape[0], 400)
            axes[0, j].imshow(cv2.cvtColor(img[:crop_h], cv2.COLOR_BGR2RGB))
            axes[0, j].set_title(f"{name}\n(Sebelum koreksi)", fontsize=10)
            axes[0, j].axis('off')

        for j, (name, img) in enumerate(wb_corrected):
            crop_h = min(img.shape[0], 400)
            axes[1, j].imshow(cv2.cvtColor(img[:crop_h], cv2.COLOR_BGR2RGB))
            axes[1, j].set_title(f"{name}\n(Sesudah koreksi)", fontsize=10)
            axes[1, j].axis('off')

        plt.suptitle("Koreksi Whiteboard: Sebelum vs Sesudah",
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "18_whiteboard_koreksi.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Koreksi whiteboard disimpan.")

except Exception as e:
    print(f"  [WARNING] Gagal membuat demo whiteboard: {e}")


# ============================================================
# LANGKAH 9: Perbandingan Ketiga Metode
# ============================================================
print("\n[LANGKAH 9] Membuat perbandingan ketiga metode stitching...")

try:
    results_comp = {
        "Vertical Stack": stacked_vertical,
        "Feature (SIFT)": feature_stitch_all,
        "Template Match": template_stitch_all
    }

    valid_results = {k: v for k, v in results_comp.items() if v is not None}
    n_res = len(valid_results)

    if n_res > 0:
        fig, axes = plt.subplots(1, n_res, figsize=(7 * n_res, 10))
        if n_res == 1:
            axes = [axes]

        for i, (name, img) in enumerate(valid_results.items()):
            # Crop bagian atas untuk visualisasi
            crop_h = min(img.shape[0], 800)
            axes[i].imshow(cv2.cvtColor(img[:crop_h], cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"{name}\n({img.shape[1]}x{img.shape[0]})", fontsize=12)
            axes[i].axis('off')

        plt.suptitle("Perbandingan 3 Metode Document Stitching",
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "18_perbandingan_3_metode.png"),
                dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Perbandingan 3 metode disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat perbandingan: {e}")


# ============================================================
# LANGKAH 10: Visualisasi Pipeline Dokumen Lengkap
# ============================================================
print("\n[LANGKAH 10] Membuat visualisasi pipeline dokumen lengkap...")

try:
    fig = plt.figure(figsize=(22, 20))

    # Baris 1: Input documents
    for i in range(min(n_docs, 3)):
        ax = fig.add_subplot(5, 3, i + 1)
        crop_h = min(documents[i].shape[0], 350)
        ax.imshow(cv2.cvtColor(documents[i][:crop_h], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Input Doc {i + 1}", fontsize=10)
        ax.axis('off')

    # Baris 2: Corrected documents
    for i in range(min(n_docs, 3)):
        ax = fig.add_subplot(5, 3, 4 + i)
        crop_h = min(corrected_docs[i].shape[0], 350)
        ax.imshow(cv2.cvtColor(corrected_docs[i][:crop_h], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Corrected Doc {i + 1}", fontsize=10)
        ax.axis('off')

    # Baris 3: Stacking results
    if stacked_vertical is not None:
        ax = fig.add_subplot(5, 3, 7)
        crop_h = min(stacked_vertical.shape[0], 350)
        ax.imshow(cv2.cvtColor(stacked_vertical[:crop_h], cv2.COLOR_BGR2RGB))
        ax.set_title("VStack", fontsize=10)
        ax.axis('off')

    if feature_stitch_all is not None:
        ax = fig.add_subplot(5, 3, 8)
        crop_h = min(feature_stitch_all.shape[0], 350)
        ax.imshow(cv2.cvtColor(feature_stitch_all[:crop_h], cv2.COLOR_BGR2RGB))
        ax.set_title("Feature SIFT", fontsize=10)
        ax.axis('off')

    if template_stitch_all is not None:
        ax = fig.add_subplot(5, 3, 9)
        crop_h = min(template_stitch_all.shape[0], 350)
        ax.imshow(cv2.cvtColor(template_stitch_all[:crop_h], cv2.COLOR_BGR2RGB))
        ax.set_title("Template Match", fontsize=10)
        ax.axis('off')

    # Baris 4: Enhancement methods
    enhancement_names = list(enhanced_results.keys())
    for i, name in enumerate(enhancement_names[:3]):
        ax = fig.add_subplot(5, 3, 10 + i)
        crop_h = min(enhanced_results[name].shape[0], 350)
        ax.imshow(cv2.cvtColor(enhanced_results[name][:crop_h], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Enhancement: {name}", fontsize=10)
        ax.axis('off')

    # Baris 5: Horizontal stacking if available
    if stacked_horizontal is not None:
        ax = fig.add_subplot(5, 1, 5)
        crop_h = min(stacked_horizontal.shape[0], 250)
        ax.imshow(cv2.cvtColor(stacked_horizontal[:crop_h], cv2.COLOR_BGR2RGB))
        ax.set_title("Horizontal Stack (HStack)", fontsize=11)
        ax.axis('off')

    plt.suptitle("Pipeline Lengkap Document Stitching (Percobaan 18)",
                  fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_pipeline_lengkap.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Pipeline lengkap disimpan.")
except Exception as e:
    print(f"  [WARNING] Gagal membuat pipeline lengkap: {e}")


# ============================================================
# LANGKAH 11: Statistik dan Ringkasan
# ============================================================
print("\n[LANGKAH 11] Statistik dokumen stitching:")
print("=" * 60)
print(f"{'Metode':<20} {'Ukuran (WxH)':<20} {'Piksel Total':<15}")
print("-" * 60)

all_results = {
    "VStack": stacked_vertical,
    "HStack": stacked_horizontal,
    "Feature (SIFT)": feature_stitch_all,
    "Template Match": template_stitch_all
}

for name, img in all_results.items():
    if img is not None:
        h, w = img.shape[:2]
        total = h * w
        print(f"{name:<20} {w}x{h:<14} {total:>12,}")
    else:
        print(f"{name:<20} {'N/A':<20} {'N/A':<15}")

print("-" * 60)

# Informasi dokumen input
print(f"\nDokumen input:")
for i, doc in enumerate(documents):
    print(f"  Doc {i + 1}: {doc.shape[1]}x{doc.shape[0]}")

for i, doc in enumerate(corrected_docs):
    print(f"  Doc {i + 1} (corrected): {doc.shape[1]}x{doc.shape[0]}")


# ============================================================
# RINGKASAN PROGRAM
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 18")
print("=" * 65)
print("""
Apa yang telah dipelajari:
1. Deteksi Sudut Dokumen:
   - findContours + approxPolyDP untuk menemukan 4 sudut
   - Sorting kontour berdasarkan area
   - Verifikasi bahwa aproksimasi memiliki tepat 4 sudut

2. Koreksi Perspektif:
   - getPerspectiveTransform dari 4 pasang titik
   - warpPerspective untuk meratakan dokumen
   - Ordering sudut: kiri-atas → kanan-atas → kanan-bawah → kiri-bawah

3. Document Stacking:
   - np.vstack: penggabungan vertikal (atas-bawah)
   - np.hstack: penggabungan horizontal (kiri-kanan)
   - Resize ke dimensi yang sama sebelum stacking

4. Feature-Based Stitching:
   - SIFT + FLANN untuk alignment dokumen yang overlap
   - Distance-based blending di area overlap

5. Template Matching Alignment:
   - matchTemplate untuk menemukan posisi overlap terbaik
   - TM_CCOEFF_NORMED memberikan confidence score 0-1

6. Enhancement Dokumen:
   - Otsu binarization: threshold otomatis global
   - Adaptive threshold: threshold per region
   - CLAHE: kontras adaptif tanpa binarisasi

7. Whiteboard Correction:
   - Menangani dokumen yang difoto dari sudut miring
   - Koreksi perspektif + enhancement

File output disimpan di folder: output/
""")

print("Program selesai dijalankan.")
print("=" * 65)
