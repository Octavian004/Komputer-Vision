"""
==========================================================================
PERCOBAAN 09: KOREKSI PERSPEKTIF DOKUMEN
==========================================================================
Aplikasi praktis homography: mengoreksi perspektif dokumen miring
menjadi tampak datar (bird's eye view). Sangat berguna untuk
scanner dokumen di smartphone.

Pipeline:
1. Preprocessing (grayscale, blur, edge)
2. Deteksi kontur terbesar (kertas)
3. Approximasi 4 sudut
4. Hitung homography
5. Warp ke tampilan datar

Fungsi utama:
- cv2.findContours()          : cari kontur
- cv2.approxPolyDP()          : approksasi poligon (4 sudut)
- cv2.contourArea()           : luas kontur
- cv2.getPerspectiveTransform(): hitung H dari 4 titik
- cv2.warpPerspective()       : warp gambar
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 09: KOREKSI PERSPEKTIF DOKUMEN")
print("=" * 60)

# ============================================================
# 1. Memuat gambar dokumen nyata dan simulasi perspektif miring
# ============================================================
print("\n--- 1. Memuat Gambar Dokumen Nyata ---")

img_path = os.path.join(IMAGE_DIR, "dokumen_asli.jpg")
if not os.path.exists(img_path):
    print("[ERROR] dokumen_asli.jpg tidak ditemukan. Jalankan download_image.py!"); exit()

doc_original = cv2.imread(img_path)
doc_original = cv2.resize(doc_original, (480, 640))  # portrait (dokumen)
h_d, w_d = doc_original.shape[:2]

# Simulasi perspektif miring: warp gambar nyata ke tampilan sudut kamera
# (seperti foto dokumen dari sudut 45 derajat)
bg = np.full((640, 800, 3), (80, 60, 40), dtype=np.uint8)  # latar meja
src_pts = np.float32([[0, 0], [w_d, 0], [w_d, h_d], [0, h_d]])
dst_pts = np.float32([[140, 60], [560, 90], [590, 570], [110, 550]])
H_place = cv2.getPerspectiveTransform(src_pts, dst_pts)
doc_warped = cv2.warpPerspective(doc_original, H_place, (800, 640))
mask = cv2.warpPerspective(np.full((h_d, w_d), 255, dtype=np.uint8),
                           H_place, (800, 640))
bg[mask > 0] = doc_warped[mask > 0]
img_scene = bg
print(f"  Gambar dokumen nyata dimuat dan dimiringkan: {img_scene.shape}")

# ============================================================
# 2. Preprocessing: grayscale, blur, edge detection
# ============================================================
print("\n--- 2. Preprocessing ---")

# Konversi ke grayscale
gray = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# Gaussian blur untuk mengurangi noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Deteksi tepi dengan Canny
edges = cv2.Canny(blurred, 50, 150)

# Dilasi untuk menghubungkan gap pada tepi
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)

cv2.imwrite(os.path.join(OUTPUT_DIR, "09_edges_dokumen.png"), edges_dilated)
print(f"  Edge detection selesai")

# ============================================================
# 3. Menemukan kontur terbesar (kertas)
# ============================================================
print("\n--- 3. Mencari Kontur Dokumen ---")

# cv2.findContours mencari semua kontur di gambar biner
# RETR_EXTERNAL: hanya kontur terluar
# CHAIN_APPROX_SIMPLE: simpan hanya titik ujung (hemat memori)
contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

# Urutkan berdasarkan luas (terbesar dulu)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(f"  Total kontur: {len(contours)}")

# Cari kontur yang bisa di-approximasi menjadi 4 titik (segiempat)
doc_contour = None
for cnt in contours:
    # cv2.arcLength menghitung keliling kontur
    perimeter = cv2.arcLength(cnt, True)
    
    # cv2.approxPolyDP mengaproksimasi kontur menjadi poligon
    # epsilon = 2% dari keliling → toleransi aproksimasi
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    
    # Jika hasilnya 4 titik dan cukup besar, ini kemungkinan dokumen
    if len(approx) == 4 and cv2.contourArea(approx) > 5000:
        doc_contour = approx
        print(f"  Dokumen ditemukan! Luas: {cv2.contourArea(approx):.0f}")
        break

# Gambar kontur yang ditemukan
img_contour = img_scene.copy()
if doc_contour is not None:
    cv2.drawContours(img_contour, [doc_contour], -1, (0, 255, 0), 3)
    for pt in doc_contour:
        cv2.circle(img_contour, tuple(pt[0]), 8, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "09_contour_detected.png"), img_contour)

# ============================================================
# 4. Mengurutkan 4 titik sudut
# ============================================================
print("\n--- 4. Mengurutkan Titik Sudut ---")

def order_points(pts):
    """
    Mengurutkan 4 titik: top-left, top-right, bottom-right, bottom-left.
    Urutan ini penting agar homography benar.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    # Top-left memiliki sum (x+y) terkecil
    # Bottom-right memiliki sum (x+y) terbesar
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # Top-right memiliki diff (y-x) terkecil
    # Bottom-left memiliki diff (y-x) terbesar
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    
    return rect

if doc_contour is not None:
    # Reshape dari (4,1,2) ke (4,2) lalu urutkan
    pts_found = doc_contour.reshape(4, 2).astype(np.float32)
    pts_ordered = order_points(pts_found)
    
    labels = ['TL', 'TR', 'BR', 'BL']
    for lbl, pt in zip(labels, pts_ordered):
        print(f"  {lbl}: ({pt[0]:.0f}, {pt[1]:.0f})")

# ============================================================
# 5. Menghitung dimensi output dan Homography
# ============================================================
print("\n--- 5. Homography + Warp ---")

if doc_contour is not None:
    # Hitung lebar dan tinggi dokumen asli
    # Lebar = max(jarak TL-TR, jarak BL-BR)
    width_top = np.linalg.norm(pts_ordered[1] - pts_ordered[0])
    width_bot = np.linalg.norm(pts_ordered[2] - pts_ordered[3])
    max_width = int(max(width_top, width_bot))
    
    # Tinggi = max(jarak TL-BL, jarak TR-BR)
    height_left = np.linalg.norm(pts_ordered[3] - pts_ordered[0])
    height_right = np.linalg.norm(pts_ordered[2] - pts_ordered[1])
    max_height = int(max(height_left, height_right))
    
    print(f"  Dimensi output: {max_width} x {max_height}")
    
    # Titik tujuan: persegi panjang sempurna
    pts_target = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])
    
    # Hitung homography dari miring → datar
    H_correct = cv2.getPerspectiveTransform(pts_ordered, pts_target)
    
    # Warp gambar
    doc_corrected = cv2.warpPerspective(img_scene, H_correct,
                                         (max_width, max_height))
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "09_dokumen_koreksi.png"), doc_corrected)
    print(f"  Dokumen dikoreksi dan disimpan")

# ============================================================
# 6. Post-processing: enhance kontras
# ============================================================
print("\n--- 6. Post-processing ---")

if doc_contour is not None:
    # Konversi ke grayscale
    doc_gray = cv2.cvtColor(doc_corrected, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding untuk hasil seperti scan
    doc_thresh = cv2.adaptiveThreshold(doc_gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "09_dokumen_scan.png"), doc_thresh)
    print(f"  Scan effect diterapkan")

# ============================================================
# 7. Visualisasi pipeline lengkap
# ============================================================
print("\n--- 7. Visualisasi Pipeline ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0][0].imshow(cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB))
axes[0][0].set_title("1. Input (Dokumen Miring)")

axes[0][1].imshow(edges_dilated, cmap='gray')
axes[0][1].set_title("2. Edge Detection")

if doc_contour is not None:
    axes[0][2].imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
else:
    axes[0][2].imshow(cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB))
axes[0][2].set_title("3. Contour Detection")

if doc_contour is not None:
    axes[1][0].imshow(cv2.cvtColor(doc_corrected, cv2.COLOR_BGR2RGB))
    axes[1][0].set_title("4. Perspective Corrected")
    
    axes[1][1].imshow(doc_gray, cmap='gray')
    axes[1][1].set_title("5. Grayscale")
    
    axes[1][2].imshow(doc_thresh, cmap='gray')
    axes[1][2].set_title("6. Scan Effect")

for ax in axes.flat:
    ax.axis('off')

plt.suptitle("Pipeline Koreksi Perspektif Dokumen", fontsize=14)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "09_pipeline_dokumen.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Disimpan: {output_path}")

print("\n" + "=" * 60)
print("PERCOBAAN 09 SELESAI")
print("=" * 60)
