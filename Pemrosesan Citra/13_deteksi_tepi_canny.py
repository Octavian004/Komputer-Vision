"""
==========================================================================
PERCOBAAN 13: DETEKSI TEPI CANNY
==========================================================================
Canny edge detector adalah metode multi-tahap:
1. Gaussian blur (noise reduction)
2. Gradient (Sobel)
3. Non-maximum suppression (penipisan tepi)
4. Double threshold (kuat/lemah)
5. Hysteresis (tracking edge)

Fungsi:
- cv2.Canny(src, threshold1, threshold2, apertureSize, L2gradient)
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(os.path.join(IMAGE_DIR, "garis_tepi.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 13: DETEKSI TEPI CANNY")
print("=" * 60)

# ============================================================
# 1. Canny Dasar
# ============================================================
print("\n--- 1. Canny Dasar ---")

# threshold1 = low threshold, threshold2 = high threshold
edges = cv2.Canny(gray, 50, 150)
edge_count = np.sum(edges > 0)
print(f"  T_low=50, T_high=150: {edge_count} piksel tepi")

# ============================================================
# 2. Variasi Threshold
# ============================================================
print("\n--- 2. Variasi Threshold ---")

thresholds = [(10, 50), (30, 100), (50, 150), (100, 200), (150, 250)]
thresh_results = []

for t1, t2 in thresholds:
    result = cv2.Canny(gray, t1, t2)
    thresh_results.append(result)
    n_edges = np.sum(result > 0)
    print(f"  ({t1:3d}, {t2:3d}): {n_edges:6d} piksel tepi")

# ============================================================
# 3. Auto Canny (berdasarkan median)
# ============================================================
print("\n--- 3. Auto Canny ---")

def auto_canny(image, sigma=0.33):
    """Hitung threshold Canny otomatis berdasarkan median."""
    # Hitung median intensitas
    med = np.median(image)
    # Tentukan low dan high threshold
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(image, lower, upper), lower, upper

auto_edges, auto_lo, auto_hi = auto_canny(gray)
print(f"  Median: {np.median(gray):.0f}")
print(f"  Auto threshold: ({auto_lo}, {auto_hi})")
print(f"  Edge piksel: {np.sum(auto_edges > 0)}")

# ============================================================
# 4. Pengaruh Blur Sebelum Canny
# ============================================================
print("\n--- 4. Blur + Canny ---")

blur_sizes = [0, 3, 5, 7, 11]
blur_results = []

for ks in blur_sizes:
    if ks > 0:
        blurred = cv2.GaussianBlur(gray, (ks, ks), 0)
    else:
        blurred = gray
    result = cv2.Canny(blurred, 50, 150)
    blur_results.append(result)
    print(f"  blur ksize={ks:2d}: {np.sum(result > 0):6d} edge piksel")

# ============================================================
# 5. L1 vs L2 Gradient
# ============================================================
print("\n--- 5. L1 vs L2 Gradient ---")

# L1: |Gx| + |Gy| (default, lebih cepat)
canny_l1 = cv2.Canny(gray, 50, 150, L2gradient=False)
# L2: sqrt(Gx² + Gy²) (lebih akurat)
canny_l2 = cv2.Canny(gray, 50, 150, L2gradient=True)
diff = cv2.absdiff(canny_l1, canny_l2)
print(f"  L1: {np.sum(canny_l1 > 0)} piksel")
print(f"  L2: {np.sum(canny_l2 > 0)} piksel")
print(f"  Perbedaan: {np.sum(diff > 0)} piksel")

# ============================================================
# 6. Canny pada Gambar Berwarna
# ============================================================
print("\n--- 6. Canny Berwarna ---")

# Canny per channel
edges_b = cv2.Canny(img[:, :, 0], 50, 150)
edges_g = cv2.Canny(img[:, :, 1], 50, 150)
edges_r = cv2.Canny(img[:, :, 2], 50, 150)
# Gabungkan (OR) semua channel
edges_all = cv2.bitwise_or(cv2.bitwise_or(edges_b, edges_g), edges_r)
print(f"  Canny per channel-OR: {np.sum(edges_all > 0)} piksel")

# ============================================================
# 7. Overlay Edge pada Gambar Asli
# ============================================================
overlay = img.copy()
# Buat edge berwarna merah pada gambar asli
overlay[edges > 0] = [0, 0, 255]  # Merah di posisi edge

# ============================================================
# 8. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Variasi threshold
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

for i, ((t1, t2), res) in enumerate(zip(thresholds[:3], thresh_results[:3])):
    axes[0, i + 1].imshow(res, cmap='gray')
    axes[0, i + 1].set_title(f"({t1},{t2})")
    axes[0, i + 1].axis("off")

# Baris 2: Blur effect + auto
for i, (ks, res) in enumerate(zip([0, 5, 11],
                                   [blur_results[0], blur_results[2],
                                    blur_results[4]])):
    axes[1, i].imshow(res, cmap='gray')
    axes[1, i].set_title(f"blur={ks}")
    axes[1, i].axis("off")

axes[1, 3].imshow(auto_edges, cmap='gray')
axes[1, 3].set_title(f"Auto ({auto_lo},{auto_hi})")
axes[1, 3].axis("off")

# Baris 3: Color + overlay
axes[2, 0].imshow(edges_all, cmap='gray')
axes[2, 0].set_title("Multi-Channel")
axes[2, 0].axis("off")

axes[2, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("Edge Overlay")
axes[2, 1].axis("off")

axes[2, 2].imshow(canny_l2, cmap='gray')
axes[2, 2].set_title("L2 Gradient")
axes[2, 2].axis("off")

axes[2, 3].axis("off")

plt.suptitle("Percobaan 13: Deteksi Tepi Canny", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "13_canny_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 13")
print("=" * 60)
print("""
1. cv2.Canny(src, T_low, T_high) → multi-tahap edge detection
2. Gradient > T_high → edge kuat (pasti edge)
3. T_low < gradient < T_high → edge lemah (edge jika terhubung)
4. Gaussian blur sebelum Canny mengurangi false edges (noise)
5. Auto threshold: berdasarkan median intensitas gambar
6. L2gradient=True lebih akurat tapi sedikit lebih lambat
""")
