"""
==========================================================================
PERCOBAAN 14: DETEKSI TEPI LAPLACIAN
==========================================================================
Laplacian menggunakan turunan kedua untuk mendeteksi tepi.
Mendeteksi tepi di semua arah sekaligus (isotropic).
Zero-crossing pada Laplacian menandakan posisi tepi.

Fungsi:
- cv2.Laplacian(src, ddepth, ksize) → turunan kedua
- LoG (Laplacian of Gaussian) = blur lalu Laplacian
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
print("PERCOBAAN 14: DETEKSI TEPI LAPLACIAN")
print("=" * 60)

# ============================================================
# 1. Laplacian Dasar
# ============================================================
print("\n--- 1. Laplacian Dasar ---")

# ddepth=CV_64F untuk menyimpan nilai positif dan negatif
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
print(f"  Range: [{lap.min():.0f}, {lap.max():.0f}]")

# Konversi ke absolut untuk visualisasi
lap_abs = np.abs(lap).astype(np.uint8)

# ============================================================
# 2. Variasi ksize
# ============================================================
print("\n--- 2. Variasi ksize ---")

ksize_list = [1, 3, 5, 7]
ksize_results = []

for ks in ksize_list:
    result = cv2.Laplacian(gray, cv2.CV_64F, ksize=ks)
    norm = cv2.normalize(np.abs(result), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ksize_results.append(norm)
    print(f"  ksize={ks}: max={np.abs(result).max():.0f}")

# ============================================================
# 3. Laplacian of Gaussian (LoG)
# ============================================================
print("\n--- 3. Laplacian of Gaussian (LoG) ---")

# Tahap 1: Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
# Tahap 2: Laplacian
log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
log_abs = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
print(f"  LoG (σ=1.4, ksize=3): max={np.abs(log).max():.0f}")

# Variasi sigma untuk LoG
log_results = []
for sigma in [0.5, 1.0, 2.0, 4.0]:
    ks = int(6 * sigma + 1) | 1  # pastikan ganjil
    b = cv2.GaussianBlur(gray, (ks, ks), sigma)
    l = cv2.Laplacian(b, cv2.CV_64F, ksize=3)
    l_norm = cv2.normalize(np.abs(l), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    log_results.append(l_norm)
    print(f"  LoG σ={sigma}: tepi={np.sum(l_norm > 30)} piksel")

# ============================================================
# 4. Zero-Crossing Detection
# ============================================================
print("\n--- 4. Zero-Crossing ---")

def zero_crossing(laplacian):
    """Mendeteksi zero-crossing pada output Laplacian."""
    # Cek perubahan tanda antara piksel bertetangga
    zc = np.zeros(laplacian.shape, dtype=np.uint8)
    # Horizontal: cek kiri-kanan
    for j in range(1, laplacian.shape[0]):
        for i in range(1, laplacian.shape[1]):
            # Cek 4 tetangga
            if (laplacian[j, i] * laplacian[j, i-1] < 0 or
                laplacian[j, i] * laplacian[j-1, i] < 0):
                zc[j, i] = 255
    return zc

# Gunakan LoG lalu zero-crossing
zc_result = zero_crossing(log)
print(f"  Zero-crossing piksel: {np.sum(zc_result > 0)}")

# ============================================================
# 5. Perbandingan Laplacian vs Sobel vs Canny
# ============================================================
print("\n--- 5. Perbandingan ---")

# Sobel magnitude
sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.normalize(cv2.magnitude(sx, sy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Canny
canny = cv2.Canny(gray, 50, 150)

print(f"  Laplacian: {np.sum(lap_abs > 30)} piksel (threshold=30)")
print(f"  Sobel:     {np.sum(sobel_mag > 30)} piksel (threshold=30)")
print(f"  Canny:     {np.sum(canny > 0)} piksel")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: Dasar + ksize
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

for i, (ks, res) in enumerate(zip(ksize_list[:3], ksize_results[:3])):
    axes[0, i + 1].imshow(res, cmap='gray')
    axes[0, i + 1].set_title(f"Lap ksize={ks}")
    axes[0, i + 1].axis("off")

# Baris 2: LoG variasi + zero-crossing
for i, (sigma, res) in enumerate(zip([0.5, 1.0, 2.0], log_results[:3])):
    axes[1, i].imshow(res, cmap='gray')
    axes[1, i].set_title(f"LoG σ={sigma}")
    axes[1, i].axis("off")

axes[1, 3].imshow(zc_result, cmap='gray')
axes[1, 3].set_title("Zero-Crossing")
axes[1, 3].axis("off")

# Baris 3: Perbandingan
axes[2, 0].imshow(lap_abs, cmap='gray')
axes[2, 0].set_title("Laplacian")
axes[2, 0].axis("off")

axes[2, 1].imshow(sobel_mag, cmap='gray')
axes[2, 1].set_title("Sobel")
axes[2, 1].axis("off")

axes[2, 2].imshow(canny, cmap='gray')
axes[2, 2].set_title("Canny")
axes[2, 2].axis("off")

axes[2, 3].axis("off")

plt.suptitle("Percobaan 14: Deteksi Tepi Laplacian", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "14_laplacian_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 14")
print("=" * 60)
print("""
1. Laplacian = turunan kedua → ∇²f = d²f/dx² + d²f/dy²
2. Mendeteksi tepi di semua arah (isotropic)
3. Sensitif terhadap noise → gunakan LoG (Gaussian dulu)
4. Zero-crossing pada Laplacian = posisi tepi
5. ksize lebih besar → respon lebih smooth
6. Laplacian + Canny menghasilkan deteksi tepi paling lengkap
""")
