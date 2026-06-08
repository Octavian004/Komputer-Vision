"""
==========================================================================
PERCOBAAN 06: THRESHOLDING OTSU DAN TRIANGLE
==========================================================================
Otsu dan Triangle menentukan threshold optimal secara otomatis.
- Otsu: meminimalkan varians intra-kelas (bimodal histogram)
- Triangle: garis dari puncak ke ujung histogram, cari jarak max

Fungsi:
- cv2.threshold(src, 0, 255, THRESH_BINARY + THRESH_OTSU)
- cv2.threshold(src, 0, 255, THRESH_BINARY + THRESH_TRIANGLE)
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

img = cv2.imread(os.path.join(IMAGE_DIR, "dokumen.jpg"))
if img is None:
    print("[ERROR] Jalankan download_image.py terlebih dahulu!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=" * 60)
print("PERCOBAAN 06: THRESHOLDING OTSU DAN TRIANGLE")
print("=" * 60)

# ============================================================
# 1. Otsu's Thresholding
# ============================================================
print("\n--- 1. Otsu's Thresholding ---")

# Nilai threshold di parameter ke-2 diabaikan (gunakan 0)
# Flag THRESH_OTSU menghitung T optimal otomatis
ret_otsu, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"  Otsu optimal T = {ret_otsu:.0f}")
white_pct = np.sum(otsu == 255) / otsu.size * 100
print(f"  Piksel putih: {white_pct:.1f}%")

# ============================================================
# 2. Triangle Thresholding
# ============================================================
print("\n--- 2. Triangle Thresholding ---")

ret_tri, triangle = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
print(f"  Triangle optimal T = {ret_tri:.0f}")
white_pct_t = np.sum(triangle == 255) / triangle.size * 100
print(f"  Piksel putih: {white_pct_t:.1f}%")

# ============================================================
# 3. Otsu dengan Pre-filter Gaussian Blur
# ============================================================
print("\n--- 3. Otsu + Gaussian Blur ---")

# Blur mengurangi noise → histogram lebih smooth → Otsu lebih akurat
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ret_otsu_blur, otsu_blur = cv2.threshold(blurred, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"  Otsu T tanpa blur: {ret_otsu:.0f}")
print(f"  Otsu T dengan blur: {ret_otsu_blur:.0f}")

# ============================================================
# 4. Implementasi Otsu Manual
# ============================================================
print("\n--- 4. Otsu Manual ---")

def otsu_manual(image):
    """Implementasi algoritma Otsu dari nol."""
    # Hitung histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    total = image.size
    # Normalisasi histogram
    prob = hist / total

    best_thresh = 0
    best_variance = 0

    # Coba semua threshold dari 0 sampai 255
    for t in range(256):
        # Kelas 0: piksel <= t
        w0 = prob[:t + 1].sum()
        # Kelas 1: piksel > t
        w1 = prob[t + 1:].sum()

        if w0 == 0 or w1 == 0:
            continue

        # Mean kelas 0 dan kelas 1
        mu0 = np.sum(np.arange(t + 1) * prob[:t + 1]) / w0
        mu1 = np.sum(np.arange(t + 1, 256) * prob[t + 1:]) / w1

        # Between-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh = t

    return best_thresh

t_manual = otsu_manual(gray)
print(f"  Otsu manual T = {t_manual}")
print(f"  Otsu OpenCV T = {ret_otsu:.0f}")
print(f"  Selisih: {abs(t_manual - ret_otsu):.0f}")

# ============================================================
# 5. Perbandingan pada Gambar Berbeda
# ============================================================
print("\n--- 5. Perbandingan pada Gambar Lain ---")

nature = cv2.imread(os.path.join(IMAGE_DIR, "nature.jpg"))
if nature is not None:
    gray_n = cv2.cvtColor(nature, cv2.COLOR_BGR2GRAY)
    ret_n_otsu, _ = cv2.threshold(gray_n, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret_n_tri, _ = cv2.threshold(gray_n, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    print(f"  Nature: Otsu T={ret_n_otsu:.0f}, Triangle T={ret_n_tri:.0f}")

buah = cv2.imread(os.path.join(IMAGE_DIR, "buah.jpg"))
if buah is not None:
    gray_b = cv2.cvtColor(buah, cv2.COLOR_BGR2GRAY)
    ret_b_otsu, _ = cv2.threshold(gray_b, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret_b_tri, _ = cv2.threshold(gray_b, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    print(f"  Buah:   Otsu T={ret_b_otsu:.0f}, Triangle T={ret_b_tri:.0f}")

# ============================================================
# 6. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Baris 1: Perbandingan
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(otsu, cmap='gray')
axes[0, 1].set_title(f"Otsu (T={ret_otsu:.0f})")
axes[0, 1].axis("off")

axes[0, 2].imshow(triangle, cmap='gray')
axes[0, 2].set_title(f"Triangle (T={ret_tri:.0f})")
axes[0, 2].axis("off")

axes[0, 3].imshow(otsu_blur, cmap='gray')
axes[0, 3].set_title(f"Otsu+Blur (T={ret_otsu_blur:.0f})")
axes[0, 3].axis("off")

# Baris 2: Histogram + threshold lines
axes[1, 0].hist(gray.ravel(), 256, [0, 256], color='steelblue', alpha=0.7)
axes[1, 0].axvline(x=ret_otsu, color='red', linestyle='--', linewidth=2, label=f'Otsu={ret_otsu:.0f}')
axes[1, 0].axvline(x=ret_tri, color='green', linestyle=':', linewidth=2, label=f'Triangle={ret_tri:.0f}')
axes[1, 0].set_title("Histogram + Threshold")
axes[1, 0].legend()

if nature is not None:
    axes[1, 1].hist(gray_n.ravel(), 256, [0, 256], color='steelblue', alpha=0.7)
    axes[1, 1].axvline(x=ret_n_otsu, color='red', linestyle='--', label=f'Otsu={ret_n_otsu:.0f}')
    axes[1, 1].axvline(x=ret_n_tri, color='green', linestyle=':', label=f'Tri={ret_n_tri:.0f}')
    axes[1, 1].set_title("Histogram Nature")
    axes[1, 1].legend()

    _, otsu_n = cv2.threshold(gray_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    axes[1, 2].imshow(otsu_n, cmap='gray')
    axes[1, 2].set_title(f"Nature Otsu")
    axes[1, 2].axis("off")

axes[1, 3].axis("off")

plt.suptitle("Percobaan 06: Otsu & Triangle Thresholding", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "06_otsu_triangle_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 06")
print("=" * 60)
print("""
1. Otsu: threshold otomatis, optimal untuk histogram bimodal
2. Triangle: metode alternatif, baik untuk histogram unimodal
3. Flag THRESH_OTSU/THRESH_TRIANGLE ditambahkan ke tipe threshold
4. Gaussian blur sebelum Otsu meningkatkan akurasi
5. Between-class variance σ²_B = w₀·w₁·(μ₀-μ₁)²
6. OpenCV mengembalikan nilai T optimal di return value
""")
