"""
==========================================================================
PERCOBAAN 19: FILTER FREKUENSI
==========================================================================
Setelah gambar diubah ke domain frekuensi (DFT), kita bisa memfilter
komponen frekuensi tertentu:
- Low-Pass Filter (LPF): loloskan rendah, buang tinggi → blur
- High-Pass Filter (HPF): loloskan tinggi, buang rendah → edge
- Band-Pass Filter (BPF): loloskan range tertentu

Jenis filter: Ideal, Gaussian, Butterworth

Fungsi:
- np.fft.fft2() / np.fft.ifft2() → DFT / inverse DFT
- np.fft.fftshift() → pindahkan DC ke tengah
- cv2.dft() / cv2.idft() → alternatif OpenCV
- np.meshgrid() → buat grid jarak dari pusat
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

img = cv2.imread(os.path.join(IMAGE_DIR, "kota.jpg"), cv2.IMREAD_GRAYSCALE)
if img is None:
    print("[ERROR] Jalankan download_image.py!"); exit()
img = cv2.resize(img, (512, 512))

print("=" * 60)
print("PERCOBAAN 19: FILTER FREKUENSI")
print("=" * 60)

# ============================================================
# Fungsi bantu: buat distance matrix dari pusat
# ============================================================
def distance_from_center(rows, cols):
    """Buat matriks jarak setiap piksel dari pusat."""
    # Hitung koordinat baris dan kolom
    u = np.arange(rows)
    v = np.arange(cols)
    # Jarak dari pusat
    u_c = rows // 2
    v_c = cols // 2
    # Meshgrid
    U, V = np.meshgrid(v - v_c, u - u_c)
    # Euclidean distance
    D = np.sqrt(U**2 + V**2)
    return D

rows, cols = img.shape
D = distance_from_center(rows, cols)

# ============================================================
# 1. Ideal Low-Pass Filter (ILPF)
# ============================================================
print("\n--- 1. Ideal Low-Pass Filter ---")

# DFT
f_shift = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))

results_ilpf = []
for cutoff in [10, 30, 60]:
    # Buat mask: 1 jika D <= cutoff, 0 sebaliknya
    H = np.zeros((rows, cols), dtype=np.float32)
    H[D <= cutoff] = 1.0
    # Terapkan filter di domain frekuensi
    filtered = f_shift * H
    # Inverse DFT
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    result = np.clip(result, 0, 255).astype(np.uint8)
    results_ilpf.append((cutoff, result, H))
    print(f"  Cutoff D0={cutoff}: ringing effect terlihat")

# ============================================================
# 2. Gaussian Low-Pass Filter (GLPF)
# ============================================================
print("\n--- 2. Gaussian Low-Pass Filter ---")

results_glpf = []
for cutoff in [10, 30, 60]:
    # Gaussian: H = exp(-D^2 / 2*D0^2)
    H = np.exp(-(D**2) / (2 * (cutoff**2)))
    # Terapkan filter
    filtered = f_shift * H
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    result = np.clip(result, 0, 255).astype(np.uint8)
    results_glpf.append((cutoff, result, H))
    print(f"  Cutoff D0={cutoff}: smooth tanpa ringing")

# ============================================================
# 3. Butterworth Low-Pass Filter
# ============================================================
print("\n--- 3. Butterworth Low-Pass Filter ---")

results_blpf = []
n_order = 2  # orde filter
for cutoff in [10, 30, 60]:
    # Butterworth: H = 1 / (1 + (D/D0)^(2n))
    H = 1 / (1 + (D / cutoff)**(2 * n_order))
    # Terapkan filter
    filtered = f_shift * H
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    result = np.clip(result, 0, 255).astype(np.uint8)
    results_blpf.append((cutoff, result, H))
    print(f"  Cutoff D0={cutoff}, n={n_order}")

# ============================================================
# 4. Ideal High-Pass Filter (IHPF)
# ============================================================
print("\n--- 4. Ideal High-Pass Filter ---")

results_ihpf = []
for cutoff in [10, 30, 60]:
    # HPF = 1 - LPF
    H = np.ones((rows, cols), dtype=np.float32)
    H[D <= cutoff] = 0.0
    filtered = f_shift * H
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results_ihpf.append((cutoff, result, H))
    print(f"  Cutoff D0={cutoff}: edges detected")

# ============================================================
# 5. Gaussian High-Pass Filter (GHPF)
# ============================================================
print("\n--- 5. Gaussian High-Pass Filter ---")

results_ghpf = []
for cutoff in [10, 30, 60]:
    # GHPF = 1 - GLPF
    H = 1 - np.exp(-(D**2) / (2 * (cutoff**2)))
    filtered = f_shift * H
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results_ghpf.append((cutoff, result, H))
    print(f"  Cutoff D0={cutoff}: smooth edge extraction")

# ============================================================
# 6. Band-Pass Filter
# ============================================================
print("\n--- 6. Band-Pass Filter ---")

results_bp = []
bands = [(10, 50), (20, 80), (40, 120)]
for lo, hi in bands:
    # Band-pass: loloskan frekuensi antara lo dan hi
    H = np.zeros((rows, cols), dtype=np.float32)
    H[(D >= lo) & (D <= hi)] = 1.0
    filtered = f_shift * H
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results_bp.append((lo, hi, result, H))
    print(f"  Band [{lo}, {hi}]")

# ============================================================
# 7. Band-Reject (Notch) Filter
# ============================================================
print("\n--- 7. Band-Reject Filter ---")

lo_r, hi_r = 20, 60
# Band-reject: blok frekuensi antara lo dan hi
H_reject = np.ones((rows, cols), dtype=np.float32)
H_reject[(D >= lo_r) & (D <= hi_r)] = 0.0
filtered_reject = f_shift * H_reject
result_reject = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_reject)))
result_reject = np.clip(result_reject, 0, 255).astype(np.uint8)
print(f"  Band-reject [{lo_r}, {hi_r}]")

# ============================================================
# 8. Visualisasi — Low-Pass Comparison
# ============================================================
fig1, axes1 = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: ILPF
axes1[0, 0].imshow(img, cmap='gray')
axes1[0, 0].set_title("Original"); axes1[0, 0].axis("off")
for idx, (c, res, H) in enumerate(results_ilpf):
    axes1[0, idx + 1].imshow(res, cmap='gray')
    axes1[0, idx + 1].set_title(f"Ideal LPF D0={c}"); axes1[0, idx + 1].axis("off")

# Baris 2: GLPF
axes1[1, 0].imshow(img, cmap='gray')
axes1[1, 0].set_title("Original"); axes1[1, 0].axis("off")
for idx, (c, res, H) in enumerate(results_glpf):
    axes1[1, idx + 1].imshow(res, cmap='gray')
    axes1[1, idx + 1].set_title(f"Gaussian LPF D0={c}"); axes1[1, idx + 1].axis("off")

# Baris 3: BLPF
axes1[2, 0].imshow(img, cmap='gray')
axes1[2, 0].set_title("Original"); axes1[2, 0].axis("off")
for idx, (c, res, H) in enumerate(results_blpf):
    axes1[2, idx + 1].imshow(res, cmap='gray')
    axes1[2, idx + 1].set_title(f"Butterworth LPF D0={c}"); axes1[2, idx + 1].axis("off")

plt.suptitle("Low-Pass Filters: Ideal vs Gaussian vs Butterworth", fontsize=16, fontweight="bold")
plt.tight_layout()

path1 = os.path.join(OUTPUT_DIR, "19_lpf_comparison.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[OUTPUT] {path1}")

# ============================================================
# 9. Visualisasi — High-Pass dan Band Filters
# ============================================================
fig2, axes2 = plt.subplots(3, 4, figsize=(20, 15))

# Baris 1: IHPF
axes2[0, 0].imshow(img, cmap='gray')
axes2[0, 0].set_title("Original"); axes2[0, 0].axis("off")
for idx, (c, res, H) in enumerate(results_ihpf):
    axes2[0, idx + 1].imshow(res, cmap='gray')
    axes2[0, idx + 1].set_title(f"Ideal HPF D0={c}"); axes2[0, idx + 1].axis("off")

# Baris 2: GHPF
axes2[1, 0].imshow(img, cmap='gray')
axes2[1, 0].set_title("Original"); axes2[1, 0].axis("off")
for idx, (c, res, H) in enumerate(results_ghpf):
    axes2[1, idx + 1].imshow(res, cmap='gray')
    axes2[1, idx + 1].set_title(f"Gaussian HPF D0={c}"); axes2[1, idx + 1].axis("off")

# Baris 3: Band
axes2[2, 0].imshow(img, cmap='gray')
axes2[2, 0].set_title("Original"); axes2[2, 0].axis("off")
for idx, (lo, hi, res, H) in enumerate(results_bp):
    axes2[2, idx + 1].imshow(res, cmap='gray')
    axes2[2, idx + 1].set_title(f"Band-Pass [{lo},{hi}]"); axes2[2, idx + 1].axis("off")

plt.suptitle("High-Pass dan Band-Pass Filters", fontsize=16, fontweight="bold")
plt.tight_layout()

path2 = os.path.join(OUTPUT_DIR, "19_hpf_band_comparison.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.show()
print(f"[OUTPUT] {path2}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 19")
print("=" * 60)
print("""
1. Low-Pass Filter: loloskan frekuensi rendah → blur/smoothing
2. High-Pass Filter: loloskan frekuensi tinggi → edge detection
3. Band-Pass Filter: loloskan range frekuensi tertentu
4. Band-Reject Filter: blok range frekuensi tertentu
5. Ideal Filter → ringing artifact, transisi tajam
6. Gaussian Filter → transisi smooth tanpa ringing
7. Butterworth Filter → kompromi antara ideal dan Gaussian
8. D0 kecil = filter ketat, D0 besar = filter longgar
""")
