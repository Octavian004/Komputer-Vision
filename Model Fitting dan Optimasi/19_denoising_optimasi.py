"""
==========================================================================
PERCOBAAN 19: DENOISING DENGAN OPTIMASI
==========================================================================
Image denoising memanfaatkan konsep optimasi: meminimalkan fungsi energi
yang terdiri dari data fidelity term (kecocokan dengan data) dan
regularization term (kehalusan / prior).

Fungsi utama:
- cv2.fastNlMeansDenoising()       : Non-local means denoising (grayscale)
- cv2.fastNlMeansDenoisingColored() : Non-local means denoising (berwarna)
- Total Variation Denoising         : minimisasi TV + data fidelity
- cv2.bilateralFilter()           : filtering edge-preserving
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
print("PERCOBAAN 19: DENOISING DENGAN OPTIMASI")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Memuat dan menambah noise
# ============================================================
print("\n--- 1. Memuat Gambar dan Menambah Noise ---")

img_path = os.path.join(IMAGE_DIR, "clean_img.png")
if not os.path.exists(img_path):
    print("[ERROR] img tidak ditemukan. Jalankan download_image.py!"); exit()

clean = cv2.imread(img_path)
gray_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

# Tambah Gaussian noise
sigma_noise = 30
noise_gauss = np.random.normal(0, sigma_noise, clean.shape)
noisy = np.clip(clean.astype(np.float64) + noise_gauss, 0, 255).astype(np.uint8)
gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)

# Tambah Salt & Pepper noise
noisy_sp = clean.copy()
prob = 0.05
salt = np.random.random(clean.shape[:2]) < prob
pepper = np.random.random(clean.shape[:2]) < prob
noisy_sp[salt] = 255
noisy_sp[pepper] = 0

cv2.imwrite(os.path.join(OUTPUT_DIR, "19_noisy_gaussian.png"), noisy)
cv2.imwrite(os.path.join(OUTPUT_DIR, "19_noisy_sp.png"), noisy_sp)

print(f"  Gambar: {clean.shape}")
print(f"  Sigma noise (Gaussian): {sigma_noise}")
print(f"  Salt & Pepper prob: {prob}")

# ============================================================
# 2. Non-Local Means Denoising
# ============================================================
print("\n--- 2. Non-Local Means Denoising ---")

# cv2.fastNlMeansDenoising untuk grayscale
# h: filter strength (besar → lebih smooth)
# templateWindowSize: ukuran patch (harus ganjil)
# searchWindowSize: ukuran area pencarian (harus ganjil)

h_values = [5, 10, 20, 30]
nlm_results = {}

for h in h_values:
    denoised = cv2.fastNlMeansDenoising(gray_noisy, None, h, 7, 21)
    psnr = cv2.PSNR(gray_clean, denoised)
    nlm_results[h] = (denoised, psnr)
    print(f"  h={h:2d}: PSNR={psnr:.2f} dB")

# Colored version
denoised_color = cv2.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21)
psnr_color = cv2.PSNR(clean, denoised_color)
print(f"  Color (h=10): PSNR={psnr_color:.2f} dB")

cv2.imwrite(os.path.join(OUTPUT_DIR, "19_nlm_color.png"), denoised_color)

# ============================================================
# 3. Total Variation Denoising (manual)
# ============================================================
print("\n--- 3. Total Variation Denoising ---")

def total_variation_denoise(noisy_img, weight=0.1, n_iter=100):
    """
    Total Variation Denoising via gradient descent.
    Energi: E = ||u - f||² + λ * TV(u)
    TV(u) = sum |∇u|
    
    u: gambar denoised (yang dioptimasi)
    f: gambar noisy (observed)
    λ (weight): bobot regularisasi
    """
    u = noisy_img.astype(np.float64).copy()
    f = noisy_img.astype(np.float64)
    
    dt = 0.25  # step size (harus < 0.25 untuk stabilitas)
    
    for i in range(n_iter):
        # Hitung gradien TV (divergence of normalized gradient)
        # Forward difference
        dx = np.roll(u, -1, axis=1) - u
        dy = np.roll(u, -1, axis=0) - u
        
        # Magnitude gradient + epsilon (hindari pembagian nol)
        grad_mag = np.sqrt(dx**2 + dy**2 + 1e-8)
        
        # Normalized gradient
        nx = dx / grad_mag
        ny = dy / grad_mag
        
        # Divergence (backward difference of normalized gradient)
        div = (nx - np.roll(nx, 1, axis=1)) + (ny - np.roll(ny, 1, axis=0))
        
        # Update: gradient descent
        # dE/du = 2(u - f) - lambda * div(∇u/|∇u|)
        u = u + dt * (weight * div - 2.0 * (u - f))
        u = np.clip(u, 0, 255)
    
    return u.astype(np.uint8)

# Berbagai weight regularisasi
weights = [0.05, 0.1, 0.5, 1.0]
tv_results = {}

for w in weights:
    denoised_tv = total_variation_denoise(gray_noisy, weight=w, n_iter=50)
    psnr_tv = cv2.PSNR(gray_clean, denoised_tv)
    tv_results[w] = (denoised_tv, psnr_tv)
    print(f"  weight={w:.2f}: PSNR={psnr_tv:.2f} dB")

# ============================================================
# 4. Bilateral Filter
# ============================================================
print("\n--- 4. Bilateral Filter ---")

# cv2.bilateralFilter: edge-preserving smoothing
# d: diameter of pixel neighborhood
# sigmaColor: filter sigma in color space
# sigmaSpace: filter sigma in coordinate space

bilateral_params = [
    (5, 30, 30),
    (9, 50, 50),
    (9, 75, 75),
    (15, 100, 100),
]

bilateral_results = {}

for d, sc, ss in bilateral_params:
    denoised_bf = cv2.bilateralFilter(noisy, d, sc, ss)
    psnr_bf = cv2.PSNR(clean, denoised_bf)
    bilateral_results[(d, sc, ss)] = (denoised_bf, psnr_bf)
    print(f"  d={d}, sigmaColor={sc}, sigmaSpace={ss}: PSNR={psnr_bf:.2f} dB")

# ============================================================
# 5. Perbandingan metode denoising
# ============================================================
print("\n--- 5. Perbandingan Metode ---")

# Gaussian Blur
blurred = cv2.GaussianBlur(noisy, (5, 5), 0)
psnr_blur = cv2.PSNR(clean, blurred)

# Median Filter (baik untuk salt & pepper)
median = cv2.medianBlur(noisy, 5)
psnr_median = cv2.PSNR(clean, median)

# NLM terbaik
best_h = max(nlm_results, key=lambda h: nlm_results[h][1])
psnr_nlm_best = nlm_results[best_h][1]

# TV terbaik
best_w = max(tv_results, key=lambda w: tv_results[w][1])
psnr_tv_best = tv_results[best_w][1]

# Bilateral terbaik
best_bf = max(bilateral_results, key=lambda k: bilateral_results[k][1])
psnr_bf_best = bilateral_results[best_bf][1]

print(f"  Input PSNR: {cv2.PSNR(clean, noisy):.2f} dB")
print(f"  Gaussian Blur: {psnr_blur:.2f} dB")
print(f"  Median Filter: {psnr_median:.2f} dB")
print(f"  NLM (h={best_h}): {psnr_nlm_best:.2f} dB")
print(f"  TV (w={best_w}): {psnr_tv_best:.2f} dB")
print(f"  Bilateral {best_bf}: {psnr_bf_best:.2f} dB")

# ============================================================
# 6. Denoising untuk Salt & Pepper
# ============================================================
print("\n--- 6. Denoising Salt & Pepper ---")

median_sp = cv2.medianBlur(noisy_sp, 5)
psnr_median_sp = cv2.PSNR(clean, median_sp)

gauss_sp = cv2.GaussianBlur(noisy_sp, (5, 5), 0)
psnr_gauss_sp = cv2.PSNR(clean, gauss_sp)

nlm_sp = cv2.fastNlMeansDenoisingColored(noisy_sp, None, 10, 10, 7, 21)
psnr_nlm_sp = cv2.PSNR(clean, nlm_sp)

print(f"  Median Filter: {psnr_median_sp:.2f} dB")
print(f"  Gaussian Blur: {psnr_gauss_sp:.2f} dB")
print(f"  NLM: {psnr_nlm_sp:.2f} dB")
print(f"  → Median filter terbaik untuk S&P noise")

# ============================================================
# 7. Visualisasi gabungan
# ============================================================
print("\n--- 7. Visualisasi Gabungan ---")

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

axes[0, 0].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Clean")

axes[0, 1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"Noisy (σ={sigma_noise})\nPSNR={cv2.PSNR(clean, noisy):.1f}")

axes[0, 2].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"Gaussian Blur\nPSNR={psnr_blur:.1f}")

axes[0, 3].imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(f"Median Filter\nPSNR={psnr_median:.1f}")

# NLM results
for idx, h in enumerate([5, 10, 20, 30]):
    denoised_gray, psnr = nlm_results[h]
    axes[1, idx].imshow(denoised_gray, cmap='gray')
    axes[1, idx].set_title(f"NLM h={h}\nPSNR={psnr:.1f}")

axes[2, 0].imshow(cv2.cvtColor(denoised_color, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title(f"NLM Color\nPSNR={psnr_color:.1f}")

best_bf_img = bilateral_results[best_bf][0]
axes[2, 1].imshow(cv2.cvtColor(best_bf_img, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title(f"Bilateral {best_bf}\nPSNR={psnr_bf_best:.1f}")

best_tv_img = tv_results[best_w][0]
axes[2, 2].imshow(best_tv_img, cmap='gray')
axes[2, 2].set_title(f"TV w={best_w}\nPSNR={psnr_tv_best:.1f}")

# PSNR comparison bar
methods_psnr = {
    'Noisy': cv2.PSNR(clean, noisy),
    'Gauss': psnr_blur,
    'Median': psnr_median,
    'NLM': psnr_nlm_best,
    'TV': psnr_tv_best,
    'Bilateral': psnr_bf_best,
}
axes[2, 3].bar(methods_psnr.keys(), methods_psnr.values(), color='steelblue')
axes[2, 3].set_ylabel("PSNR (dB)")
axes[2, 3].set_title("PSNR Comparison")

for ax in axes.flat[:12]:
    if ax != axes[2, 3]:
        ax.axis('off')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "19_denoising_all.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Disimpan: {output_path}")

print("\n" + "=" * 60)
print("PERCOBAAN 19 SELESAI")
print("=" * 60)
