"""
==========================================================================
PERCOBAAN 04: RANSAC FITTING GARIS
==========================================================================
RANSAC (Random Sample Consensus) adalah algoritma robust untuk fitting
model yang tahan terhadap outlier. Berbeda dengan Least Squares yang
mudah terpengaruh outlier, RANSAC memilih subset acak, membuat model,
lalu menghitung seberapa banyak titik yang cocok (inlier).

Algoritma RANSAC untuk garis:
1. Pilih 2 titik acak → buat garis
2. Hitung jarak semua titik ke garis
3. Hitung jumlah inlier (jarak < threshold)
4. Ulangi N kali, pilih model dengan inlier terbanyak
5. Re-fit menggunakan semua inlier terbaik

Fungsi utama:
- cv2.fitLine()           : fit garis (perbandingan)
- np.linalg.lstsq()      : re-fit garis dari inlier
- np.random.choice()     : sampling acak
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
print("PERCOBAAN 04: RANSAC FITTING GARIS")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Membuat data dengan BANYAK outlier
# ============================================================
print("\n--- 1. Membuat Data dengan Outlier ---")

# Inlier: 100 titik di sekitar y = 1.5x + 20
n_inlier = 100
x_inlier = np.random.uniform(0, 100, n_inlier)
y_inlier = 1.5 * x_inlier + 20 + np.random.randn(n_inlier) * 5

# Outlier: 40 titik acak di area gambar
n_outlier = 40
x_outlier = np.random.uniform(0, 100, n_outlier)
y_outlier = np.random.uniform(0, 200, n_outlier)

# Gabungkan semua data
x_all = np.concatenate([x_inlier, x_outlier])
y_all = np.concatenate([y_inlier, y_outlier])
N = len(x_all)

# Label: 0=inlier, 1=outlier (ground truth)
labels_gt = np.concatenate([np.zeros(n_inlier), np.ones(n_outlier)])

print(f"  Total titik: {N}")
print(f"  Inlier: {n_inlier}, Outlier: {n_outlier}")
print(f"  Rasio outlier: {n_outlier/N*100:.1f}%")

# ============================================================
# 2. OLS — terpengaruh outlier
# ============================================================
print("\n--- 2. OLS (Terpengaruh Outlier) ---")

A = np.vstack([x_all, np.ones(N)]).T
params_ols, _, _, _ = np.linalg.lstsq(A, y_all, rcond=None)
print(f"  OLS: y = {params_ols[0]:.4f}x + {params_ols[1]:.4f}")
print(f"  (True: y = 1.5x + 20)")

# ============================================================
# 3. RANSAC Manual — implementasi dari nol
# ============================================================
print("\n--- 3. RANSAC Manual ---")

def ransac_fit_line(x, y, n_iters=1000, threshold=10.0):
    """
    Implementasi RANSAC untuk fitting garis.
    
    Parameters:
    - x, y: koordinat titik
    - n_iters: jumlah iterasi maksimum
    - threshold: jarak maksimum titik ke garis untuk dianggap inlier
    
    Returns:
    - best_params: [slope, intercept] model terbaik
    - best_inliers: mask boolean inlier terbaik
    """
    N = len(x)
    best_n_inliers = 0
    best_params = None
    best_inliers = None
    
    for i in range(n_iters):
        # Langkah 1: Pilih 2 titik acak
        idx = np.random.choice(N, 2, replace=False)
        x1, y1 = x[idx[0]], y[idx[0]]
        x2, y2 = x[idx[1]], y[idx[1]]
        
        # Hindari garis vertikal
        if abs(x2 - x1) < 1e-10:
            continue
        
        # Langkah 2: Hitung parameter garis dari 2 titik
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Langkah 3: Hitung jarak orthogonal semua titik ke garis
        # Garis: slope*x - y + intercept = 0
        distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
        
        # Langkah 4: Hitung inlier (titik dengan jarak < threshold)
        inlier_mask = distances < threshold
        n_inliers = np.sum(inlier_mask)
        
        # Langkah 5: Simpan model terbaik
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inliers = inlier_mask
            best_params = [slope, intercept]
    
    # Langkah 6: Re-fit menggunakan SEMUA inlier dari model terbaik
    if best_inliers is not None and np.sum(best_inliers) >= 2:
        A_inlier = np.vstack([x[best_inliers], np.ones(np.sum(best_inliers))]).T
        params_refit, _, _, _ = np.linalg.lstsq(A_inlier, y[best_inliers], rcond=None)
        best_params = params_refit.tolist()
    
    return best_params, best_inliers

# Jalankan RANSAC
params_ransac, inlier_mask = ransac_fit_line(x_all, y_all, n_iters=2000, threshold=10.0)
print(f"  RANSAC: y = {params_ransac[0]:.4f}x + {params_ransac[1]:.4f}")
print(f"  Inlier ditemukan: {np.sum(inlier_mask)}/{N}")

# ============================================================
# 4. Evaluasi akurasi RANSAC
# ============================================================
print("\n--- 4. Evaluasi ---")

# Hitung berapa inlier yang benar-benar inlier (ground truth)
true_positives = np.sum(inlier_mask[:n_inlier])  # inlier asli yang terdeteksi
false_positives = np.sum(inlier_mask[n_inlier:])  # outlier yang salah terdeteksi
precision = true_positives / np.sum(inlier_mask) if np.sum(inlier_mask) > 0 else 0
recall = true_positives / n_inlier

print(f"  True Positives: {true_positives}/{n_inlier}")
print(f"  False Positives: {false_positives}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")

# Error terhadap model sebenarnya
err_ols_slope = abs(params_ols[0] - 1.5)
err_ransac_slope = abs(params_ransac[0] - 1.5)
print(f"  Error slope OLS: {err_ols_slope:.4f}")
print(f"  Error slope RANSAC: {err_ransac_slope:.4f}")

# ============================================================
# 5. Visualisasi
# ============================================================
print("\n--- 5. Visualisasi ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot kiri: data mentah + OLS
ax1 = axes[0]
ax1.scatter(x_inlier, y_inlier, c='blue', s=15, alpha=0.6, label=f'Inlier ({n_inlier})')
ax1.scatter(x_outlier, y_outlier, c='red', s=15, alpha=0.6, label=f'Outlier ({n_outlier})')
x_line = np.array([0, 100])
ax1.plot(x_line, 1.5 * x_line + 20, 'g--', linewidth=2, label='True')
ax1.plot(x_line, params_ols[0] * x_line + params_ols[1], 'r-', linewidth=2, label='OLS')
ax1.set_title("OLS (Terpengaruh Outlier)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot tengah: RANSAC result
ax2 = axes[1]
ax2.scatter(x_all[inlier_mask], y_all[inlier_mask], c='blue', s=15, alpha=0.6, label='Inlier (RANSAC)')
ax2.scatter(x_all[~inlier_mask], y_all[~inlier_mask], c='red', s=15, alpha=0.6, label='Outlier (RANSAC)')
ax2.plot(x_line, 1.5 * x_line + 20, 'g--', linewidth=2, label='True')
ax2.plot(x_line, params_ransac[0] * x_line + params_ransac[1], 'b-', linewidth=2, label='RANSAC')
ax2.set_title("RANSAC (Robust)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot kanan: perbandingan langsung
ax3 = axes[2]
ax3.scatter(x_all, y_all, c='gray', s=10, alpha=0.4)
ax3.plot(x_line, 1.5 * x_line + 20, 'g--', linewidth=2, label='True: y=1.5x+20')
ax3.plot(x_line, params_ols[0] * x_line + params_ols[1], 'r-', linewidth=2,
         label=f'OLS: y={params_ols[0]:.2f}x+{params_ols[1]:.2f}')
ax3.plot(x_line, params_ransac[0] * x_line + params_ransac[1], 'b-', linewidth=2,
         label=f'RANSAC: y={params_ransac[0]:.2f}x+{params_ransac[1]:.2f}')
ax3.set_title("Perbandingan OLS vs RANSAC")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "04_ransac_garis.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Disimpan: {output_path}")

# ============================================================
# 6. Pengaruh jumlah iterasi terhadap kualitas
# ============================================================
print("\n--- 6. Pengaruh Jumlah Iterasi ---")

for n_iter in [10, 50, 100, 500, 1000, 5000]:
    p, mask = ransac_fit_line(x_all, y_all, n_iters=n_iter, threshold=10.0)
    err = abs(p[0] - 1.5) if p else float('inf')
    inl = np.sum(mask) if mask is not None else 0
    print(f"  Iter={n_iter:5d}: slope={p[0]:.4f}, error={err:.4f}, inlier={inl}")

# ============================================================
# 7. Formula jumlah iterasi minimum
# N = log(1-p) / log(1-w^n)
# ============================================================
print("\n--- 7. Estimasi Iterasi Minimum ---")

p_success = 0.99  # probabilitas keberhasilan
w = n_inlier / N  # rasio inlier
n_min = 2  # titik minimal untuk garis

N_min_iters = np.log(1 - p_success) / np.log(1 - w**n_min)
print(f"  Rasio inlier (w): {w:.4f}")
print(f"  Prob sukses (p): {p_success}")
print(f"  Iterasi minimum: {N_min_iters:.0f}")

print("\n" + "=" * 60)
print("PERCOBAAN 04 SELESAI")
print("=" * 60)
