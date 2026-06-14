"""
==========================================================================
PERCOBAAN 05: RANSAC FITTING LINGKARAN
==========================================================================
Menerapkan RANSAC untuk fitting lingkaran di antara data dengan outlier.
Minimal 3 titik diperlukan untuk mendefinisikan sebuah lingkaran.

Tiga titik menentukan lingkaran:
- Center (cx, cy) dan radius r didapat dari sistem persamaan:
  (x1-cx)^2 + (y1-cy)^2 = r^2
  (x2-cx)^2 + (y2-cy)^2 = r^2
  (x3-cx)^2 + (y3-cy)^2 = r^2

Fungsi utama:
- np.linalg.solve()       : selesaikan sistem linear
- cv2.circle()            : gambar lingkaran
- cv2.minEnclosingCircle(): lingkaran minimum (perbandingan)
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
print("PERCOBAAN 05: RANSAC FITTING LINGKARAN")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Membuat data lingkaran dengan outlier
# ============================================================
print("\n--- 1. Data Lingkaran + Outlier ---")

# Parameter lingkaran sebenarnya
true_cx, true_cy, true_r = 250, 250, 120

# Membuat 150 inlier di sekitar lingkaran
n_inlier = 150
angles = np.random.uniform(0, 2 * np.pi, n_inlier)
noise_r = np.random.randn(n_inlier) * 5  # noise kecil pada radius
x_inlier = true_cx + (true_r + noise_r) * np.cos(angles)
y_inlier = true_cy + (true_r + noise_r) * np.sin(angles)

# Membuat 50 outlier acak
n_outlier = 50
x_outlier = np.random.uniform(20, 480, n_outlier)
y_outlier = np.random.uniform(20, 480, n_outlier)

# Gabungkan semua data
x_all = np.concatenate([x_inlier, x_outlier])
y_all = np.concatenate([y_inlier, y_outlier])
N = len(x_all)

print(f"  True circle: center=({true_cx}, {true_cy}), r={true_r}")
print(f"  Total titik: {N} (inlier={n_inlier}, outlier={n_outlier})")

# ============================================================
# 2. Fungsi: fitting lingkaran dari 3 titik
# ============================================================

def fit_circle_3pts(p1, p2, p3):
    """
    Menghitung lingkaran dari 3 titik.
    Menggunakan persamaan: x^2 + y^2 + Dx + Ey + F = 0
    → center = (-D/2, -E/2), r = sqrt(D^2/4 + E^2/4 - F)
    """
    # Menyusun sistem linear: [x, y, 1] * [D, E, F]^T = -[x^2 + y^2]
    A = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1]
    ])
    b = -np.array([
        p1[0]**2 + p1[1]**2,
        p2[0]**2 + p2[1]**2,
        p3[0]**2 + p3[1]**2
    ])
    
    try:
        params = np.linalg.solve(A, b)
        D, E, F = params
        cx = -D / 2
        cy = -E / 2
        r_sq = D**2/4 + E**2/4 - F
        if r_sq <= 0:
            return None
        r = np.sqrt(r_sq)
        return (cx, cy, r)
    except np.linalg.LinAlgError:
        return None

# ============================================================
# 3. RANSAC untuk lingkaran
# ============================================================
print("\n--- 3. RANSAC Fitting Lingkaran ---")

def ransac_fit_circle(x, y, n_iters=3000, threshold=8.0):
    """
    RANSAC untuk fitting lingkaran.
    Minimal 3 titik per sampel.
    """
    N = len(x)
    best_n_inliers = 0
    best_circle = None
    best_inliers = None
    
    for _ in range(n_iters):
        # Langkah 1: Pilih 3 titik acak
        idx = np.random.choice(N, 3, replace=False)
        p1 = (x[idx[0]], y[idx[0]])
        p2 = (x[idx[1]], y[idx[1]])
        p3 = (x[idx[2]], y[idx[2]])
        
        # Langkah 2: Fit lingkaran dari 3 titik
        result = fit_circle_3pts(p1, p2, p3)
        if result is None:
            continue
        cx, cy, r = result
        
        # Abaikan lingkaran yang terlalu besar atau kecil
        if r < 10 or r > 400:
            continue
        
        # Langkah 3: Hitung jarak semua titik ke lingkaran
        # Jarak = |sqrt((x-cx)^2 + (y-cy)^2) - r|
        distances = np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r)
        
        # Langkah 4: Hitung inlier
        inlier_mask = distances < threshold
        n_inliers = np.sum(inlier_mask)
        
        # Langkah 5: Simpan model terbaik
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_circle = (cx, cy, r)
            best_inliers = inlier_mask
    
    return best_circle, best_inliers

# Jalankan RANSAC
circle_ransac, inlier_mask = ransac_fit_circle(x_all, y_all, n_iters=5000, threshold=10.0)
print(f"  RANSAC: center=({circle_ransac[0]:.1f}, {circle_ransac[1]:.1f}), r={circle_ransac[2]:.1f}")
print(f"  True:   center=({true_cx}, {true_cy}), r={true_r}")
print(f"  Inlier ditemukan: {np.sum(inlier_mask)}/{N}")

# Error
err_cx = abs(circle_ransac[0] - true_cx)
err_cy = abs(circle_ransac[1] - true_cy)
err_r = abs(circle_ransac[2] - true_r)
print(f"  Error center: ({err_cx:.2f}, {err_cy:.2f}), Error r: {err_r:.2f}")

# ============================================================
# 4. Perbandingan dengan cv2.minEnclosingCircle
# ============================================================
print("\n--- 4. Perbandingan dengan minEnclosingCircle ---")

# minEnclosingCircle menggunakan SEMUA titik (termasuk outlier)
pts_all = np.column_stack([x_all, y_all]).astype(np.float32)
(cx_enc, cy_enc), r_enc = cv2.minEnclosingCircle(pts_all)
print(f"  minEnclosingCircle: center=({cx_enc:.1f}, {cy_enc:.1f}), r={r_enc:.1f}")
print(f"  (Terpengaruh outlier → radius terlalu besar)")

# ============================================================
# 5. Visualisasi pada gambar
# ============================================================
print("\n--- 5. Visualisasi ---")

# Buat gambar 500x500
img = np.ones((500, 500, 3), dtype=np.uint8) * 240

# Gambar titik inlier (biru) dan outlier (merah) berdasarkan RANSAC
for i in range(N):
    x_pt, y_pt = int(x_all[i]), int(y_all[i])
    if 0 <= x_pt < 500 and 0 <= y_pt < 500:
        if inlier_mask[i]:
            cv2.circle(img, (x_pt, y_pt), 3, (200, 100, 0), -1)  # biru
        else:
            cv2.circle(img, (x_pt, y_pt), 3, (0, 0, 200), -1)    # merah

# Gambar lingkaran true (hijau putus-putus - approx)
cv2.circle(img, (true_cx, true_cy), true_r, (0, 180, 0), 2)

# Gambar lingkaran RANSAC (biru)
cv2.circle(img, (int(circle_ransac[0]), int(circle_ransac[1])),
           int(circle_ransac[2]), (255, 0, 0), 2)

# Gambar lingkaran minEnclosing (merah)
cv2.circle(img, (int(cx_enc), int(cy_enc)), int(r_enc), (0, 0, 255), 2)

# Label
cv2.putText(img, "Hijau=True, Biru=RANSAC, Merah=MinEnclosing",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

output_path = os.path.join(OUTPUT_DIR, "05_ransac_lingkaran.png")
cv2.imwrite(output_path, img)
print(f"  Disimpan: {output_path}")

# ============================================================
# 6. Visualisasi matplotlib (lebih detail)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Plot kiri: semua titik + ground truth
ax1 = axes[0]
ax1.scatter(x_inlier, y_inlier, c='blue', s=10, alpha=0.5, label='True Inlier')
ax1.scatter(x_outlier, y_outlier, c='red', s=10, alpha=0.5, label='True Outlier')
theta = np.linspace(0, 2 * np.pi, 100)
ax1.plot(true_cx + true_r * np.cos(theta), true_cy + true_r * np.sin(theta),
         'g--', linewidth=2, label=f'True (r={true_r})')
ax1.set_title("Ground Truth")
ax1.legend()
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Plot kanan: hasil RANSAC
ax2 = axes[1]
ax2.scatter(x_all[inlier_mask], y_all[inlier_mask], c='blue', s=10, alpha=0.5, label='RANSAC Inlier')
ax2.scatter(x_all[~inlier_mask], y_all[~inlier_mask], c='red', s=10, alpha=0.5, label='RANSAC Outlier')
ax2.plot(circle_ransac[0] + circle_ransac[2] * np.cos(theta),
         circle_ransac[1] + circle_ransac[2] * np.sin(theta),
         'b-', linewidth=2, label=f'RANSAC (r={circle_ransac[2]:.1f})')
ax2.plot(true_cx + true_r * np.cos(theta), true_cy + true_r * np.sin(theta),
         'g--', linewidth=2, label=f'True (r={true_r})')
ax2.set_title("RANSAC Result")
ax2.legend()
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = os.path.join(OUTPUT_DIR, "05_ransac_lingkaran_plot.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Disimpan: {output_path2}")

print("\n" + "=" * 60)
print("PERCOBAAN 05 SELESAI")
print("=" * 60)
