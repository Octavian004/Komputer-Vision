"""
==========================================================================
PERCOBAAN 03: TOTAL LEAST SQUARES (TLS)
==========================================================================
TLS meminimalkan jarak orthogonal (tegak lurus) titik ke garis/model,
bukan hanya residual vertikal seperti OLS. Lebih tepat jika kedua
variabel (x dan y) mengandung noise.

Metode: SVD (Singular Value Decomposition)
- Dekomposisi A = U S V^T
- Right singular vector terakhir = normal bidang terbaik

Fungsi utama:
- np.linalg.svd()        : Singular Value Decomposition
- cv2.fitLine()           : fit garis ke titik (menggunakan jarak ortogonal)
- np.linalg.lstsq()      : OLS untuk perbandingan
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 03: TOTAL LEAST SQUARES (TLS)")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Membuat data sintetis di mana KEDUA variabel ada noise
# Model sebenarnya: y = 2x + 3
# ============================================================
print("\n--- 1. Data dengan Noise pada X dan Y ---")

N = 100

# Titik-titik ideal di sepanjang garis y = 2x + 3
t = np.linspace(0, 5, N)
x_true = t
y_true = 2 * t + 3

# Menambahkan noise pada KEDUA sumbu (x dan y)
x_noise = np.random.randn(N) * 0.8
y_noise = np.random.randn(N) * 0.8
x_data = x_true + x_noise
y_data = y_true + y_noise

print(f"  Jumlah titik: {N}")
print(f"  True model: y = 2x + 3")
print(f"  Noise std (x): 0.8, Noise std (y): 0.8")

# ============================================================
# 2. OLS Fitting (hanya residual vertikal)
# ============================================================
print("\n--- 2. OLS Fitting ---")

# Matrix desain untuk OLS
A_ols = np.vstack([x_data, np.ones(N)]).T

# Solusi least squares standar
params_ols, _, _, _ = np.linalg.lstsq(A_ols, y_data, rcond=None)
print(f"  OLS: y = {params_ols[0]:.4f}x + {params_ols[1]:.4f}")

# ============================================================
# 3. TLS Fitting menggunakan SVD
# Prinsip: titik-titik disusun dalam matrix [x-mean_x, y-mean_y]
# SVD menemukan arah dengan variansi minimum (= normal garis)
# ============================================================
print("\n--- 3. TLS Fitting via SVD ---")

# Menghitung centroid (rata-rata) dari data
mean_x = np.mean(x_data)
mean_y = np.mean(y_data)

# Memusatkan data (subtract mean)
X_centered = np.column_stack([x_data - mean_x, y_data - mean_y])

# SVD: U, S, Vt = svd(X_centered)
# Vt[-1] (singular vector terakhir) = arah normal garis
U, S, Vt = np.linalg.svd(X_centered)

# Normal garis: [a, b] = Vt[-1]
normal = Vt[-1]
a_tls, b_tls = normal[0], normal[1]

# Mengubah ke bentuk y = mx + c
# ax + by = 0 (centered) → y = -(a/b)x
# Kembali ke koordinat asli: y - mean_y = -(a/b)(x - mean_x)
slope_tls = -a_tls / b_tls
intercept_tls = mean_y - slope_tls * mean_x
print(f"  TLS: y = {slope_tls:.4f}x + {intercept_tls:.4f}")

# ============================================================
# 4. cv2.fitLine() — implementasi OpenCV untuk line fitting
# ============================================================
print("\n--- 4. cv2.fitLine() ---")

# Menyiapkan titik untuk cv2.fitLine (format Nx1x2 float32)
points_cv = np.column_stack([x_data, y_data]).reshape(-1, 1, 2).astype(np.float32)

# cv2.fitLine(points, distType, param, reps, aeps)
# distType: cv2.DIST_L2 = least squares (seperti TLS versi OpenCV)
# Mengembalikan [vx, vy, x0, y0] — arah vektor dan titik pada garis
line_params = cv2.fitLine(points_cv, cv2.DIST_L2, 0, 0.01, 0.01)
vx, vy, x0, y0 = line_params.flatten()

# Konversi ke y = mx + c
slope_cv = vy / vx
intercept_cv = y0 - slope_cv * x0
print(f"  cv2.fitLine: y = {slope_cv:.4f}x + {intercept_cv:.4f}")
print(f"  Direction vector: ({vx:.4f}, {vy:.4f})")

# ============================================================
# 5. Menghitung jarak orthogonal ke garis
# ============================================================
print("\n--- 5. Jarak Orthogonal ---")

# Fungsi menghitung jarak orthogonal titik ke garis ax + by + c = 0
def orthogonal_distance(x_pts, y_pts, slope, intercept):
    """Menghitung jarak tegak lurus ke garis y = slope*x + intercept"""
    # Bentuk umum: slope*x - y + intercept = 0
    # Jarak = |slope*x - y + intercept| / sqrt(slope^2 + 1)
    a = slope
    b = -1
    c = intercept
    distances = np.abs(a * x_pts + b * y_pts + c) / np.sqrt(a**2 + b**2)
    return distances

# Jarak orthogonal menggunakan garis OLS
dist_ols = orthogonal_distance(x_data, y_data, params_ols[0], params_ols[1])

# Jarak orthogonal menggunakan garis TLS
dist_tls = orthogonal_distance(x_data, y_data, slope_tls, intercept_tls)

print(f"  Mean ortho dist (OLS): {dist_ols.mean():.4f}")
print(f"  Mean ortho dist (TLS): {dist_tls.mean():.4f}")
print(f"  Sum ortho dist^2 (OLS): {np.sum(dist_ols**2):.4f}")
print(f"  Sum ortho dist^2 (TLS): {np.sum(dist_tls**2):.4f}")
print(f"  TLS ortho distance lebih kecil: {np.sum(dist_tls**2) < np.sum(dist_ols**2)}")

# ============================================================
# 6. Visualisasi perbandingan OLS vs TLS
# ============================================================
print("\n--- 6. Visualisasi ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot kiri: OLS vs TLS fitting
ax1 = axes[0]
ax1.scatter(x_data, y_data, c='gray', alpha=0.5, s=20, label='Data (noisy x & y)')
x_line = np.array([-1, 7])
ax1.plot(x_line, 2 * x_line + 3, 'g--', linewidth=2, label='True: y=2x+3')
ax1.plot(x_line, params_ols[0] * x_line + params_ols[1], 'r-', linewidth=2,
         label=f'OLS: y={params_ols[0]:.2f}x+{params_ols[1]:.2f}')
ax1.plot(x_line, slope_tls * x_line + intercept_tls, 'b-', linewidth=2,
         label=f'TLS: y={slope_tls:.2f}x+{intercept_tls:.2f}')
ax1.set_title("OLS vs TLS Fitting")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 7)

# Plot tengah: residual vertikal vs orthogonal
ax2 = axes[1]
residual_vert_ols = np.abs(y_data - (params_ols[0] * x_data + params_ols[1]))
ax2.bar(np.arange(N) - 0.2, dist_ols[:50], width=0.4, alpha=0.6, color='red', label='Ortho OLS')
ax2.bar(np.arange(N)[:50] + 0.2, dist_tls[:50], width=0.4, alpha=0.6, color='blue', label='Ortho TLS')
ax2.set_title("Jarak Orthogonal (50 titik pertama)")
ax2.set_xlabel("Index titik")
ax2.set_ylabel("Jarak")
ax2.legend()
ax2.set_xlim(-1, 50)

# Plot kanan: ilustrasi perbedaan residual
ax3 = axes[2]
# Gambar beberapa titik dengan garis residual
idx_show = np.arange(0, N, 10)  # tampilkan setiap 10 titik
ax3.scatter(x_data, y_data, c='gray', alpha=0.3, s=15)
ax3.plot(x_line, slope_tls * x_line + intercept_tls, 'b-', linewidth=2, label='TLS')
# Garis tegak lurus dari titik ke garis TLS
for i in idx_show:
    px, py = x_data[i], y_data[i]
    # Proyeksi titik ke garis: y = slope*x + intercept
    # Titik terdekat pada garis secara orthogonal
    m = slope_tls
    c = intercept_tls
    x_proj = (px + m * (py - c)) / (1 + m**2)
    y_proj = m * x_proj + c
    ax3.plot([px, x_proj], [py, y_proj], 'b-', alpha=0.5, linewidth=0.8)
ax3.set_title("Jarak Orthogonal (TLS)")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "03_total_least_squares.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Disimpan: {output_path}")

# ============================================================
# 7. Perbandingan distType pada cv2.fitLine
# ============================================================
print("\n--- 7. Variasi distType pada cv2.fitLine ---")

# Menambahkan outlier untuk melihat efek distType
x_outlier = np.append(x_data, [2.0, 3.0, 4.0])
y_outlier = np.append(y_data, [25.0, 28.0, 30.0])  # outlier jauh dari garis
pts_out = np.column_stack([x_outlier, y_outlier]).reshape(-1, 1, 2).astype(np.float32)

# Mencoba berbagai distType:
dist_types = {
    'DIST_L2': cv2.DIST_L2,       # Least squares (sensitif outlier)
    'DIST_L1': cv2.DIST_L1,       # Lebih robust (median-like)
    'DIST_L12': cv2.DIST_L12,     # Kombinasi L1 dan L2
    'DIST_HUBER': cv2.DIST_HUBER, # Huber loss (robust)
}

for name, dist in dist_types.items():
    # cv2.fitLine dengan distType berbeda
    line = cv2.fitLine(pts_out, dist, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    m = vy / vx
    c = y0 - m * x0
    print(f"  {name:12s}: y = {m:.4f}x + {c:.4f}")

print("\n  DIST_L2 sensitif outlier, DIST_HUBER lebih robust")

print("\n" + "=" * 60)
print("PERCOBAAN 03 SELESAI")
print("=" * 60)
