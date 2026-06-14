"""
==========================================================================
PERCOBAAN 10: IRLS (ITERATIVELY REWEIGHTED LEAST SQUARES)
==========================================================================
IRLS adalah metode robust fitting yang secara iteratif mengurangi
pengaruh outlier. Setiap iterasi:
1. Fit model menggunakan Weighted Least Squares
2. Hitung residual
3. Update bobot: outlier mendapat bobot kecil
4. Ulangi sampai konvergen

Robust loss functions:
- Huber: linear untuk residual besar (bounded influence)
- Tukey Bisquare: nol untuk residual sangat besar (hard rejection)
- Cauchy: menurun lambat (soft rejection)

Fungsi utama:
- np.linalg.lstsq()      : weighted least squares per iterasi
- cv2.fitLine()           : perbandingan (distType=DIST_HUBER)
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
print("PERCOBAAN 10: IRLS (ITERATIVELY REWEIGHTED LEAST SQUARES)")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Membuat data dengan outlier
# ============================================================
print("\n--- 1. Membuat Data ---")

# 80 inlier di sekitar y = 2x + 5
n_inlier = 80
x_in = np.linspace(0, 10, n_inlier)
y_in = 2 * x_in + 5 + np.random.randn(n_inlier) * 1.5

# 15 outlier
n_outlier = 15
x_out = np.random.uniform(1, 9, n_outlier)
y_out = np.random.uniform(25, 40, n_outlier)

x_all = np.concatenate([x_in, x_out])
y_all = np.concatenate([y_in, y_out])
N = len(x_all)

print(f"  Total: {N} titik (inlier={n_inlier}, outlier={n_outlier})")

# ============================================================
# 2. Fungsi weight berdasarkan robust loss
# ============================================================

def huber_weights(residuals, c=1.345):
    """
    Huber weight function.
    w = 1 jika |r| <= c, w = c/|r| jika |r| > c
    Parameter c=1.345 menghasilkan 95% efisiensi pada data normal.
    """
    abs_r = np.abs(residuals)
    weights = np.where(abs_r <= c, 1.0, c / abs_r)
    return weights

def tukey_weights(residuals, c=4.685):
    """
    Tukey Bisquare weight function.
    w = (1 - (r/c)^2)^2 jika |r| <= c, w = 0 jika |r| > c
    Hard rejection: outlier mendapat bobot NOL.
    """
    abs_r = np.abs(residuals)
    weights = np.where(abs_r <= c, (1 - (residuals / c)**2)**2, 0.0)
    return weights

def cauchy_weights(residuals, c=2.385):
    """
    Cauchy weight function.
    w = 1 / (1 + (r/c)^2)
    Soft rejection: bobot menurun tapi tidak pernah nol.
    """
    weights = 1.0 / (1.0 + (residuals / c)**2)
    return weights

# ============================================================
# 3. Implementasi IRLS
# ============================================================
print("\n--- 3. IRLS Implementation ---")

def irls_fit_line(x, y, weight_func, max_iter=50, tol=1e-6):
    """
    IRLS untuk fitting garis y = ax + b.
    
    Parameters:
    - x, y: data
    - weight_func: fungsi yang menghitung bobot dari residual
    - max_iter: iterasi maksimum
    - tol: toleransi konvergensi
    
    Returns:
    - params: [a, b]
    - weights_history: bobot di setiap iterasi
    - params_history: parameter di setiap iterasi
    """
    N = len(x)
    A = np.vstack([x, np.ones(N)]).T
    
    # Mulai dengan OLS (semua bobot = 1)
    weights = np.ones(N)
    params_history = []
    weights_history = []
    
    for iteration in range(max_iter):
        # Weighted Least Squares
        W = np.diag(weights)
        AtWA = A.T @ W @ A
        AtWy = A.T @ W @ y
        params = np.linalg.solve(AtWA, AtWy)
        params_history.append(params.copy())
        
        # Hitung residual
        residuals = y - (params[0] * x + params[1])
        
        # Estimasi skala (Median Absolute Deviation)
        # MAD lebih robust daripada std
        mad = np.median(np.abs(residuals - np.median(residuals)))
        sigma = mad / 0.6745  # konversi ke std equivalent
        if sigma < 1e-10:
            sigma = 1.0
        
        # Normalisasi residual
        std_residuals = residuals / sigma
        
        # Update bobot menggunakan robust weight function
        new_weights = weight_func(std_residuals)
        weights_history.append(new_weights.copy())
        
        # Cek konvergensi
        if iteration > 0 and np.max(np.abs(new_weights - weights)) < tol:
            print(f"    Konvergen pada iterasi {iteration + 1}")
            break
        
        weights = new_weights
    
    return params, weights_history, params_history

# ============================================================
# 4. Jalankan IRLS dengan berbagai weight functions
# ============================================================

# OLS (baseline)
A_base = np.vstack([x_all, np.ones(N)]).T
params_ols, _, _, _ = np.linalg.lstsq(A_base, y_all, rcond=None)
print(f"\n  OLS:    a={params_ols[0]:.4f}, b={params_ols[1]:.4f}")

# IRLS Huber
print("\n  IRLS Huber:")
params_huber, w_hist_h, p_hist_h = irls_fit_line(x_all, y_all, huber_weights)
print(f"    a={params_huber[0]:.4f}, b={params_huber[1]:.4f}")

# IRLS Tukey
print("\n  IRLS Tukey:")
params_tukey, w_hist_t, p_hist_t = irls_fit_line(x_all, y_all, tukey_weights)
print(f"    a={params_tukey[0]:.4f}, b={params_tukey[1]:.4f}")

# IRLS Cauchy
print("\n  IRLS Cauchy:")
params_cauchy, w_hist_c, p_hist_c = irls_fit_line(x_all, y_all, cauchy_weights)
print(f"    a={params_cauchy[0]:.4f}, b={params_cauchy[1]:.4f}")

print(f"\n  True: a=2.0, b=5.0")

# ============================================================
# 5. Visualisasi
# ============================================================
print("\n--- 5. Visualisasi ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Data + semua garis
x_line = np.array([0, 10])
ax = axes[0][0]
ax.scatter(x_in, y_in, c='blue', s=15, alpha=0.6, label='Inlier')
ax.scatter(x_out, y_out, c='red', s=15, alpha=0.6, label='Outlier')
ax.plot(x_line, 2*x_line+5, 'g--', linewidth=2, label='True')
ax.plot(x_line, params_ols[0]*x_line+params_ols[1], 'r-', linewidth=2, label='OLS')
ax.plot(x_line, params_huber[0]*x_line+params_huber[1], 'b-', linewidth=2, label='Huber')
ax.plot(x_line, params_tukey[0]*x_line+params_tukey[1], 'm-', linewidth=2, label='Tukey')
ax.plot(x_line, params_cauchy[0]*x_line+params_cauchy[1], 'c-', linewidth=2, label='Cauchy')
ax.set_title("Perbandingan Semua Metode")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Weight functions
ax2 = axes[0][1]
r = np.linspace(-6, 6, 200)
ax2.plot(r, huber_weights(r), 'b-', linewidth=2, label='Huber')
ax2.plot(r, tukey_weights(r), 'm-', linewidth=2, label='Tukey')
ax2.plot(r, cauchy_weights(r), 'c-', linewidth=2, label='Cauchy')
ax2.set_title("Weight Functions")
ax2.set_xlabel("Standardized Residual")
ax2.set_ylabel("Weight")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Final weights (Huber)
ax3 = axes[0][2]
final_w = w_hist_h[-1]
colors = ['blue' if i < n_inlier else 'red' for i in range(N)]
ax3.bar(range(N), final_w, color=colors, alpha=0.7)
ax3.set_title("Final Weights (Huber)")
ax3.set_xlabel("Point Index")
ax3.set_ylabel("Weight")
ax3.axvline(n_inlier - 0.5, color='gray', linestyle='--', alpha=0.5)

# Konvergensi slope di setiap iterasi
ax4 = axes[1][0]
slopes_h = [p[0] for p in p_hist_h]
slopes_t = [p[0] for p in p_hist_t]
slopes_c = [p[0] for p in p_hist_c]
ax4.plot(slopes_h, 'b-o', markersize=4, label='Huber')
ax4.plot(slopes_t, 'm-s', markersize=4, label='Tukey')
ax4.plot(slopes_c, 'c-^', markersize=4, label='Cauchy')
ax4.axhline(2.0, color='green', linestyle='--', label='True (a=2)')
ax4.axhline(params_ols[0], color='red', linestyle=':', label='OLS')
ax4.set_title("Konvergensi Slope per Iterasi")
ax4.set_xlabel("Iterasi")
ax4.set_ylabel("Slope (a)")
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# Residual boxplot
ax5 = axes[1][1]
res_ols = y_all - (params_ols[0]*x_all + params_ols[1])
res_hub = y_all - (params_huber[0]*x_all + params_huber[1])
res_tuk = y_all - (params_tukey[0]*x_all + params_tukey[1])
ax5.boxplot([res_ols, res_hub, res_tuk], labels=['OLS', 'Huber', 'Tukey'])
ax5.set_title("Distribusi Residual")
ax5.set_ylabel("Residual")
ax5.grid(True, alpha=0.3)

# Final weights Tukey (hard rejection)
ax6 = axes[1][2]
final_w_t = w_hist_t[-1]
ax6.bar(range(N), final_w_t, color=colors, alpha=0.7)
ax6.set_title("Final Weights (Tukey — Hard Rejection)")
ax6.set_xlabel("Point Index")
ax6.set_ylabel("Weight")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "10_irls_robust_fitting.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Disimpan: {output_path}")

# ============================================================
# 6. Perbandingan error
# ============================================================
print("\n--- 6. Error terhadap True Model ---")

methods = {
    'OLS': params_ols,
    'Huber': params_huber,
    'Tukey': params_tukey,
    'Cauchy': params_cauchy,
}

for name, p in methods.items():
    err_slope = abs(p[0] - 2.0)
    err_intercept = abs(p[1] - 5.0)
    print(f"  {name:8s}: err_slope={err_slope:.4f}, err_intercept={err_intercept:.4f}")

print("\n" + "=" * 60)
print("PERCOBAAN 10 SELESAI")
print("=" * 60)
