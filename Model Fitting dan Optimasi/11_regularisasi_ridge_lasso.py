"""
==========================================================================
PERCOBAAN 11: REGULARISASI (RIDGE DAN LASSO)
==========================================================================
Regularisasi menambahkan penalty pada parameter model untuk mencegah
overfitting. Sangat penting ketika data sedikit atau model kompleks.

- Ridge (L2): min ||Ax - b||^2 + λ||x||^2
  → Solusi: x = (A^T A + λI)^{-1} A^T b
  → Semua parameter mengecil (shrinkage)

- Lasso (L1): min ||Ax - b||^2 + λ||x||_1
  → Beberapa parameter menjadi tepat nol (sparse)
  → Diselesaikan iteratif (coordinate descent)

Fungsi utama:
- np.linalg.solve()       : solusi Ridge (closed-form)
- np.polyfit()            : fitting polinomial (baseline)
- np.polyval()            : evaluasi polinomial
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
print("PERCOBAAN 11: REGULARISASI (RIDGE DAN LASSO)")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Membuat data sintetis (sedikit data, model kompleks)
# ============================================================
print("\n--- 1. Data Sintetis ---")

# Hanya 15 titik data (sedikit) — rawan overfitting
N = 15
x_data = np.linspace(0, 1, N)

# True function: y = sin(2πx)
y_true = np.sin(2 * np.pi * x_data)

# Tambahkan noise
y_data = y_true + np.random.randn(N) * 0.3

# Data untuk evaluasi (dense)
x_test = np.linspace(0, 1, 200)
y_test = np.sin(2 * np.pi * x_test)

print(f"  Titik data: {N}")
print(f"  True function: sin(2πx)")

# ============================================================
# 2. Polynomial fitting TANPA regularisasi (overfitting!)
# ============================================================
print("\n--- 2. Polynomial Fit Tanpa Regularisasi ---")

degrees = [1, 3, 5, 9, 14]

for deg in degrees:
    # np.polyfit: fit polinomial derajat deg
    coeffs = np.polyfit(x_data, y_data, deg)
    y_pred = np.polyval(coeffs, x_test)
    
    # Error pada data training dan test
    y_train_pred = np.polyval(coeffs, x_data)
    mse_train = np.mean((y_data - y_train_pred)**2)
    mse_test = np.mean((y_pred - y_test)**2)
    
    print(f"  deg={deg:2d}: MSE_train={mse_train:.4f}, MSE_test={mse_test:.4f}, "
          f"max|coeff|={np.max(np.abs(coeffs)):.2f}")

# ============================================================
# 3. Ridge Regression (L2 Regularization)
# ============================================================
print("\n--- 3. Ridge Regression ---")

def ridge_regression(X, y, lam):
    """
    Ridge regression: min ||Xw - y||^2 + λ||w||^2
    Solusi closed-form: w = (X^T X + λI)^{-1} X^T y
    
    Parameters:
    - X: design matrix (N x D)
    - y: target (N,)
    - lam: parameter regularisasi λ
    
    Returns:
    - w: parameter model
    """
    D = X.shape[1]
    # Identitas matrix (jangan regularisasi bias/intercept term)
    I = np.eye(D)
    # Solusi: (X^T X + λI)^{-1} X^T y
    w = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
    return w

# Buat design matrix polinomial derajat tinggi (9)
deg_ridge = 9

def poly_features(x, degree):
    """Membuat matrix fitur polinomial [1, x, x^2, ..., x^degree]"""
    return np.vstack([x**i for i in range(degree + 1)]).T

X_train = poly_features(x_data, deg_ridge)
X_test_dense = poly_features(x_test, deg_ridge)

# Coba berbagai nilai lambda
lambdas = [0, 1e-6, 1e-4, 1e-2, 0.1, 1, 10, 100]

print(f"  Degree: {deg_ridge}")
for lam in lambdas:
    w = ridge_regression(X_train, y_data, lam)
    y_pred_train = X_train @ w
    y_pred_test = X_test_dense @ w
    mse_train = np.mean((y_data - y_pred_train)**2)
    mse_test = np.mean((y_pred_test - y_test)**2)
    print(f"  λ={lam:8.1e}: MSE_train={mse_train:.4f}, MSE_test={mse_test:.4f}, "
          f"||w||={np.linalg.norm(w):.4f}")

# ============================================================
# 4. Lasso Regression (L1 Regularization) - Coordinate Descent
# ============================================================
print("\n--- 4. Lasso Regression ---")

def lasso_regression(X, y, lam, max_iter=1000, tol=1e-6):
    """
    Lasso: min ||Xw - y||^2 + λ||w||_1
    Diselesaikan dengan coordinate descent.
    
    Soft thresholding: w_j = sign(z) * max(|z| - λ, 0)
    """
    N, D = X.shape
    w = np.zeros(D)
    
    for iteration in range(max_iter):
        w_old = w.copy()
        for j in range(D):
            # Residual tanpa fitur j
            r = y - X @ w + X[:, j] * w[j]
            # Korelasi fitur j dengan residual
            z = X[:, j] @ r / N
            # Norm kolom
            norm_j = np.sum(X[:, j]**2) / N
            # Soft thresholding
            if z > lam / (2 * N):
                w[j] = (z - lam / (2 * N)) / norm_j
            elif z < -lam / (2 * N):
                w[j] = (z + lam / (2 * N)) / norm_j
            else:
                w[j] = 0.0
        
        # Cek konvergensi
        if np.max(np.abs(w - w_old)) < tol:
            break
    
    return w

# Coba berbagai lambda untuk Lasso
lambdas_lasso = [0.001, 0.01, 0.1, 1.0, 5.0]

for lam in lambdas_lasso:
    w = lasso_regression(X_train, y_data, lam)
    y_pred = X_test_dense @ w
    mse = np.mean((y_pred - y_test)**2)
    n_zero = np.sum(np.abs(w) < 1e-8)  # koefisien yang jadi nol
    print(f"  λ={lam:5.3f}: MSE_test={mse:.4f}, "
          f"koef_nol={n_zero}/{len(w)}, ||w||_1={np.sum(np.abs(w)):.4f}")

# ============================================================
# 5. Visualisasi perbandingan
# ============================================================
print("\n--- 5. Visualisasi ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Tanpa regularisasi (overfitting)
ax = axes[0][0]
ax.scatter(x_data, y_data, c='black', s=30, zorder=5, label='Data')
ax.plot(x_test, y_test, 'g--', linewidth=2, label='True')
for deg, color in zip([3, 9, 14], ['blue', 'orange', 'red']):
    coeffs = np.polyfit(x_data, y_data, deg)
    ax.plot(x_test, np.polyval(coeffs, x_test), color=color, linewidth=1.5,
            label=f'deg={deg}')
ax.set_title("Tanpa Regularisasi")
ax.set_ylim(-2, 2)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 2: Ridge dengan berbagai lambda
ax = axes[0][1]
ax.scatter(x_data, y_data, c='black', s=30, zorder=5, label='Data')
ax.plot(x_test, y_test, 'g--', linewidth=2, label='True')
for lam, color in zip([0, 1e-4, 1e-2, 1], ['red', 'orange', 'blue', 'purple']):
    w = ridge_regression(X_train, y_data, lam)
    ax.plot(x_test, X_test_dense @ w, color=color, linewidth=1.5,
            label=f'λ={lam:.0e}')
ax.set_title(f"Ridge (L2) deg={deg_ridge}")
ax.set_ylim(-2, 2)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 3: Lasso
ax = axes[0][2]
ax.scatter(x_data, y_data, c='black', s=30, zorder=5, label='Data')
ax.plot(x_test, y_test, 'g--', linewidth=2, label='True')
for lam, color in zip([0.001, 0.01, 0.1, 1.0], ['red', 'orange', 'blue', 'purple']):
    w = lasso_regression(X_train, y_data, lam)
    ax.plot(x_test, X_test_dense @ w, color=color, linewidth=1.5,
            label=f'λ={lam}')
ax.set_title(f"Lasso (L1) deg={deg_ridge}")
ax.set_ylim(-2, 2)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 4: L-curve (trade-off bias-variance)
ax = axes[1][0]
lambdas_sweep = np.logspace(-6, 2, 50)
mse_trains = []
mse_tests = []
norms = []
for lam in lambdas_sweep:
    w = ridge_regression(X_train, y_data, lam)
    mse_trains.append(np.mean((y_data - X_train @ w)**2))
    mse_tests.append(np.mean((X_test_dense @ w - y_test)**2))
    norms.append(np.linalg.norm(w))
ax.semilogx(lambdas_sweep, mse_trains, 'b-', label='MSE train')
ax.semilogx(lambdas_sweep, mse_tests, 'r-', label='MSE test')
ax.set_title("Bias-Variance Trade-off (Ridge)")
ax.set_xlabel("λ")
ax.set_ylabel("MSE")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Koefisien path (Ridge)
ax = axes[1][1]
coeff_paths = np.array([ridge_regression(X_train, y_data, lam) for lam in lambdas_sweep])
for i in range(min(deg_ridge + 1, 10)):
    ax.semilogx(lambdas_sweep, coeff_paths[:, i], linewidth=1.5, label=f'w{i}')
ax.set_title("Ridge Coefficient Path")
ax.set_xlabel("λ")
ax.set_ylabel("Coefficient value")
ax.grid(True, alpha=0.3)

# Plot 6: Koefisien path (Lasso)
ax = axes[1][2]
lambdas_l = np.logspace(-3, 1, 30)
coeff_paths_l = np.array([lasso_regression(X_train, y_data, lam) for lam in lambdas_l])
for i in range(min(deg_ridge + 1, 10)):
    ax.semilogx(lambdas_l, coeff_paths_l[:, i], linewidth=1.5, label=f'w{i}')
ax.set_title("Lasso Coefficient Path (Sparse!)")
ax.set_xlabel("λ")
ax.set_ylabel("Coefficient value")
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "11_regularisasi.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Disimpan: {output_path}")

print("\n" + "=" * 60)
print("PERCOBAAN 11 SELESAI")
print("=" * 60)
