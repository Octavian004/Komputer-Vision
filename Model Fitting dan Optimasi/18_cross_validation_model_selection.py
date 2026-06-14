"""
==========================================================================
PERCOBAAN 18: CROSS-VALIDATION DAN MODEL SELECTION
==========================================================================
Cross-validation digunakan untuk mengevaluasi dan memilih model terbaik.
Dalam konteks computer vision, ini berguna untuk memilih parameter model
fitting seperti derajat polinomial, threshold RANSAC, dll.

Konsep utama:
- K-Fold Cross-Validation  : bagi data menjadi K fold, latih di K-1, uji di 1
- Leave-One-Out (LOO)      : K = jumlah data (K-Fold ekstrem)
- Train/Test Split          : evaluasi sederhana
- Bias-Variance Trade-off  : model sederhana (underfitting) vs kompleks (overfitting)
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
print("PERCOBAAN 18: CROSS-VALIDATION DAN MODEL SELECTION")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Generate data dengan noise
# ============================================================
print("\n--- 1. Generate Data ---")

# Buat data sinusoidal dengan noise
n_samples = 50
x = np.sort(np.random.uniform(0, 2 * np.pi, n_samples))
y_true = np.sin(x)
noise = np.random.normal(0, 0.3, n_samples)
y = y_true + noise

print(f"  Jumlah sampel: {n_samples}")
print(f"  Noise std: 0.3")
print(f"  Fungsi true: sin(x)")

# ============================================================
# 2. Fitting dengan berbagai derajat polinomial
# ============================================================
print("\n--- 2. Fitting Polinomial ---")

degrees = [1, 2, 3, 5, 7, 10, 15, 20]
x_plot = np.linspace(0, 2 * np.pi, 200)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, deg in enumerate(degrees):
    ax = axes.flat[idx]
    
    # np.polyfit: fitting polinomial derajat n
    coeffs = np.polyfit(x, y, deg)
    y_pred = np.polyval(coeffs, x_plot)
    
    # Hitung MSE pada data training
    y_train_pred = np.polyval(coeffs, x)
    train_mse = np.mean((y - y_train_pred) ** 2)
    
    ax.scatter(x, y, c='blue', s=20, alpha=0.5, label='Data')
    ax.plot(x_plot, np.sin(x_plot), 'g--', linewidth=1, label='True')
    ax.plot(x_plot, y_pred, 'r-', linewidth=2, label=f'Deg={deg}')
    ax.set_title(f"Degree={deg}, MSE={train_mse:.4f}")
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "18_polynomial_degrees.png")
plt.show()
            dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 3. K-Fold Cross-Validation manual
# ============================================================
print("\n--- 3. K-Fold Cross-Validation ---")

def k_fold_cv(x, y, degree, k=5):
    """
    K-Fold Cross-Validation untuk fitting polinomial.
    Membagi data menjadi K bagian, latih di K-1, uji di 1 bagian secara bergantian.
    """
    n = len(x)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    fold_size = n // k
    cv_errors = []
    
    for fold in range(k):
        # Indeks test: 1 fold
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k - 1 else n
        test_idx = indices[test_start:test_end]
        
        # Indeks train: semua kecuali fold test
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        # Fit model pada data training
        coeffs = np.polyfit(x[train_idx], y[train_idx], degree)
        
        # Evaluasi pada data test
        y_pred = np.polyval(coeffs, x[test_idx])
        mse = np.mean((y[test_idx] - y_pred) ** 2)
        cv_errors.append(mse)
    
    return np.mean(cv_errors), np.std(cv_errors)

# Evaluasi setiap derajat polinomial
cv_results = {}
k = 5

for deg in degrees:
    mean_err, std_err = k_fold_cv(x, y, deg, k=k)
    cv_results[deg] = (mean_err, std_err)
    print(f"  Degree={deg:2d}: CV mean MSE={mean_err:.4f} ± {std_err:.4f}")

# Degree terbaik berdasarkan CV
best_degree = min(cv_results, key=lambda d: cv_results[d][0])
print(f"\n  >>> Model terbaik: derajat {best_degree} "
      f"(CV MSE = {cv_results[best_degree][0]:.4f})")

# ============================================================
# 4. Training Error vs CV Error
# ============================================================
print("\n--- 4. Training Error vs CV Error ---")

train_errors = []
cv_mean_errors = []
cv_std_errors = []

for deg in degrees:
    # Training error
    coeffs = np.polyfit(x, y, deg)
    y_train_pred = np.polyval(coeffs, x)
    train_mse = np.mean((y - y_train_pred) ** 2)
    train_errors.append(train_mse)
    
    # CV error
    cv_mean, cv_std = cv_results[deg]
    cv_mean_errors.append(cv_mean)
    cv_std_errors.append(cv_std)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(degrees, train_errors, 'b-o', label='Training Error')
ax.plot(degrees, cv_mean_errors, 'r-o', label='CV Error')
ax.fill_between(degrees,
                np.array(cv_mean_errors) - np.array(cv_std_errors),
                np.array(cv_mean_errors) + np.array(cv_std_errors),
                alpha=0.2, color='red')
ax.axvline(x=best_degree, color='green', linestyle='--',
           label=f'Best degree={best_degree}')
ax.set_xlabel("Polynomial Degree")
ax.set_ylabel("MSE")
ax.set_title("Training Error vs Cross-Validation Error\n(Bias-Variance Trade-off)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "18_train_vs_cv.png"),
plt.show()
            dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. Leave-One-Out Cross-Validation
# ============================================================
print("\n--- 5. Leave-One-Out Cross-Validation ---")

def loo_cv(x, y, degree):
    """
    Leave-One-Out CV: K = N (setiap data jadi test set sekali).
    """
    n = len(x)
    errors = []
    
    for i in range(n):
        # Train pada semua kecuali titik i
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        
        coeffs = np.polyfit(x[train_mask], y[train_mask], degree)
        y_pred = np.polyval(coeffs, x[i])
        errors.append((y[i] - y_pred) ** 2)
    
    return np.mean(errors), np.std(errors)

# LOO CV untuk beberapa derajat kunci
loo_degrees = [1, 3, 5, 7, 10]
for deg in loo_degrees:
    loo_mean, loo_std = loo_cv(x, y, deg)
    kf_mean, kf_std = cv_results[deg]
    print(f"  Degree={deg}: LOO MSE={loo_mean:.4f}, 5-Fold MSE={kf_mean:.4f}")

# ============================================================
# 6. CV untuk memilih RANSAC threshold
# ============================================================
print("\n--- 6. CV untuk RANSAC Threshold ---")

# Buat data garis dengan outlier
n_line = 80
x_line = np.random.uniform(0, 100, n_line)
y_line = 2 * x_line + 10 + np.random.normal(0, 5, n_line)
# Tambah outlier
n_outlier = 20
x_outlier = np.random.uniform(0, 100, n_outlier)
y_outlier = np.random.uniform(-50, 200, n_outlier)
x_all = np.concatenate([x_line, x_outlier])
y_all = np.concatenate([y_line, y_outlier])

# Evaluasi berbagai threshold RANSAC menggunakan CV
thresholds = [1, 3, 5, 10, 15, 20, 30]
threshold_scores = {}

for thresh in thresholds:
    inlier_ratios = []
    residuals = []
    
    for trial in range(20):  # 20 random RANSAC runs
        # Random sample 2 titik
        idx = np.random.choice(len(x_all), 2, replace=False)
        x_s, y_s = x_all[idx], y_all[idx]
        
        if abs(x_s[1] - x_s[0]) < 1e-6:
            continue
        
        # Fit garis
        slope = (y_s[1] - y_s[0]) / (x_s[1] - x_s[0])
        intercept = y_s[0] - slope * x_s[0]
        
        # Hitung residual
        y_pred_all = slope * x_all + intercept
        res = np.abs(y_all - y_pred_all)
        
        # Hitung inlier ratio
        inliers = res < thresh
        inlier_ratios.append(np.mean(inliers))
        residuals.append(np.mean(res[inliers]) if np.sum(inliers) > 0 else float('inf'))
    
    mean_inlier = np.mean(inlier_ratios)
    mean_residual = np.mean(residuals)
    threshold_scores[thresh] = (mean_inlier, mean_residual)
    print(f"  Threshold={thresh:2d}: inlier_ratio={mean_inlier:.3f}, "
          f"mean_residual={mean_residual:.2f}")

# ============================================================
# 7. Learning Curve
# ============================================================
print("\n--- 7. Learning Curve ---")

train_sizes = [5, 10, 15, 20, 30, 40, 50]
degr = best_degree

train_curve_err = []
cv_curve_err = []

for size in train_sizes:
    if size > n_samples:
        break
    
    # Subsample data
    idx_sub = np.random.choice(n_samples, size, replace=False)
    x_sub, y_sub = x[idx_sub], y[idx_sub]
    
    # Training error
    coeffs = np.polyfit(x_sub, y_sub, min(degr, size - 1))
    y_pred_train = np.polyval(coeffs, x_sub)
    train_err = np.mean((y_sub - y_pred_train) ** 2)
    train_curve_err.append(train_err)
    
    # CV error (3-fold jika cukup data)
    if size >= 6:
        cv_err, _ = k_fold_cv(x_sub, y_sub, min(degr, size - 1), k=3)
    else:
        cv_err = train_err * 2
    cv_curve_err.append(cv_err)
    
    print(f"  N={size:2d}: train_err={train_err:.4f}, cv_err={cv_err:.4f}")

# ============================================================
# 8. Visualisasi gabungan
# ============================================================
print("\n--- 8. Visualisasi Gabungan ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Best model fit
coeffs_best = np.polyfit(x, y, best_degree)
axes[0, 0].scatter(x, y, c='blue', s=20, alpha=0.5)
axes[0, 0].plot(x_plot, np.sin(x_plot), 'g--', label='True')
axes[0, 0].plot(x_plot, np.polyval(coeffs_best, x_plot), 'r-',
                label=f'Best (d={best_degree})')
axes[0, 0].legend()
axes[0, 0].set_title(f"Best Model: degree={best_degree}")

# Plot 2: Train vs CV error
axes[0, 1].plot(degrees, train_errors, 'b-o', label='Train')
axes[0, 1].plot(degrees, cv_mean_errors, 'r-o', label='CV')
axes[0, 1].axvline(x=best_degree, color='green', linestyle='--')
axes[0, 1].set_xlabel("Degree")
axes[0, 1].set_ylabel("MSE")
axes[0, 1].set_title("Bias-Variance Trade-off")
axes[0, 1].legend()
axes[0, 1].set_yscale('log')

# Plot 3: K-Fold error by degree
cv_means = [cv_results[d][0] for d in degrees]
cv_stds = [cv_results[d][1] for d in degrees]
axes[0, 2].bar(range(len(degrees)), cv_means, yerr=cv_stds,
               tick_label=degrees, color='steelblue', alpha=0.7)
axes[0, 2].set_xlabel("Polynomial Degree")
axes[0, 2].set_ylabel("CV MSE")
axes[0, 2].set_title("5-Fold CV Error per Degree")

# Plot 4: RANSAC threshold analysis
ax4 = axes[1, 0]
t_list = list(threshold_scores.keys())
inlier_list = [threshold_scores[t][0] for t in t_list]
residual_list = [threshold_scores[t][1] for t in t_list]
ax4_twin = ax4.twinx()
ax4.plot(t_list, inlier_list, 'b-o', label='Inlier ratio')
ax4_twin.plot(t_list, residual_list, 'r-s', label='Mean residual')
ax4.set_xlabel("RANSAC Threshold")
ax4.set_ylabel("Inlier Ratio", color='blue')
ax4_twin.set_ylabel("Mean Residual", color='red')
ax4.set_title("RANSAC Threshold Selection")

# Plot 5: Learning curve
axes[1, 1].plot(train_sizes[:len(train_curve_err)], train_curve_err, 'b-o', label='Train')
axes[1, 1].plot(train_sizes[:len(cv_curve_err)], cv_curve_err, 'r-o', label='CV')
axes[1, 1].set_xlabel("Training Size")
axes[1, 1].set_ylabel("MSE")
axes[1, 1].set_title("Learning Curve")
axes[1, 1].legend()

# Plot 6: Residual plot
coeffs_final = np.polyfit(x, y, best_degree)
y_pred_final = np.polyval(coeffs_final, x)
residuals_final = y - y_pred_final
axes[1, 2].scatter(x, residuals_final, c='steelblue', s=20)
axes[1, 2].axhline(y=0, color='red', linestyle='--')
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("Residual")
axes[1, 2].set_title(f"Residual Plot (degree={best_degree})")

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "18_cross_validation_all.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Disimpan: {output_path}")

print("\n" + "=" * 60)
print("PERCOBAAN 18 SELESAI")
print("=" * 60)
