"""
==========================================================================
PERCOBAAN 01: ORDINARY LEAST SQUARES (OLS)
==========================================================================
Program ini mempelajari teknik Ordinary Least Squares (OLS) yang
merupakan dasar dari semua teknik model fitting dalam computer vision.

OLS meminimalkan jumlah kuadrat residual (selisih) antara data
observasi dan model prediksi. Metode ini sangat penting untuk
fitting garis, kurva, dan model matematika pada data nyata.

Konsep yang dipelajari:
- Normal equation: x = (A^T A)^(-1) A^T b
- Fitting garis lurus (linear regression)
- Fitting polinomial (derajat 1, 2, 3)
- cv2.fitLine() untuk fitting garis pada titik 2D
- Perbandingan jenis distance type (L2, L1, Huber)
- R-squared sebagai metrik kebaikan model

Fungsi utama:
- np.linalg.lstsq(A, b)       : solusi least squares (stabil numerik)
- np.polyfit(x, y, deg)        : fitting polinomial derajat deg
- np.polyval(p, x)             : evaluasi polinomial di titik x
- cv2.fitLine(pts, distType)   : fit garis ke kumpulan titik 2D

Hasil: Visualisasi perbandingan berbagai metode OLS disimpan ke output/
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor NumPy untuk operasi matriks dan array
import numpy as np

# Mengimpor os untuk operasi file dan folder
import os

# Mengimpor matplotlib backend non-GUI agar bisa menyimpan tanpa tampilan
import matplotlib

# Mengimpor pyplot untuk membuat grafik visualisasi
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fitting_garis_linear():
    """
    Melakukan fitting garis lurus menggunakan OLS.
    Membuat data sintetis y = 2x + 10 + noise, lalu mencocokkan
    garis menggunakan 3 metode: manual, lstsq, dan polyfit.
    """
    print("\n--- 1. Linear Fitting (y = ax + b) ---")

    # Mengatur seed agar hasil dapat direproduksi
    np.random.seed(42)

    # Membuat 100 titik data sintetis
    N = 100
    x_data = np.linspace(0, 10, N)

    # Model sebenarnya: y = 2x + 10
    y_true = 2 * x_data + 10

    # Menambahkan noise Gaussian dengan standar deviasi 3
    noise = np.random.randn(N) * 3
    y_data = y_true + noise

    # Menyusun matriks desain A = [x, 1] untuk normal equation
    A = np.vstack([x_data, np.ones(N)]).T

    # Metode 1: Solusi manual menggunakan normal equation
    params_manual = np.linalg.inv(A.T @ A) @ A.T @ y_data
    print(f"  Manual: a={params_manual[0]:.4f}, b={params_manual[1]:.4f}")

    # Metode 2: Menggunakan np.linalg.lstsq (lebih stabil numerik)
    params_lstsq, residuals, rank, sv = np.linalg.lstsq(A, y_data, rcond=None)
    print(f"  lstsq:  a={params_lstsq[0]:.4f}, b={params_lstsq[1]:.4f}")

    # Metode 3: Menggunakan np.polyfit untuk fitting derajat 1
    coeffs = np.polyfit(x_data, y_data, 1)
    print(f"  polyfit: a={coeffs[0]:.4f}, b={coeffs[1]:.4f}")

    # Menghitung R-squared untuk mengukur kebaikan model
    y_pred = params_lstsq[0] * x_data + params_lstsq[1]
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    print(f"  R² = {r_squared:.4f}")

    return x_data, y_data, y_true, y_pred, params_lstsq, r_squared


def fitting_polinomial(x_data):
    """
    Melakukan fitting polinomial derajat 1, 2, 3 pada data kuadratik.
    Membandingkan MSE setiap derajat untuk menunjukkan efek
    underfitting dan overfitting.
    """
    print("\n--- 2. Polynomial Fitting ---")

    np.random.seed(42)
    N = len(x_data)

    # Membuat data kuadratik: y = 0.5x^2 - 3x + 7 + noise
    y_quad = 0.5 * x_data**2 - 3 * x_data + 7 + np.random.randn(N) * 2

    # Melakukan fitting untuk derajat 1, 2, dan 3
    poly_results = {}
    for deg in [1, 2, 3]:
        p = np.polyfit(x_data, y_quad, deg)
        y_fit = np.polyval(p, x_data)
        err = np.mean((y_quad - y_fit)**2)
        poly_results[deg] = (p, y_fit, err)
        print(f"  Derajat {deg}: MSE = {err:.4f}")

    return y_quad, poly_results


def fitting_garis_gambar():
    """
    Menggunakan cv2.fitLine() untuk fitting garis pada titik-titik
    yang dideteksi dari gambar. Menunjukkan penggunaan OpenCV
    untuk model fitting langsung pada data citra.
    """
    print("\n--- 3. cv2.fitLine pada Gambar ---")

    # Membaca gambar dari file
    img_pts = cv2.imread(os.path.join(IMAGE_DIR, "lena.jpg"))
    img_result = None

    if img_pts is not None:
        # Mengkonversi ke grayscale untuk deteksi titik
        gray = cv2.cvtColor(img_pts, cv2.COLOR_BGR2GRAY)

        # Membuat mask: piksel yang cukup gelap
        mask = gray < 200

        # Mengambil koordinat titik dari mask
        pts = np.column_stack(np.where(mask))
        pts_xy = pts[:, ::-1].astype(np.float32)

        if len(pts_xy) > 10:
            # Fitting garis dengan cv2.fitLine menggunakan DIST_L2
            line = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line.flatten()
            print(f"  Vektor arah: ({vx:.4f}, {vy:.4f})")
            print(f"  Titik pada garis: ({x0:.1f}, {y0:.1f})")

            # Menggambar garis hasil fitting pada gambar
            img_result = img_pts.copy()
            t = 500
            pt1 = (int(x0 - t * vx), int(y0 - t * vy))
            pt2 = (int(x0 + t * vx), int(y0 + t * vy))
            cv2.line(img_result, pt1, pt2, (0, 255, 0), 2)
    else:
        print("  [SKIP] Gambar tidak ditemukan")

    return img_result


def perbandingan_distance_type():
    """
    Membandingkan berbagai distance type pada cv2.fitLine().
    Menunjukkan bagaimana DIST_L1 dan DIST_HUBER lebih tahan
    terhadap outlier dibandingkan DIST_L2 (least squares biasa).
    """
    print("\n--- 4. Perbandingan distType ---")

    np.random.seed(42)

    # Membuat 80 titik data yang mengikuti garis y = 0.5x + 100
    pts_outlier = []
    for i in range(80):
        x = np.random.uniform(50, 450)
        y = 0.5 * x + 100 + np.random.randn() * 5
        pts_outlier.append([x, y])

    # Menambahkan 20 titik outlier yang tersebar acak
    for i in range(20):
        pts_outlier.append([np.random.uniform(50, 450), np.random.uniform(50, 450)])

    pts_arr = np.array(pts_outlier, dtype=np.float32)

    # Mendefinisikan jenis distance type yang akan dibandingkan
    dist_types = {
        "DIST_L2": cv2.DIST_L2,
        "DIST_L1": cv2.DIST_L1,
        "DIST_L12": cv2.DIST_L12,
        "DIST_HUBER": cv2.DIST_HUBER,
    }

    # Melakukan fitting untuk setiap distance type
    fit_results = {}
    for name, dtype in dist_types.items():
        line = cv2.fitLine(pts_arr, dtype, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()
        slope = vy / vx if abs(vx) > 1e-6 else float('inf')
        fit_results[name] = (vx, vy, x0, y0, slope)
        print(f"  {name}: slope={slope:.4f}")

    return pts_arr, fit_results


def visualisasi_hasil(x_data, y_data, y_true, y_pred, params_lstsq, r_squared,
                      y_quad, poly_results, img_result, pts_arr, fit_results):
    """
    Membuat visualisasi gabungan dari semua percobaan OLS dan
    menyimpan hasilnya ke folder output sebagai file PNG.
    """
    print("\n--- 5. Menyimpan Visualisasi ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Linear fit dengan OLS
    axes[0, 0].scatter(x_data, y_data, s=10, alpha=0.5, label="Data")
    axes[0, 0].plot(x_data, y_pred, 'r-', linewidth=2,
                    label=f"OLS: y={params_lstsq[0]:.2f}x+{params_lstsq[1]:.2f}")
    axes[0, 0].plot(x_data, y_true, 'g--', linewidth=1, label="True")
    axes[0, 0].set_title(f"Linear Fit (R²={r_squared:.4f})")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Polynomial fitting derajat 1, 2, 3
    axes[0, 1].scatter(x_data, y_quad, s=10, alpha=0.5, label="Data")
    for deg, (p, y_f, err) in poly_results.items():
        axes[0, 1].plot(x_data, y_f, linewidth=2, label=f"deg={deg}, MSE={err:.2f}")
    axes[0, 1].set_title("Polynomial Fitting")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Hasil cv2.fitLine pada gambar
    if img_result is not None:
        axes[1, 0].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("cv2.fitLine (DIST_L2)")
    axes[1, 0].axis("off")

    # Plot 4: Perbandingan distance type dengan outlier
    axes[1, 1].scatter(pts_arr[:, 0], pts_arr[:, 1], s=10, alpha=0.5)
    colors_line = ['r', 'g', 'b', 'm']
    for (name, (vx, vy, x0, y0, slope)), col in zip(fit_results.items(), colors_line):
        xs = np.linspace(50, 450, 100)
        ys = y0 + (xs - x0) * vy / vx
        axes[1, 1].plot(xs, ys, col, linewidth=2, label=name)
    axes[1, 1].set_title("Perbandingan distType")
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 500)
    axes[1, 1].set_ylim(0, 500)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Percobaan 01: Ordinary Least Squares", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Menyimpan gambar hasil ke folder output
    path = os.path.join(OUTPUT_DIR, "01_ols_hasil.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OUTPUT] Disimpan: {path}")

    # Menampilkan gambar hasil
    hasil = cv2.imread(path)
    if hasil is not None:
        cv2.imshow("Percobaan 01 - OLS", hasil)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def ringkasan():
    """Menampilkan ringkasan pembelajaran dari percobaan OLS."""
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 01")
    print("=" * 60)
    print("""
1. OLS meminimalkan ||Ax - b||^2 → solusi: x = (A^T A)^(-1) A^T b
2. np.linalg.lstsq() lebih stabil dari inverse langsung
3. np.polyfit() untuk fitting polinomial (derajat n)
4. cv2.fitLine() untuk fitting garis ke kumpulan titik 2D
5. DIST_L2 = least squares, DIST_L1 = median, DIST_HUBER = robust
6. R² mendekati 1 → model fit baik
7. Derajat polinomial terlalu tinggi → overfitting
""")


def main():
    """Fungsi utama yang menjalankan seluruh percobaan OLS."""
    print("=" * 60)
    print("PERCOBAAN 01: ORDINARY LEAST SQUARES (OLS)")
    print("=" * 60)

    # Langkah 1: Fitting garis linear
    x_data, y_data, y_true, y_pred, params_lstsq, r_squared = fitting_garis_linear()

    # Langkah 2: Fitting polinomial
    y_quad, poly_results = fitting_polinomial(x_data)

    # Langkah 3: Fitting garis dari gambar
    img_result = fitting_garis_gambar()

    # Langkah 4: Perbandingan distance type
    pts_arr, fit_results = perbandingan_distance_type()

    # Langkah 5: Visualisasi dan simpan hasil
    visualisasi_hasil(x_data, y_data, y_true, y_pred, params_lstsq, r_squared,
                      y_quad, poly_results, img_result, pts_arr, fit_results)

    # Menampilkan ringkasan
    ringkasan()


if __name__ == "__main__":
    main()
