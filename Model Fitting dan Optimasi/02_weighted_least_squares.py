
def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 02: WEIGHTED LEAST SQUARES (WLS)
    ==========================================================================
    WLS memberikan bobot berbeda untuk setiap observasi. Titik yang lebih
    dipercaya mendapat bobot lebih besar, sehingga hasilnya lebih akurat
    dibandingkan OLS saat kualitas data tidak merata.

    Rumus: min ||W^{1/2}(Ax - b)||^2
    Solusi: x = (A^T W A)^{-1} A^T W b

    Fungsi utama:
    - np.linalg.lstsq()      : solusi least squares
    - np.diag()               : membuat matriks diagonal (bobot)
    - np.linalg.inv()         : invers matriks
    - matplotlib.pyplot       : visualisasi fitting
    ==========================================================================
    """

    # Mengimpor library yang diperlukan
    import cv2
    import numpy as np
    import os
    import matplotlib
    import matplotlib.pyplot as plt

    # Mendapatkan direktori script saat ini
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PERCOBAAN 02: WEIGHTED LEAST SQUARES (WLS)")
    print("=" * 60)

    # Mengatur seed agar hasil reproducible
    np.random.seed(42)

    # ============================================================
    # 1. Membuat data sintetis dengan kualitas berbeda
    # Data: y = 3x + 5 dengan noise bervariasi
    # ============================================================
    print("\n--- 1. Membuat Data dengan Heteroscedastic Noise ---")

    # Membuat 80 titik data
    N = 80

    # Variabel independen dari 0 sampai 10
    x_data = np.linspace(0, 10, N)

    # Model sebenarnya: y = 3x + 5
    y_true = 3 * x_data + 5

    # Noise yang bervariasi: kecil di awal, besar di akhir (heteroscedastic)
    # Simulasi: data di awal lebih presisi, data di akhir lebih noisy
    noise_std = 0.5 + 2.0 * (x_data / 10.0)  # standar deviasi meningkat
    noise = np.random.randn(N) * noise_std
    y_data = y_true + noise

    print(f"  Jumlah titik: {N}")
    print(f"  Noise std range: [{noise_std.min():.2f}, {noise_std.max():.2f}]")

    # ============================================================
    # 2. OLS (tanpa bobot) sebagai baseline
    # ============================================================
    print("\n--- 2. OLS (Tanpa Bobot) ---")

    # Menyusun matriks desain A = [x, 1]
    A = np.vstack([x_data, np.ones(N)]).T

    # Solusi OLS: x = (A^T A)^{-1} A^T b
    params_ols, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)
    print(f"  OLS: a={params_ols[0]:.4f}, b={params_ols[1]:.4f}")
    print(f"  (Seharusnya a=3.0, b=5.0)")

    # ============================================================
    # 3. WLS dengan bobot berdasarkan kebalikan variansi
    # Bobot w_i = 1/sigma_i^2 (data presisi tinggi = bobot tinggi)
    # ============================================================
    print("\n--- 3. WLS (Weighted Least Squares) ---")

    # Menghitung bobot: kebalikan dari variansi noise
    # Titik dengan noise kecil mendapat bobot lebih besar
    weights = 1.0 / (noise_std ** 2)

    # Membuat matriks diagonal bobot W
    W = np.diag(weights)

    # Solusi WLS: x = (A^T W A)^{-1} A^T W b
    AtWA = A.T @ W @ A
    AtWb = A.T @ W @ y_data
    params_wls = np.linalg.inv(AtWA) @ AtWb
    print(f"  WLS: a={params_wls[0]:.4f}, b={params_wls[1]:.4f}")

    # ============================================================
    # 4. Menghitung error (residual) untuk kedua metode
    # ============================================================
    print("\n--- 4. Perbandingan Error ---")

    # Prediksi menggunakan parameter OLS
    y_pred_ols = params_ols[0] * x_data + params_ols[1]

    # Prediksi menggunakan parameter WLS
    y_pred_wls = params_wls[0] * x_data + params_wls[1]

    # Root Mean Square Error terhadap model sebenarnya
    rmse_ols = np.sqrt(np.mean((y_pred_ols - y_true) ** 2))
    rmse_wls = np.sqrt(np.mean((y_pred_wls - y_true) ** 2))
    print(f"  RMSE OLS vs true: {rmse_ols:.4f}")
    print(f"  RMSE WLS vs true: {rmse_wls:.4f}")
    print(f"  WLS lebih baik: {rmse_wls < rmse_ols}")

    # ============================================================
    # 5. Visualisasi perbandingan OLS vs WLS
    # ============================================================
    print("\n--- 5. Visualisasi Hasil ---")

    # Membuat figure dengan 2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot kiri: data dengan ukuran titik proporsional ke bobot
    ax1 = axes[0]
    # Ukuran titik proporsional terhadap bobot (data presisi = besar)
    sizes = weights / weights.max() * 100 + 5
    ax1.scatter(x_data, y_data, s=sizes, c='steelblue', alpha=0.6, label='Data')
    ax1.plot(x_data, y_true, 'g--', linewidth=2, label='True: y=3x+5')
    ax1.plot(x_data, y_pred_ols, 'r-', linewidth=2, label=f'OLS: y={params_ols[0]:.2f}x+{params_ols[1]:.2f}')
    ax1.plot(x_data, y_pred_wls, 'b-', linewidth=2, label=f'WLS: y={params_wls[0]:.2f}x+{params_wls[1]:.2f}')
    ax1.set_title("Perbandingan OLS vs WLS\n(ukuran titik = bobot)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot kanan: distribusi bobot dan residual
    ax2 = axes[1]
    residual_ols = np.abs(y_data - y_pred_ols)
    residual_wls = np.abs(y_data - y_pred_wls)
    ax2.scatter(x_data, residual_ols, c='red', alpha=0.5, label='|Residual| OLS')
    ax2.scatter(x_data, residual_wls, c='blue', alpha=0.5, label='|Residual| WLS')
    ax2.fill_between(x_data, 0, noise_std * 2, alpha=0.1, color='gray', label='2σ noise band')
    ax2.set_title("Residual: OLS vs WLS")
    ax2.set_xlabel("x")
    ax2.set_ylabel("|Residual|")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Menyimpan hasil visualisasi ke file
    output_path = os.path.join(OUTPUT_DIR, "02_weighted_least_squares.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Disimpan: {output_path}")

    # ============================================================
    # 6. Aplikasi WLS pada Fitting Garis di Gambar
    # ============================================================
    print("\n--- 6. WLS pada Fitting Garis di Gambar ---")

    # Membuat gambar dengan titik-titik di sekitar garis
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Titik-titik tersebar di sekitar y = 0.6x + 80
    points_img = []
    weights_img = []
    for i in range(100):
        x = np.random.randint(30, 470)
        # Noise tergantung posisi (kiri lebih presisi)
        sigma = 5 + 20 * (x / 500.0)
        y = int(0.6 * x + 80 + np.random.randn() * sigma)
        if 0 <= y < 500:
            points_img.append([x, y])
            weights_img.append(1.0 / (sigma ** 2))
            # Warna biru = presisi tinggi (bobot besar), merah = presisi rendah
            blue_ratio = 1.0 - (x / 500.0)
            color = (int(255 * blue_ratio), 0, int(255 * (1 - blue_ratio)))
            cv2.circle(img, (x, y), 4, color, -1)

    points_img = np.array(points_img, dtype=np.float64)
    weights_img = np.array(weights_img)

    # OLS fit
    A_img = np.vstack([points_img[:, 0], np.ones(len(points_img))]).T
    params_ols_img, _, _, _ = np.linalg.lstsq(A_img, points_img[:, 1], rcond=None)

    # WLS fit
    W_img = np.diag(weights_img)
    params_wls_img = np.linalg.inv(A_img.T @ W_img @ A_img) @ A_img.T @ W_img @ points_img[:, 1]

    # Menggambar garis OLS (merah) dan WLS (biru) pada gambar
    x_line = np.array([20, 480])
    # Garis OLS
    y_ols = (params_ols_img[0] * x_line + params_ols_img[1]).astype(int)
    cv2.line(img, (x_line[0], y_ols[0]), (x_line[1], y_ols[1]), (0, 0, 255), 2)

    # Garis WLS
    y_wls = (params_wls_img[0] * x_line + params_wls_img[1]).astype(int)
    cv2.line(img, (x_line[0], y_wls[0]), (x_line[1], y_wls[1]), (255, 0, 0), 2)

    # Menambahkan label
    cv2.putText(img, "Merah=OLS, Biru=WLS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Titik biru=presisi tinggi", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Menyimpan gambar hasil
    output_path2 = os.path.join(OUTPUT_DIR, "02_wls_garis_gambar.png")
    cv2.imwrite(output_path2, img)
    print(f"  Disimpan: {output_path2}")

    print(f"\n  OLS gambar: a={params_ols_img[0]:.4f}, b={params_ols_img[1]:.4f}")
    print(f"  WLS gambar: a={params_wls_img[0]:.4f}, b={params_wls_img[1]:.4f}")
    print(f"  True: a=0.6, b=80")

    print("\n" + "=" * 60)
    print("PERCOBAAN 02 SELESAI")
    print("=" * 60)



if __name__ == "__main__":
    main()
