

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 20: PIPELINE GABUNGAN MODEL FITTING
    ==========================================================================
    Percobaan terakhir menggabungkan berbagai teknik model fitting dan
    optimasi menjadi satu pipeline end-to-end. Pipeline ini mencakup:
    1. Preprocessing → 2. Feature Detection → 3. Model Fitting (RANSAC) →
    4. Evaluasi → 5. Visualisasi

    Ini mensimulasikan workflow nyata dalam computer vision di mana
    beberapa metode digabung untuk menyelesaikan task tertentu.
    ==========================================================================
    """

    import cv2
    import numpy as np
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import time

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PERCOBAAN 20: PIPELINE GABUNGAN MODEL FITTING")
    print("=" * 60)

    np.random.seed(42)

    # ============================================================
    # PIPELINE 1: Deteksi Garis pada Gambar Jalan
    # ============================================================
    print("\n" + "=" * 50)
    print("PIPELINE 1: DETEKSI GARIS JALAN")
    print("=" * 50)

    # --- Step 1: Load dan preprocessing ---
    print("\n--- Step 1: Preprocessing ---")

    img_path = os.path.join(IMAGE_DIR, "jalan.png")
    if not os.path.exists(img_path):
        print("[ERROR] Gambar tidak ditemukan. Jalankan download_image.py!"); exit()

    img_jalan = cv2.imread(img_path)
    gray = cv2.cvtColor(img_jalan, cv2.COLOR_BGR2GRAY)

    # Gaussian blur untuk noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    print(f"  Gambar: {img_jalan.shape}")
    print(f"  Edge pixels: {np.sum(edges > 0)}")

    # --- Step 2: Hough Transform untuk kandidat garis ---
    print("\n--- Step 2: Hough Transform ---")

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    n_lines = len(lines) if lines is not None else 0
    print(f"  Kandidat garis: {n_lines}")

    # --- Step 3: RANSAC fitting untuk garis dominan ---
    print("\n--- Step 3: RANSAC Filtering ---")

    if lines is not None:
        # Kumpulkan semua titik pada garis
        all_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            all_points.extend([(x1, y1), (x2, y2)])

        all_points = np.array(all_points, dtype=np.float32)

        # Pisahkan garis berdasarkan kemiringan (kiri vs kanan)
        left_points = []
        right_points = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Garis kiri: slope negatif (di image coordinate)
            if slope < -0.3:
                left_points.extend([(x1, y1), (x2, y2)])
            elif slope > 0.3:
                right_points.extend([(x1, y1), (x2, y2)])

        print(f"  Titik garis kiri: {len(left_points)}")
        print(f"  Titik garis kanan: {len(right_points)}")

    # --- Step 4: cv2.fitLine pada setiap grup ---
    print("\n--- Step 4: Line Fitting ---")

    img_pipeline1 = img_jalan.copy()

    for name, pts in [("Kiri", left_points), ("Kanan", right_points)]:
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            # cv2.fitLine dengan metode DIST_HUBER (robust)
            vx, vy, cx, cy = cv2.fitLine(pts_arr, cv2.DIST_HUBER, 0, 0.01, 0.01)

            # Gambar garis fit (extrapolasi)
            t = 500
            x1_fit = int(cx - t * vx)
            y1_fit = int(cy - t * vy)
            x2_fit = int(cx + t * vx)
            y2_fit = int(cy + t * vy)

            color = (0, 0, 255) if name == "Kiri" else (255, 0, 0)
            cv2.line(img_pipeline1, (x1_fit, y1_fit), (x2_fit, y2_fit), color, 3)
            print(f"  {name}: direction=({vx[0]:.3f}, {vy[0]:.3f}), center=({cx[0]:.1f}, {cy[0]:.1f})")

    # Gambar edges dan garis original
    for line in (lines if lines is not None else []):
        x1, y1, x2, y2 = line[0]
        cv2.line(img_pipeline1, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "20_pipeline1_lane.png"), img_pipeline1)

    # ============================================================
    # PIPELINE 2: Estimasi Transformasi Geometri
    # ============================================================
    print("\n" + "=" * 50)
    print("PIPELINE 2: ESTIMASI TRANSFORMASI")
    print("=" * 50)

    # --- Step 1: Generate dua gambar dengan transformasi ---
    print("\n--- Step 1: Generate Data ---")

    img_src = np.zeros((300, 400, 3), dtype=np.uint8)
    pts_pattern = [(50, 50), (350, 50), (350, 250), (50, 250),
                   (200, 150), (100, 200), (300, 100)]
    for pt in pts_pattern:
        cv2.circle(img_src, pt, 8, (255, 255, 255), -1)
        cv2.circle(img_src, pt, 15, (100, 200, 100), 2)

    # Apply known transformation
    angle = 10  # derajat
    scale = 0.95
    tx, ty = 30, 20
    M_true = cv2.getRotationMatrix2D((200, 150), angle, scale)
    M_true[0, 2] += tx
    M_true[1, 2] += ty
    img_dst = cv2.warpAffine(img_src, M_true, (400, 300))

    # Tambah noise
    img_dst = np.clip(img_dst.astype(np.int16) + 
                      np.random.normal(0, 5, img_dst.shape).astype(np.int16),
                      0, 255).astype(np.uint8)

    print(f"  Transformasi true: angle={angle}°, scale={scale}, tx={tx}, ty={ty}")

    # --- Step 2: Feature matching ---
    print("\n--- Step 2: Feature Matching ---")

    orb = cv2.ORB_create(500)
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray_src, None)
    kp2, des2 = orb.detectAndCompute(gray_dst, None)

    if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        print(f"  Keypoints src: {len(kp1)}, dst: {len(kp2)}")
        print(f"  Matches: {len(matches)}")

        # --- Step 3: Estimasi affine dengan RANSAC ---
        print("\n--- Step 3: Affine Estimation ---")

        if len(matches) >= 3:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # cv2.estimateAffine2D dengan RANSAC
            M_est, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)

            if M_est is not None:
                inlier_count = np.sum(inliers)
                print(f"  Inliers: {inlier_count}/{len(inliers)}")
                print(f"  Estimated matrix:")
                for row in M_est:
                    print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
                print(f"  True matrix:")
                for row in M_true:
                    print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")

                # Hitung error
                error = np.abs(M_est - M_true).mean()
                print(f"  Mean absolute error: {error:.4f}")

                # Warp dan overlay
                warped_est = cv2.warpAffine(img_src, M_est, (400, 300))
                overlay = cv2.addWeighted(warped_est, 0.5, img_dst, 0.5, 0)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "20_pipeline2_overlay.png"), overlay)
    else:
        print("  Tidak cukup fitur untuk matching!")

    # ============================================================
    # PIPELINE 3: Benchmark Metode Fitting
    # ============================================================
    print("\n" + "=" * 50)
    print("PIPELINE 3: BENCHMARK METODE FITTING")
    print("=" * 50)

    # Generate data garis dengan outlier
    n_inlier = 100
    n_outlier = 30
    x_in = np.random.uniform(0, 100, n_inlier)
    y_in = 2.5 * x_in + 15 + np.random.normal(0, 3, n_inlier)
    x_out = np.random.uniform(0, 100, n_outlier)
    y_out = np.random.uniform(-50, 300, n_outlier)
    x_data = np.concatenate([x_in, x_out])
    y_data = np.concatenate([y_in, y_out])

    print(f"\n  Data: {n_inlier} inliers + {n_outlier} outliers")
    print(f"  True: y = 2.5x + 15")

    # --- Metode 1: OLS ---
    print("\n--- Metode 1: OLS ---")
    t0 = time.time()
    coeffs_ols = np.polyfit(x_data, y_data, 1)
    t_ols = time.time() - t0
    print(f"  y = {coeffs_ols[0]:.3f}x + {coeffs_ols[1]:.3f} (time: {t_ols*1000:.1f}ms)")

    # --- Metode 2: RANSAC ---
    print("\n--- Metode 2: RANSAC ---")
    t0 = time.time()
    best_inliers = 0
    best_line = None
    for _ in range(200):
        idx = np.random.choice(len(x_data), 2, replace=False)
        x_s = x_data[idx]
        y_s = y_data[idx]
        if abs(x_s[1] - x_s[0]) < 1e-6:
            continue
        slope = (y_s[1] - y_s[0]) / (x_s[1] - x_s[0])
        intercept = y_s[0] - slope * x_s[0]
        residuals = np.abs(y_data - (slope * x_data + intercept))
        inlier_mask = residuals < 10
        if np.sum(inlier_mask) > best_inliers:
            best_inliers = np.sum(inlier_mask)
            best_line = (slope, intercept)
            best_mask = inlier_mask

    # Refit pada inliers
    coeffs_ransac = np.polyfit(x_data[best_mask], y_data[best_mask], 1)
    t_ransac = time.time() - t0
    print(f"  y = {coeffs_ransac[0]:.3f}x + {coeffs_ransac[1]:.3f} (time: {t_ransac*1000:.1f}ms)")
    print(f"  Inliers: {best_inliers}")

    # --- Metode 3: cv2.fitLine (Huber) ---
    print("\n--- Metode 3: cv2.fitLine (Huber) ---")
    t0 = time.time()
    pts_fit = np.column_stack([x_data, y_data]).astype(np.float32).reshape(-1, 1, 2)
    vx, vy, cx, cy = cv2.fitLine(pts_fit, cv2.DIST_HUBER, 0, 0.01, 0.01)
    slope_huber = vy[0] / vx[0]
    intercept_huber = cy[0] - slope_huber * cx[0]
    t_huber = time.time() - t0
    print(f"  y = {slope_huber:.3f}x + {intercept_huber:.3f} (time: {t_huber*1000:.1f}ms)")

    # --- Metode 4: cv2.fitLine (Tukey) ---
    print("\n--- Metode 4: cv2.fitLine (Tukey) ---")
    vx, vy, cx, cy = cv2.fitLine(pts_fit, cv2.DIST_WELSCH, 0, 0.01, 0.01)
    slope_tukey = vy[0] / vx[0]
    intercept_tukey = cy[0] - slope_tukey * cx[0]
    print(f"  y = {slope_tukey:.3f}x + {intercept_tukey:.3f}")

    # --- Evaluasi ---
    print("\n--- Evaluasi ---")
    true_slope, true_intercept = 2.5, 15.0

    results = {
        'OLS': (coeffs_ols[0], coeffs_ols[1], t_ols),
        'RANSAC': (coeffs_ransac[0], coeffs_ransac[1], t_ransac),
        'Huber': (slope_huber, intercept_huber, t_huber),
        'Tukey': (slope_tukey, intercept_tukey, t_huber),
    }

    print(f"  {'Method':<10} {'Slope err':>10} {'Intercept err':>14} {'Time':>10}")
    for name, (s, i, t) in results.items():
        se = abs(s - true_slope)
        ie = abs(i - true_intercept)
        print(f"  {name:<10} {se:10.4f} {ie:14.4f} {t*1000:8.1f}ms")

    # ============================================================
    # Visualisasi gabungan semua pipeline
    # ============================================================
    print("\n--- Visualisasi Gabungan ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Pipeline 1: Lane detection
    axes[0, 0].imshow(cv2.cvtColor(img_pipeline1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Pipeline 1: Lane Detection")
    axes[0, 0].axis('off')

    # Pipeline 1: Edges
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title("Canny Edges")
    axes[0, 1].axis('off')

    # Pipeline 2: Feature matching overlay
    if 'overlay' in dir():
        axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Pipeline 2: Transform Estimation")
    else:
        axes[0, 2].text(0.5, 0.5, "N/A", ha='center')
    axes[0, 2].axis('off')

    # Pipeline 3: Data + all fits
    ax3 = axes[1, 0]
    ax3.scatter(x_in, y_in, c='blue', s=10, alpha=0.5, label='Inliers')
    ax3.scatter(x_out, y_out, c='red', s=10, alpha=0.5, label='Outliers')
    x_range = np.linspace(0, 100, 100)
    ax3.plot(x_range, true_slope * x_range + true_intercept, 'k--', linewidth=2, label='True')
    ax3.plot(x_range, coeffs_ols[0] * x_range + coeffs_ols[1], 'g-', label='OLS')
    ax3.plot(x_range, coeffs_ransac[0] * x_range + coeffs_ransac[1], 'r-', label='RANSAC')
    ax3.plot(x_range, slope_huber * x_range + intercept_huber, 'm-', label='Huber')
    ax3.legend(fontsize=8)
    ax3.set_title("Pipeline 3: Fitting Comparison")
    ax3.grid(True, alpha=0.3)

    # Error comparison bar chart
    ax_bar = axes[1, 1]
    method_names = list(results.keys())
    slope_errors = [abs(results[m][0] - true_slope) for m in method_names]
    ax_bar.bar(method_names, slope_errors, color=['green', 'red', 'purple', 'orange'])
    ax_bar.set_ylabel("Slope Error")
    ax_bar.set_title("Fitting Error Comparison")
    ax_bar.grid(True, alpha=0.3)

    # Summary text
    ax_text = axes[1, 2]
    ax_text.axis('off')
    summary = "RINGKASAN PIPELINE\n" + "=" * 35 + "\n\n"
    summary += "Pipeline 1: Lane Detection\n"
    summary += f"  - Canny + Hough + fitLine\n"
    summary += f"  - {n_lines} garis terdeteksi\n\n"
    summary += "Pipeline 2: Transform Estimation\n"
    summary += f"  - ORB + BFMatcher + RANSAC\n"
    if 'error' in dir():
        summary += f"  - MAE: {error:.4f}\n\n"
    summary += "Pipeline 3: Benchmark\n"
    summary += f"  - Terbaik: RANSAC\n"
    summary += f"  - Data: {n_inlier}+{n_outlier} pts\n"
    ax_text.text(0.1, 0.9, summary, transform=ax_text.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "20_pipeline_gabungan_all.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"\n  Disimpan: {output_path}")

    print("\n" + "=" * 60)
    print("PERCOBAAN 20 SELESAI")
    print("=" * 60)
    print("\n>>> SEMUA 20 PERCOBAAN MODUL 04 TELAH SELESAI <<<")



if __name__ == "__main__":
    main()
