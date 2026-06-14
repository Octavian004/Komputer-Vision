

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 12: FITTING ELLIPS DAN KONTUR
    ==========================================================================
    Fitting ellips ke sekumpulan titik atau kontur menggunakan least squares.
    OpenCV menyediakan fungsi built-in untuk fitting ellips, bounding box
    berotasi, dan approximasi kontur.

    Fungsi utama:
    - cv2.fitEllipse()        : fit ellips ke kontur (minimal 5 titik)
    - cv2.fitEllipseAMS()     : fit ellips AMS (lebih akurat)
    - cv2.fitEllipseDirect()  : fit ellips Direct (lebih stabil)
    - cv2.minAreaRect()       : bounding box berotasi minimum
    - cv2.approxPolyDP()      : approximasi kontur
    - cv2.convexHull()        : convex hull dari kontur
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
    print("PERCOBAAN 12: FITTING ELLIPS DAN KONTUR")
    print("=" * 60)

    np.random.seed(42)

    # ============================================================
    # 1. Memuat gambar dengan berbagai bentuk
    # ============================================================
    print("\n--- 1. Memuat Gambar ---")

    img_path = os.path.join(IMAGE_DIR, "koin.png")
    if not os.path.exists(img_path):
        print("[ERROR] koin.png tidak ditemukan. Jalankan download_image.py!"); exit()

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold untuk mendapatkan bentuk
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"  Gambar dimuat: {img.shape}")

    # ============================================================
    # 2. Menemukan kontur
    # ============================================================
    print("\n--- 2. Menemukan Kontur ---")

    # cv2.findContours mencari semua kontur di gambar biner
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Jumlah kontur: {len(contours)}")

    # Tampilkan info setiap kontur
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        print(f"  Kontur {i}: area={area:.0f}, perimeter={perimeter:.1f}, titik={len(cnt)}")

    # ============================================================
    # 3. Fitting Ellips pada setiap kontur
    # ============================================================
    print("\n--- 3. Fitting Ellips ---")

    img_ellipse = img.copy()

    for i, cnt in enumerate(contours):
        # cv2.fitEllipse memerlukan minimal 5 titik
        if len(cnt) < 5:
            print(f"  Kontur {i}: terlalu sedikit titik untuk fitEllipse")
            continue

        # cv2.fitEllipse menggunakan least squares fitting
        # Mengembalikan ((center_x, center_y), (width, height), angle)
        ellipse = cv2.fitEllipse(cnt)
        center, axes_size, angle = ellipse

        print(f"  Kontur {i}: center=({center[0]:.1f}, {center[1]:.1f}), "
              f"size=({axes_size[0]:.1f}, {axes_size[1]:.1f}), angle={angle:.1f}°")

        # Gambar ellips hasil fitting (hijau)
        cv2.ellipse(img_ellipse, ellipse, (0, 255, 0), 2)
        # Gambar center (merah)
        cv2.circle(img_ellipse, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "12_fit_ellipse.png"), img_ellipse)

    # ============================================================
    # 4. Perbandingan metode fitEllipse
    # ============================================================
    print("\n--- 4. Perbandingan Metode Fit Ellips ---")

    img_compare = img.copy()

    for i, cnt in enumerate(contours):
        if len(cnt) < 5:
            continue

        # Metode 1: fitEllipse (standard)
        e1 = cv2.fitEllipse(cnt)
        cv2.ellipse(img_compare, e1, (0, 255, 0), 2)  # hijau

        # Metode 2: fitEllipseAMS (Approximate Mean Shift)
        e2 = cv2.fitEllipseAMS(cnt)
        cv2.ellipse(img_compare, e2, (255, 0, 0), 2)  # biru

        # Metode 3: fitEllipseDirect (Direct method — lebih stabil)
        e3 = cv2.fitEllipseDirect(cnt)
        cv2.ellipse(img_compare, e3, (0, 0, 255), 2)  # merah

        print(f"  Kontur {i}:")
        print(f"    Standard: {e1[1][0]:.1f}x{e1[1][1]:.1f}, angle={e1[2]:.1f}")
        print(f"    AMS:      {e2[1][0]:.1f}x{e2[1][1]:.1f}, angle={e2[2]:.1f}")
        print(f"    Direct:   {e3[1][0]:.1f}x{e3[1][1]:.1f}, angle={e3[2]:.1f}")

    # Label
    cv2.putText(img_compare, "Green=Std, Blue=AMS, Red=Direct", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "12_ellipse_compare.png"), img_compare)

    # ============================================================
    # 5. Minimum Area Rectangle
    # ============================================================
    print("\n--- 5. Minimum Area Rectangle ---")

    img_rect = img.copy()

    for i, cnt in enumerate(contours):
        # cv2.minAreaRect: bounding box berotasi dengan area minimum
        # Mengembalikan ((cx, cy), (w, h), angle)
        rect = cv2.minAreaRect(cnt)
        center, size, angle = rect
        print(f"  Kontur {i}: center=({center[0]:.1f}, {center[1]:.1f}), "
              f"size=({size[0]:.1f}, {size[1]:.1f}), angle={angle:.1f}°")

        # cv2.boxPoints: konversi RotatedRect ke 4 titik sudut
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # konversi ke integer
        cv2.drawContours(img_rect, [box], 0, (0, 255, 255), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "12_min_area_rect.png"), img_rect)

    # ============================================================
    # 6. convexHull dan Convexity Defects
    # ============================================================
    print("\n--- 6. Convex Hull ---")

    img_hull = img.copy()

    for i, cnt in enumerate(contours):
        # cv2.convexHull: convex hull dari kontur
        hull = cv2.convexHull(cnt)

        # Gambar hull (kuning)
        cv2.drawContours(img_hull, [hull], 0, (0, 255, 255), 2)

        # Hitung convexity
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull)
        convexity = area_cnt / area_hull if area_hull > 0 else 0

        print(f"  Kontur {i}: convexity={convexity:.4f} "
              f"({'convex' if convexity > 0.95 else 'non-convex'})")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "12_convex_hull.png"), img_hull)

    # ============================================================
    # 7. Approx Polygon
    # ============================================================
    print("\n--- 7. Approximasi Poligon ---")

    img_approx = img.copy()

    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)

        # Coba berbagai epsilon (persentase keliling)
        for eps_pct in [0.01, 0.02, 0.05]:
            epsilon = eps_pct * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            print(f"  Kontur {i}, eps={eps_pct}: {len(cnt)} → {len(approx)} titik")

        # Gambar dengan epsilon 2%
        approx_draw = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        cv2.drawContours(img_approx, [approx_draw], 0, (0, 255, 0), 2)
        for pt in approx_draw:
            cv2.circle(img_approx, tuple(pt[0]), 5, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "12_approx_polygon.png"), img_approx)

    # ============================================================
    # 8. Visualisasi gabungan
    # ============================================================
    print("\n--- 8. Visualisasi Gabungan ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    titles = ["Original", "Fit Ellipse", "Ellipse Comparison",
              "Min Area Rect", "Convex Hull", "Approx Polygon"]
    images = [img, img_ellipse, img_compare, img_rect, img_hull, img_approx]

    for ax, title, im in zip(axes.flat, titles, images):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "12_fitting_kontur_all.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Disimpan: {output_path}")

    print("\n" + "=" * 60)
    print("PERCOBAAN 12 SELESAI")
    print("=" * 60)



if __name__ == "__main__":
    main()
