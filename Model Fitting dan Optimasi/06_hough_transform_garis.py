

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 06: HOUGH TRANSFORM - DETEKSI GARIS
    ==========================================================================
    Hough Transform mengubah titik-titik edge di image space menjadi kurva
    di parameter space (rho, theta). Garis terdeteksi sebagai peak
    di accumulator space.

    Persamaan: rho = x*cos(theta) + y*sin(theta)

    Fungsi utama:
    - cv2.Canny()             : deteksi tepi untuk input Hough
    - cv2.HoughLines()        : Standard Hough Transform (mengembalikan rho, theta)
    - cv2.HoughLinesP()       : Probabilistic Hough (mengembalikan endpoint)
    - cv2.line()              : menggambar garis hasil deteksi
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
    print("PERCOBAAN 06: HOUGH TRANSFORM - DETEKSI GARIS")
    print("=" * 60)

    # ============================================================
    # 1. Mempersiapkan gambar dengan garis-garis (jalan.png)
    # ============================================================
    print("\n--- 1. Memuat Gambar ---")

    # Membaca gambar jalan yang dibuat oleh download_image.py
    img_path = os.path.join(IMAGE_DIR, "jalan.png")

    # Jika gambar tidak ditemukan, buat gambar sintetis dengan garis
    if not os.path.exists(img_path):
        print("[ERROR] img tidak ditemukan. Jalankan download_image.py!"); exit()

    # Membaca gambar input
    img = cv2.imread(img_path)
    print(f"  Ukuran gambar: {img.shape}")

    # ============================================================
    # 2. Preprocessing: konversi grayscale + deteksi tepi Canny
    # ============================================================
    print("\n--- 2. Preprocessing (Grayscale + Canny) ---")

    # Konversi ke grayscale karena Hough bekerja pada gambar biner
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur untuk mengurangi noise sebelum deteksi tepi
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepi Canny — menghasilkan gambar biner tepi
    # Parameter: threshold_low=50, threshold_high=150
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    print(f"  Jumlah piksel edge: {np.count_nonzero(edges)}")

    # Simpan gambar edge
    cv2.imwrite(os.path.join(OUTPUT_DIR, "06_edges_canny.png"), edges)

    # ============================================================
    # 3. Standard Hough Transform (cv2.HoughLines)
    # ============================================================
    print("\n--- 3. Standard Hough Transform ---")

    # cv2.HoughLines(edges, rho_resolution, theta_resolution, threshold)
    # - edges: gambar biner hasil Canny
    # - rho=1: resolusi rho dalam piksel
    # - theta=pi/180: resolusi theta dalam radian (1 derajat)
    # - threshold=100: minimal vote untuk dianggap garis
    lines_std = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Gambar garis pada salinan gambar asli
    img_std = img.copy()

    if lines_std is not None:
        print(f"  Standard Hough: {len(lines_std)} garis terdeteksi")
        for line in lines_std:
            rho, theta = line[0]
            # Konversi parameter (rho, theta) ke 2 titik untuk menggambar
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # Titik-titik jauh di sepanjang garis
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(img_std, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("  Tidak ada garis terdeteksi")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "06_hough_standard.png"), img_std)

    # ============================================================
    # 4. Probabilistic Hough Transform (cv2.HoughLinesP)
    # ============================================================
    print("\n--- 4. Probabilistic Hough Transform ---")

    # cv2.HoughLinesP mengembalikan endpoint (x1,y1,x2,y2) langsung
    # - minLineLength: panjang minimum garis (abaikan garis pendek)
    # - maxLineGap: jarak gap maksimum yang masih dianggap satu garis
    lines_prob = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                  threshold=50,
                                  minLineLength=30,
                                  maxLineGap=10)

    # Gambar hasil pada salinan gambar
    img_prob = img.copy()

    if lines_prob is not None:
        print(f"  Probabilistic Hough: {len(lines_prob)} segmen garis")
        for line in lines_prob:
            x1, y1, x2, y2 = line[0]
            # Warna acak untuk setiap garis agar mudah dibedakan
            color = tuple(np.random.randint(50, 255, 3).tolist())
            cv2.line(img_prob, (x1, y1), (x2, y2), color, 2)
    else:
        print("  Tidak ada segmen terdeteksi")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "06_hough_probabilistic.png"), img_prob)

    # ============================================================
    # 5. Membuat Accumulator Space secara manual
    # ============================================================
    print("\n--- 5. Visualisasi Accumulator Space ---")

    # Ambil koordinat piksel edge
    edge_y, edge_x = np.where(edges > 0)

    # Range parameter
    theta_range = np.linspace(-np.pi/2, np.pi/2, 180)
    rho_max = int(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))

    # Membuat accumulator array
    accumulator = np.zeros((2 * rho_max + 1, len(theta_range)), dtype=np.int32)

    # Voting: setiap piksel edge memberikan vote ke semua garis yang mungkin
    for x, y in zip(edge_x, edge_y):
        for t_idx, theta in enumerate(theta_range):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            accumulator[rho + rho_max, t_idx] += 1

    # Visualisasi accumulator
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot accumulator
    ax1 = axes[0]
    ax1.imshow(accumulator, cmap='hot', aspect='auto',
               extent=[np.degrees(theta_range[0]), np.degrees(theta_range[-1]),
                       -rho_max, rho_max])
    ax1.set_title("Hough Accumulator Space")
    ax1.set_xlabel("Theta (derajat)")
    ax1.set_ylabel("Rho (piksel)")

    # Plot edge
    ax2 = axes[1]
    ax2.imshow(edges, cmap='gray')
    ax2.set_title("Edge (Canny)")

    # Plot hasil Standard Hough
    ax3 = axes[2]
    ax3.imshow(cv2.cvtColor(img_std, cv2.COLOR_BGR2RGB))
    ax3.set_title("Standard Hough Lines")

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "06_hough_accumulator.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Disimpan: {output_path}")

    # ============================================================
    # 6. Pengaruh parameter threshold
    # ============================================================
    print("\n--- 6. Pengaruh Threshold ---")

    for thresh in [30, 60, 100, 150, 200]:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=thresh)
        n = len(lines) if lines is not None else 0
        print(f"  Threshold={thresh:3d}: {n:3d} garis")

    # ============================================================
    # 7. Pengaruh parameter HoughLinesP
    # ============================================================
    print("\n--- 7. Variasi Parameter HoughLinesP ---")

    configs = [
        (30, 20, 5),   # threshold, minLineLength, maxLineGap
        (50, 30, 10),
        (50, 50, 20),
        (80, 80, 30),
    ]
    for thresh, minLen, maxGap in configs:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                 threshold=thresh, minLineLength=minLen, maxLineGap=maxGap)
        n = len(lines) if lines is not None else 0
        print(f"  thresh={thresh:3d}, minLen={minLen:3d}, maxGap={maxGap:3d}: {n:3d} segmen")

    print("\n" + "=" * 60)
    print("PERCOBAAN 06 SELESAI")
    print("=" * 60)



if __name__ == "__main__":
    main()
