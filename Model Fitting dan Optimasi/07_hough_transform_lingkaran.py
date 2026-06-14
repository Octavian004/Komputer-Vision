

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 07: HOUGH TRANSFORM - DETEKSI LINGKARAN
    ==========================================================================
    Hough Circle Transform mendeteksi lingkaran di gambar. Setiap piksel
    edge memberikan vote di parameter space 3D (cx, cy, r).

    OpenCV menggunakan metode Hough Gradient (2-pass):
    1. Tentukan center candidates dari gradient edge
    2. Tentukan radius terbaik untuk setiap center

    Fungsi utama:
    - cv2.HoughCircles()      : deteksi lingkaran
    - cv2.Canny()             : deteksi tepi (internal)
    - cv2.GaussianBlur()      : preprocessing
    - cv2.circle()            : menggambar lingkaran hasil
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
    print("PERCOBAAN 07: HOUGH TRANSFORM - DETEKSI LINGKARAN")
    print("=" * 60)

    # ============================================================
    # 1. Memuat gambar koin (lingkaran-lingkaran)
    # ============================================================
    print("\n--- 1. Memuat Gambar ---")

    img_path = os.path.join(IMAGE_DIR, "koin.png")

    # Jika koin.png tidak ada, buat gambar sintetis
    if not os.path.exists(img_path):
        print("[ERROR] img tidak ditemukan. Jalankan download_image.py!"); exit()

    img = cv2.imread(img_path)
    print(f"  Ukuran: {img.shape}")

    # ============================================================
    # 2. Preprocessing
    # ============================================================
    print("\n--- 2. Preprocessing ---")

    # Konversi ke grayscale (wajib untuk HoughCircles)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur: mengurangi noise yang menyebabkan false positive
    # ksize harus ganjil — makin besar makin smooth
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    print(f"  Blur kernel: 9x9, sigma=2")

    # ============================================================
    # 3. Deteksi lingkaran dengan HoughCircles
    # ============================================================
    print("\n--- 3. HoughCircles ---")

    # cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)
    # - method: cv2.HOUGH_GRADIENT (satu-satunya yang tersedia)
    # - dp=1: rasio resolusi accumulator (1=sama dengan gambar)
    # - minDist=50: jarak minimum antar center lingkaran yang terdeteksi
    # - param1=100: threshold atas untuk Canny internal
    # - param2=30: threshold accumulator untuk center detection
    # - minRadius=20: radius minimum
    # - maxRadius=100: radius maksimum
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    # Gambar hasil pada salinan gambar
    img_result = img.copy()

    if circles is not None:
        # HoughCircles mengembalikan float, konversi ke integer
        circles_int = np.round(circles[0]).astype(int)
        print(f"  Lingkaran terdeteksi: {len(circles_int)}")

        for i, (cx, cy, r) in enumerate(circles_int):
            # Gambar lingkaran (hijau)
            cv2.circle(img_result, (cx, cy), r, (0, 255, 0), 2)
            # Gambar titik center (merah)
            cv2.circle(img_result, (cx, cy), 3, (0, 0, 255), -1)
            # Label
            cv2.putText(img_result, f"r={r}", (cx - 20, cy - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            print(f"  #{i+1}: center=({cx}, {cy}), radius={r}")
    else:
        print("  Tidak ada lingkaran terdeteksi")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "07_hough_circles.png"), img_result)

    # ============================================================
    # 4. Pengaruh parameter dp
    # ============================================================
    print("\n--- 4. Pengaruh Parameter dp ---")

    for dp in [0.5, 1, 1.5, 2]:
        c = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp,
                              minDist=50, param1=100, param2=30,
                              minRadius=20, maxRadius=100)
        n = len(c[0]) if c is not None else 0
        print(f"  dp={dp:.1f}: {n} lingkaran")

    # ============================================================
    # 5. Pengaruh parameter param2 (akumulasi threshold)
    # ============================================================
    print("\n--- 5. Pengaruh param2 ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    param2_values = [10, 20, 30, 40, 50, 60]

    for idx, p2 in enumerate(param2_values):
        c = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                              minDist=50, param1=100, param2=p2,
                              minRadius=20, maxRadius=100)

        # Gambar hasil
        img_temp = img.copy()
        n = 0
        if c is not None:
            c_int = np.round(c[0]).astype(int)
            n = len(c_int)
            for cx, cy, r in c_int:
                cv2.circle(img_temp, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(img_temp, (cx, cy), 3, (0, 0, 255), -1)

        ax = axes[idx // 3][idx % 3]
        ax.imshow(cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB))
        ax.set_title(f"param2={p2} ({n} lingkaran)")
        ax.axis('off')

        print(f"  param2={p2:3d}: {n} lingkaran")

    plt.suptitle("Pengaruh param2 pada HoughCircles", fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "07_hough_circles_param2.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Disimpan: {output_path}")

    # ============================================================
    # 6. Deteksi pada gambar dengan noise
    # ============================================================
    print("\n--- 6. Deteksi pada Gambar Noisy ---")

    # Tambahkan noise Gaussian ke gambar
    for sigma in [0, 10, 30, 50]:
        if sigma > 0:
            noise = np.random.randn(*img.shape) * sigma
            img_noisy = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        else:
            img_noisy = img.copy()

        gray_n = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)
        blur_n = cv2.GaussianBlur(gray_n, (9, 9), 2)

        c = cv2.HoughCircles(blur_n, cv2.HOUGH_GRADIENT, dp=1,
                              minDist=50, param1=100, param2=30,
                              minRadius=20, maxRadius=100)
        n = len(c[0]) if c is not None else 0
        print(f"  Noise σ={sigma:2d}: {n} lingkaran terdeteksi")

    # ============================================================
    # 7. Membuat gambar sintetis dengan banyak lingkaran
    # ============================================================
    print("\n--- 7. Deteksi Banyak Lingkaran ---")

    # Buat gambar dengan 15 lingkaran acak
    img_many = np.ones((600, 800, 3), dtype=np.uint8) * 220
    np.random.seed(42)
    true_circles = []
    for _ in range(15):
        cx = np.random.randint(60, 740)
        cy = np.random.randint(60, 540)
        r = np.random.randint(15, 60)
        color = tuple(np.random.randint(40, 200, 3).tolist())
        cv2.circle(img_many, (cx, cy), r, color, -1)
        cv2.circle(img_many, (cx, cy), r, (0, 0, 0), 2)
        true_circles.append((cx, cy, r))

    # Deteksi
    gray_many = cv2.cvtColor(img_many, cv2.COLOR_BGR2GRAY)
    blur_many = cv2.GaussianBlur(gray_many, (9, 9), 2)
    c_many = cv2.HoughCircles(blur_many, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=40, param1=80, param2=25,
                               minRadius=10, maxRadius=80)

    img_detect = img_many.copy()
    n_detect = 0
    if c_many is not None:
        c_int = np.round(c_many[0]).astype(int) 
        n_detect = len(c_int)
        for cx, cy, r in c_int:
            cv2.circle(img_detect, (cx, cy), r, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "07_hough_banyak_lingkaran.png"), img_detect)
    print(f"  True: {len(true_circles)} lingkaran, Detected: {n_detect}")

    print("\n" + "=" * 60)
    print("PERCOBAAN 07 SELESAI")
    print("=" * 60)



if __name__ == "__main__":
    main()
