

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 1: IMAGE STITCHING SEDERHANA (MANUAL PIPELINE)
    ==========================================================================
    Program ini membangun pipeline stitching gambar dari awal secara manual.
    Tahapan pipeline meliputi: deteksi fitur → pencocokan fitur → estimasi
    homography → warping perspektif → penggabungan gambar.

    Konsep yang dipelajari:
    - Pipeline lengkap image stitching dari nol
    - Deteksi fitur menggunakan SIFT (Scale-Invariant Feature Transform)
    - Pencocokan fitur menggunakan FLANN-based matcher
    - Lowe's ratio test untuk memfilter kecocokan yang baik
    - Estimasi matriks homography menggunakan RANSAC
    - Warping perspektif untuk menyelaraskan gambar
    - Penggabungan (compositing) gambar pada canvas bersama

    Fungsi utama yang dipelajari:
    - cv2.SIFT_create()        : Membuat detektor fitur SIFT
    - cv2.FlannBasedMatcher()   : Membuat matcher berbasis FLANN (Fast Library
                                  for Approximate Nearest Neighbors)
    - cv2.findHomography()      : Mengestimasi matriks homography 3x3 dengan RANSAC
    - cv2.warpPerspective()     : Melakukan perspective warping pada gambar
    - cv2.drawMatches()         : Menggambar visualisasi hasil matching fitur
    - cv2.perspectiveTransform(): Mentransformasi titik menggunakan homography
    ==========================================================================
    """

    # Mengimpor library OpenCV untuk pemrosesan gambar dan computer vision
    import cv2

    # Mengimpor library NumPy untuk operasi array dan matriks
    import numpy as np

    # Mengimpor library os untuk operasi path file dan folder
    import os

    # Mengimpor matplotlib untuk visualisasi dan penyimpanan grafik perbandingan
    import matplotlib.pyplot as plt

    # Mengimpor modul time untuk mengukur waktu eksekusi
    import time

    # ============================================================
    # SETUP PATH DIREKTORI
    # ============================================================

    # Mendapatkan direktori tempat script ini berada
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Mendefinisikan path folder gambar input
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

    # Mendefinisikan path folder output untuk menyimpan hasil
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    # Membuat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # HEADER PROGRAM
    # ============================================================
    print("=" * 65)
    print("PERCOBAAN 1: IMAGE STITCHING SEDERHANA (MANUAL PIPELINE)")
    print("=" * 65)

    # ============================================================
    # LANGKAH 1: Memuat Gambar Pasangan (Pair Images)
    # ============================================================
    print("\n[LANGKAH 1] Memuat gambar pasangan untuk stitching...")

    # Membaca gambar kiri dari file pair_left.jpg
    img_left = cv2.imread(os.path.join(IMAGE_DIR, "pair_left.jpg"))

    # Membaca gambar kanan dari file pair_right.jpg
    img_right = cv2.imread(os.path.join(IMAGE_DIR, "pair_right.jpg"))

    # Memastikan kedua gambar berhasil dimuat
    if img_left is None or img_right is None:
        print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
        exit()

    # Menampilkan informasi dimensi kedua gambar
    print(f"  Gambar kiri  : {img_left.shape[1]}x{img_left.shape[0]} piksel")
    print(f"  Gambar kanan : {img_right.shape[1]}x{img_right.shape[0]} piksel")

    # Mengkonversi gambar ke grayscale untuk deteksi fitur
    # SIFT bekerja pada gambar grayscale (intensitas tunggal)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Menampilkan informasi konversi grayscale
    print(f"  Konversi grayscale berhasil untuk kedua gambar.")

    # ============================================================
    # LANGKAH 2: Deteksi Fitur SIFT pada Kedua Gambar
    # ============================================================
    print("\n[LANGKAH 2] Mendeteksi fitur SIFT pada kedua gambar...")

    # Membuat objek detektor SIFT (Scale-Invariant Feature Transform)
    # SIFT mendeteksi keypoints yang invariant terhadap skala dan rotasi
    sift = cv2.SIFT_create()

    # Mendeteksi keypoints dan menghitung deskriptor pada gambar kiri
    # keypoints = lokasi fitur yang terdeteksi (x, y, size, angle)
    # descriptors = vektor 128-dimensi yang mendeskripsikan setiap keypoint
    kp_left, desc_left = sift.detectAndCompute(gray_left, None)

    # Mendeteksi keypoints dan menghitung deskriptor pada gambar kanan
    kp_right, desc_right = sift.detectAndCompute(gray_right, None)

    # Menampilkan jumlah keypoints yang terdeteksi
    print(f"  Keypoints gambar kiri  : {len(kp_left)} fitur")
    print(f"  Keypoints gambar kanan : {len(kp_right)} fitur")
    print(f"  Dimensi deskriptor     : {desc_left.shape[1]} (SIFT = 128-D)")

    # Menggambar keypoints pada kedua gambar untuk visualisasi
    img_kp_left = cv2.drawKeypoints(img_left, kp_left, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_kp_right = cv2.drawKeypoints(img_right, kp_right, None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Menyimpan visualisasi keypoints
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_keypoints_kiri.jpg"), img_kp_left)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_keypoints_kanan.jpg"), img_kp_right)
    print("  [OK] Visualisasi keypoints disimpan.")

    # ============================================================
    # LANGKAH 3: Mencocokkan Fitur Menggunakan FLANN Matcher
    # ============================================================
    print("\n[LANGKAH 3] Mencocokkan fitur menggunakan FLANN matcher...")

    # Mengonfigurasi parameter FLANN (Fast Library for Approximate Nearest Neighbors)
    # FLANN_INDEX_KDTREE = 1: menggunakan algoritma KD-Tree untuk pencarian
    FLANN_INDEX_KDTREE = 1

    # Parameter untuk indexing: menggunakan KD-Tree dengan 5 pohon pencarian
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    # Parameter pencarian: memeriksa 50 node untuk setiap pencarian
    search_params = dict(checks=50)

    # Membuat objek FLANN-based matcher dengan parameter yang sudah dikonfigurasi
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Melakukan pencarian k-nearest neighbors (k=2) untuk setiap deskriptor
    # knnMatch mengembalikan 2 tetangga terdekat untuk setiap fitur
    matches_knn = flann.knnMatch(desc_left, desc_right, k=2)

    # Menampilkan jumlah total kecocokan mentah
    print(f"  Total kecocokan mentah (raw matches): {len(matches_knn)}")

    # ============================================================
    # LANGKAH 4: Lowe's Ratio Test untuk Memfilter Kecocokan
    # ============================================================
    print("\n[LANGKAH 4] Menerapkan Lowe's ratio test (rasio=0.75)...")

    # Menerapkan Lowe's ratio test dengan threshold default 0.75
    # Logika: jarak ke tetangga terdekat harus jauh lebih kecil dari tetangga kedua
    # Jika m.distance < 0.75 * n.distance, maka kecocokan dianggap baik
    ratio_threshold = 0.75
    good_matches = []
    for m, n in matches_knn:
        # m = tetangga terdekat (best match), n = tetangga terdekat kedua
        if m.distance < ratio_threshold * n.distance:
            # Kecocokan dianggap baik jika best match jauh lebih dekat dari runner-up
            good_matches.append(m)

    # Menampilkan jumlah kecocokan yang lolos ratio test
    print(f"  Kecocokan baik (good matches): {len(good_matches)}")
    print(f"  Persentase lolos: {len(good_matches)/len(matches_knn)*100:.1f}%")

    # ============================================================
    # LANGKAH 5: Menggambar Visualisasi Matches
    # ============================================================
    print("\n[LANGKAH 5] Menggambar visualisasi kecocokan fitur...")

    # Mengurutkan kecocokan berdasarkan jarak (semakin kecil = semakin baik)
    good_matches_sorted = sorted(good_matches, key=lambda x: x.distance)

    # Menggambar 50 kecocokan terbaik pada kedua gambar
    img_matches = cv2.drawMatches(
        img_left, kp_left,           # Gambar kiri dan keypoints-nya
        img_right, kp_right,         # Gambar kanan dan keypoints-nya
        good_matches_sorted[:50],    # 50 kecocokan terbaik
        None,                        # Output image (None = buat baru)
        matchColor=(0, 255, 0),      # Warna garis kecocokan (hijau)
        singlePointColor=(255, 0, 0), # Warna titik tanpa kecocokan (biru)
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # Jangan gambar titik tanpa match
    )

    # Menyimpan visualisasi kecocokan
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_matches_visualisasi.jpg"), img_matches)
    print(f"  [OK] Visualisasi 50 kecocokan terbaik disimpan.")

    # Menggambar semua kecocokan (untuk perbandingan)
    img_all_matches = cv2.drawMatches(
        img_left, kp_left,
        img_right, kp_right,
        good_matches_sorted,         # Semua kecocokan baik
        None,
        matchColor=(0, 200, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Menyimpan visualisasi semua kecocokan
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_matches_semua.jpg"), img_all_matches)
    print(f"  [OK] Visualisasi semua kecocokan ({len(good_matches_sorted)}) disimpan.")

    # ============================================================
    # LANGKAH 6: Estimasi Matriks Homography dengan RANSAC
    # ============================================================
    print("\n[LANGKAH 6] Mengestimasi matriks homography (RANSAC)...")

    # Memastikan ada cukup kecocokan (minimal 4 untuk homography)
    MIN_MATCH_COUNT = 10
    if len(good_matches) < MIN_MATCH_COUNT:
        print(f"  [ERROR] Tidak cukup kecocokan! ({len(good_matches)} < {MIN_MATCH_COUNT})")
        exit()

    # Mengekstrak koordinat titik dari kecocokan yang baik
    # src_pts = titik-titik di gambar kiri (source)
    src_pts = np.float32([kp_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # dst_pts = titik-titik di gambar kanan (destination / referensi)
    dst_pts = np.float32([kp_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Mengestimasi homography menggunakan RANSAC
    # H = matriks 3x3 yang memetakan titik dari gambar kiri ke gambar kanan
    # mask = mask inlier (1 = inlier, 0 = outlier)
    # ransacReprojThreshold = 5.0 piksel (toleransi error reprojeksi)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

    # Menghitung jumlah inlier (kecocokan yang konsisten dengan homography)
    inlier_count = mask.ravel().sum()
    total_matches = len(good_matches)

    # Menampilkan matriks homography dan statistik
    print(f"\n  Matriks Homography (3x3):")
    print(f"  {'-' * 45}")
    for i in range(3):
        print(f"  | {H[i, 0]:12.6f}  {H[i, 1]:12.6f}  {H[i, 2]:12.6f} |")
    print(f"  {'-' * 45}")
    print(f"\n  Jumlah inlier  : {inlier_count} dari {total_matches} kecocokan")
    print(f"  Rasio inlier   : {inlier_count/total_matches*100:.1f}%")

    # ============================================================
    # LANGKAH 7: Menentukan Ukuran Canvas Output
    # ============================================================
    print("\n[LANGKAH 7] Menentukan ukuran canvas output...")

    # Mendapatkan dimensi kedua gambar
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    # Mendefinisikan 4 sudut gambar kiri dalam koordinat homogen
    corners_left = np.float32([
        [0, 0],              # Sudut kiri atas
        [w_left, 0],         # Sudut kanan atas
        [w_left, h_left],    # Sudut kanan bawah
        [0, h_left]          # Sudut kiri bawah
    ]).reshape(-1, 1, 2)

    # Mentransformasi sudut gambar kiri ke koordinat gambar kanan menggunakan H
    # perspectiveTransform mengaplikasikan matriks homography pada titik-titik
    corners_left_transformed = cv2.perspectiveTransform(corners_left, H)

    # Mendefinisikan 4 sudut gambar kanan
    corners_right = np.float32([
        [0, 0],
        [w_right, 0],
        [w_right, h_right],
        [0, h_right]
    ]).reshape(-1, 1, 2)

    # Menggabungkan semua sudut untuk menentukan batas canvas
    all_corners = np.concatenate([corners_left_transformed, corners_right], axis=0)

    # Mencari koordinat minimum dan maksimum untuk menentukan ukuran canvas
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # Menambahkan margin kecil untuk keamanan
    x_min = min(x_min, 0)
    y_min = min(y_min, 0)

    # Menghitung ukuran canvas
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    # Menampilkan info canvas
    print(f"  Batas koordinat: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    print(f"  Ukuran canvas  : {canvas_width} x {canvas_height} piksel")

    # Membuat matriks translasi untuk menggeser gambar ke area positif
    # Karena sudut yang ditransformasi bisa memiliki koordinat negatif
    translation_matrix = np.array([
        [1, 0, -x_min],   # Translasi horizontal
        [0, 1, -y_min],   # Translasi vertikal
        [0, 0, 1]         # Baris homogen
    ], dtype=np.float64)

    # ============================================================
    # LANGKAH 8: Warping Gambar Kiri ke Canvas
    # ============================================================
    print("\n[LANGKAH 8] Melakukan warping gambar kiri ke canvas...")

    # Menghitung matriks homography gabungan (translasi + homography asli)
    # H_final = translasi @ H_original
    H_final = translation_matrix @ H

    # Melakukan perspective warping pada gambar kiri
    # warpPerspective menerapkan transformasi perspektif pada seluruh gambar
    warped_left = cv2.warpPerspective(
        img_left,                          # Gambar sumber
        H_final,                           # Matriks transformasi
        (canvas_width, canvas_height)      # Ukuran output (width, height)
    )

    # Menampilkan status warping
    print(f"  Warping gambar kiri selesai: {warped_left.shape}")

    # Menyimpan hasil warping gambar kiri saja
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_warped_kiri_saja.jpg"), warped_left)
    print("  [OK] Hasil warping gambar kiri disimpan.")

    # ============================================================
    # LANGKAH 9: Menempatkan Gambar Kanan pada Canvas
    # ============================================================
    print("\n[LANGKAH 9] Menempatkan gambar kanan pada canvas...")

    # Membuat salinan canvas untuk hasil akhir
    canvas = warped_left.copy()

    # Menghitung posisi offset untuk gambar kanan
    offset_x = -x_min  # Offset horizontal karena translasi
    offset_y = -y_min  # Offset vertikal karena translasi

    # Menempatkan gambar kanan pada canvas
    # Area yang ditempati gambar kanan dimulai dari offset
    y_start = offset_y
    y_end = offset_y + h_right
    x_start = offset_x
    x_end = offset_x + w_right

    # Memastikan koordinat tidak melebihi batas canvas
    y_end = min(y_end, canvas_height)
    x_end = min(x_end, canvas_width)

    # Menyalin piksel gambar kanan ke canvas (overwrite)
    # Ini adalah metode "no blending" -- gambar kanan langsung ditempatkan di atas
    canvas[y_start:y_end, x_start:x_end] = img_right[:y_end - y_start, :x_end - x_start]

    # Menampilkan informasi penempatan
    print(f"  Gambar kanan ditempatkan di: ({x_start},{y_start}) sampai ({x_end},{y_end})")

    # ============================================================
    # LANGKAH 10: Menyimpan Hasil Stitching Mentah (Tanpa Blending)
    # ============================================================
    print("\n[LANGKAH 10] Menyimpan hasil stitching tanpa blending...")

    # Menyimpan hasil stitching mentah
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_stitching_raw.jpg"), canvas)
    print("  [OK] Hasil stitching mentah disimpan.")

    # ============================================================
    # LANGKAH 11: Mengidentifikasi Area Overlap
    # ============================================================
    print("\n[LANGKAH 11] Mengidentifikasi area overlap...")

    # Membuat mask untuk gambar kiri yang sudah di-warp
    # Piksel non-hitam pada warped image = area yang terisi
    mask_left = cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY) > 0

    # Membuat mask untuk gambar kanan pada canvas
    mask_right = np.zeros((canvas_height, canvas_width), dtype=bool)
    mask_right[y_start:y_end, x_start:x_end] = True

    # Area overlap = irisan kedua mask
    overlap_mask = mask_left & mask_right

    # Menghitung jumlah piksel overlap
    overlap_pixels = np.sum(overlap_mask)
    total_left_pixels = np.sum(mask_left)
    total_right_pixels = np.sum(mask_right)

    # Menampilkan informasi overlap
    print(f"  Piksel gambar kiri (warped) : {total_left_pixels}")
    print(f"  Piksel gambar kanan         : {total_right_pixels}")
    print(f"  Piksel overlap              : {overlap_pixels}")
    if total_left_pixels > 0:
        print(f"  Rasio overlap terhadap kiri : {overlap_pixels/total_left_pixels*100:.1f}%")

    # Membuat visualisasi area overlap (highlight merah)
    overlap_vis = canvas.copy()
    overlap_vis[overlap_mask] = (
        overlap_vis[overlap_mask] * 0.5 + np.array([0, 0, 200]) * 0.5
    ).astype(np.uint8)

    # Menyimpan visualisasi overlap
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_overlap_area.jpg"), overlap_vis)
    print("  [OK] Visualisasi area overlap disimpan.")

    # ============================================================
    # LANGKAH 12: Percobaan dengan Berbagai Threshold Ratio Test
    # ============================================================
    print("\n[LANGKAH 12] Membandingkan threshold ratio test...")

    # Daftar threshold ratio yang akan diuji
    ratio_thresholds = [0.6, 0.75, 0.9]

    # Dictionary untuk menyimpan hasil setiap threshold
    ratio_results = {}

    for ratio in ratio_thresholds:
        # Memfilter kecocokan dengan threshold saat ini
        filtered = []
        for m, n in matches_knn:
            if m.distance < ratio * n.distance:
                filtered.append(m)

        # Menghitung jumlah kecocokan yang lolos
        jumlah = len(filtered)

        # Jika cukup kecocokan, hitung homography dan inlier
        inlier = 0
        if jumlah >= MIN_MATCH_COUNT:
            # Mengekstrak titik-titik dari kecocokan yang lolos
            src = np.float32([kp_left[m.queryIdx].pt for m in filtered]).reshape(-1, 1, 2)
            dst = np.float32([kp_right[m.trainIdx].pt for m in filtered]).reshape(-1, 1, 2)

            # Mengestimasi homography
            H_test, mask_test = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

            # Menghitung inlier
            if mask_test is not None:
                inlier = mask_test.ravel().sum()

        # Menyimpan hasil
        ratio_results[ratio] = {
            'matches': jumlah,
            'inliers': inlier,
            'ratio': inlier / jumlah * 100 if jumlah > 0 else 0
        }

        # Menampilkan hasil
        print(f"  Ratio={ratio:.2f}: {jumlah:4d} matches, {inlier:4d} inliers ({ratio_results[ratio]['ratio']:.1f}%)")

        # Menggambar visualisasi kecocokan untuk setiap threshold
        sorted_filtered = sorted(filtered, key=lambda x: x.distance)
        img_ratio_match = cv2.drawMatches(
            img_left, kp_left,
            img_right, kp_right,
            sorted_filtered[:30],
            None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"01_matches_ratio_{ratio:.2f}.jpg"),
            img_ratio_match
        )

    print("  [OK] Visualisasi masing-masing ratio test disimpan.")

    # ============================================================
    # LANGKAH 13: Membuat Grid Perbandingan dengan Matplotlib
    # ============================================================
    print("\n[LANGKAH 13] Membuat grid perbandingan matplotlib...")

    # Membuat figure dengan 2 baris dan 3 kolom
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Baris 1: Gambar kiri, gambar kanan, visualisasi matches
    # Subplot (0,0): Gambar kiri dengan keypoints
    axes[0, 0].imshow(cv2.cvtColor(img_kp_left, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Gambar Kiri ({len(kp_left)} keypoints)", fontsize=11)
    axes[0, 0].axis("off")

    # Subplot (0,1): Gambar kanan dengan keypoints
    axes[0, 1].imshow(cv2.cvtColor(img_kp_right, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Gambar Kanan ({len(kp_right)} keypoints)", fontsize=11)
    axes[0, 1].axis("off")

    # Subplot (0,2): Visualisasi matches
    axes[0, 2].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"Feature Matches ({len(good_matches)} matches)", fontsize=11)
    axes[0, 2].axis("off")

    # Baris 2: Warped kiri, hasil stitching, area overlap
    # Subplot (1,0): Gambar kiri yang sudah di-warp
    axes[1, 0].imshow(cv2.cvtColor(warped_left, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Warped Gambar Kiri", fontsize=11)
    axes[1, 0].axis("off")

    # Subplot (1,1): Hasil stitching final
    axes[1, 1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Hasil Stitching (No Blending)", fontsize=11)
    axes[1, 1].axis("off")

    # Subplot (1,2): Area overlap
    axes[1, 2].imshow(cv2.cvtColor(overlap_vis, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"Area Overlap ({overlap_pixels} piksel)", fontsize=11)
    axes[1, 2].axis("off")

    # Menambahkan judul utama figure
    plt.suptitle("Percobaan 1: Pipeline Stitching Manual\n"
                 f"(SIFT + FLANN + RANSAC Homography)",
                 fontsize=14, fontweight="bold")

    # Mengatur layout agar tidak tumpang tindih
    plt.tight_layout()

    # Menyimpan figure ke folder output
    plt.savefig(os.path.join(OUTPUT_DIR, "01_grid_pipeline_manual.png"), dpi=150, bbox_inches="tight")
    print("  [OK] Grid perbandingan pipeline disimpan.")

    # Menutup figure untuk membebaskan memori
    plt.close()

    # --- Grid kedua: Perbandingan ratio test ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ratio in enumerate(ratio_thresholds):
        # Membaca gambar hasil matches untuk ratio ini
        ratio_img_path = os.path.join(OUTPUT_DIR, f"01_matches_ratio_{ratio:.2f}.jpg")
        ratio_img = cv2.imread(ratio_img_path)
        if ratio_img is not None:
            axes2[idx].imshow(cv2.cvtColor(ratio_img, cv2.COLOR_BGR2RGB))

        # Menambahkan label dengan informasi statistik
        info = ratio_results[ratio]
        axes2[idx].set_title(
            f"Ratio = {ratio:.2f}\n"
            f"Matches: {info['matches']}, Inliers: {info['inliers']} ({info['ratio']:.1f}%)",
            fontsize=10
        )
        axes2[idx].axis("off")

    # Menambahkan judul untuk grid ratio test
    plt.suptitle("Perbandingan Lowe's Ratio Test Threshold",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Menyimpan grid perbandingan ratio test
    plt.savefig(os.path.join(OUTPUT_DIR, "01_grid_ratio_test.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Grid perbandingan ratio test disimpan.")
    plt.close()

    # ============================================================
    # LANGKAH 14: Ringkasan dan Tabel Statistik
    # ============================================================
    print("\n" + "=" * 65)
    print("RINGKASAN PERCOBAAN 1: MANUAL STITCHING PIPELINE")
    print("=" * 65)

    # Menampilkan tabel perbandingan ratio test
    print("\n  Tabel Perbandingan Ratio Test:")
    print(f"  {'Ratio':>8} | {'Matches':>8} | {'Inliers':>8} | {'Rasio %':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for ratio in ratio_thresholds:
        info = ratio_results[ratio]
        print(f"  {ratio:>8.2f} | {info['matches']:>8d} | {info['inliers']:>8d} | {info['ratio']:>7.1f}%")

    # Menampilkan pipeline yang telah dilakukan
    print("\n  Pipeline yang dibangun:")
    print("  1. Memuat gambar pasangan (pair_left.jpg, pair_right.jpg)")
    print("  2. Deteksi fitur SIFT pada kedua gambar")
    print("  3. Pencocokan fitur menggunakan FLANN + k-NN (k=2)")
    print("  4. Lowe's ratio test untuk memfilter kecocokan baik")
    print("  5. Estimasi homography 3x3 menggunakan RANSAC")
    print("  6. Menentukan ukuran canvas output")
    print("  7. Warping perspektif gambar kiri ke koordinat gambar kanan")
    print("  8. Menempatkan gambar kanan pada canvas (no blending)")
    print("  9. Identifikasi area overlap")

    # Menampilkan daftar output yang dihasilkan
    print("\n  File output yang dihasilkan:")
    output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("01_")])
    for f in output_files:
        filepath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"    - {f} ({size_kb:.1f} KB)")

    print("\n  Fungsi utama yang dipelajari:")
    print("    cv2.SIFT_create()        → Detektor fitur SIFT")
    print("    cv2.FlannBasedMatcher()   → Matcher berbasis FLANN")
    print("    cv2.findHomography()      → Estimasi homography + RANSAC")
    print("    cv2.warpPerspective()     → Warping perspektif gambar")
    print("    cv2.drawMatches()         → Visualisasi kecocokan fitur")
    print("    cv2.perspectiveTransform()→ Transformasi titik via homography")
    print("=" * 65)



if __name__ == "__main__":
    main()
