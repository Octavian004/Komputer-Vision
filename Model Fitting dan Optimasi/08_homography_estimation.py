

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 08: HOMOGRAPHY ESTIMATION (DLT)
    ==========================================================================
    Homography adalah transformasi proyektif 3x3 yang memetakan titik dari
    satu bidang ke bidang lain. Digunakan untuk koreksi perspektif,
    image stitching, dan augmented reality.

    Persamaan: x' = H * x (koordinat homogen)
    Minimal 4 pasang titik korespondensi diperlukan.
    Cara estimasi: Direct Linear Transform (DLT)

    Fungsi utama:
    - cv2.findHomography()        : estimasi matriks homography
    - cv2.warpPerspective()       : transformasi gambar dengan H
    - cv2.perspectiveTransform()  : transformasi titik dengan H
    - cv2.getPerspectiveTransform(): H dari tepat 4 titik
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
    print("PERCOBAAN 08: HOMOGRAPHY ESTIMATION (DLT)")
    print("=" * 60)

    # ============================================================
    # 1. Homography dari 4 titik (Exact Solution)
    # ============================================================
    print("\n--- 1. Homography dari 4 Titik ---")

    # Membaca gambar papan catur
    img_path = os.path.join(IMAGE_DIR, "papan_catur.jpg")
    if not os.path.exists(img_path):
        print("[ERROR] papan_catur.jpg tidak ditemukan. Jalankan download_image.py!"); exit()

    img = cv2.imread(img_path)

    # Definisi 4 titik sumber (sudut papan catur dalam gambar)
    # Format: titik kiri-atas, kanan-atas, kanan-bawah, kiri-bawah
    pts_src = np.float32([[80, 0], [560, 0], [560, 480], [80, 480]])

    # 4 titik tujuan (perspektif miring — simulasi view dari sudut)
    pts_dst = np.float32([[120, 30], [520, 10], [580, 460], [60, 430]])

    # Menghitung homography menggunakan cv2.getPerspectiveTransform
    # Ini menghitung H secara EXACT dari 4 pasang titik
    H_exact = cv2.getPerspectiveTransform(pts_src, pts_dst)
    print(f"  Matriks Homography (exact):")
    for row in H_exact:
        print(f"    [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f}]")

    # Warp gambar menggunakan homography
    warped = cv2.warpPerspective(img, H_exact, (640, 480))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "08_homography_warp.png"), warped)
    print(f"  Gambar di-warp dan disimpan")

    # ============================================================
    # 2. Homography dari >= 4 titik + RANSAC
    # ============================================================
    print("\n--- 2. findHomography + RANSAC ---")

    # Buat banyak pasangan titik (simulasi feature matching)
    n_pairs = 30
    np.random.seed(42)

    # Generate titik sumber acak di area papan
    pts_src_many = np.random.uniform(80, 560, (n_pairs, 2)).astype(np.float32)

    # Transformasi ke tujuan menggunakan H yang benar + noise
    pts_src_h = np.hstack([pts_src_many, np.ones((n_pairs, 1))])
    pts_dst_many = (H_exact @ pts_src_h.T).T
    pts_dst_many = pts_dst_many[:, :2] / pts_dst_many[:, 2:3]

    # Tambahkan noise kecil (simulasi ketidakpastian matching)
    pts_dst_many += np.random.randn(n_pairs, 2) * 2

    # Tambahkan outlier (matching salah)
    n_outlier = 8
    outlier_idx = np.random.choice(n_pairs, n_outlier, replace=False)
    pts_dst_many[outlier_idx] = np.random.uniform(0, 640, (n_outlier, 2))

    pts_src_many = pts_src_many.astype(np.float32)
    pts_dst_many = pts_dst_many.astype(np.float32)

    # findHomography dengan RANSAC
    # Mengembalikan H dan mask inlier
    H_ransac, mask = cv2.findHomography(pts_src_many, pts_dst_many,
                                         cv2.RANSAC, ransacReprojThreshold=5.0)

    n_inliers = np.sum(mask)
    print(f"  findHomography (RANSAC): {n_inliers}/{n_pairs} inlier")
    print(f"  True outlier: {n_outlier}, Detected outlier: {n_pairs - n_inliers}")

    # ============================================================
    # 3. Koreksi Perspektif (Bird's Eye View)
    # ============================================================
    print("\n--- 3. Koreksi Perspektif ---")

    # Buat gambar dengan perspektif miring
    img_persp = img.copy()
    # Simulasi gambar yang difoto dari sudut: warp dulu
    M_tilt = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [640, 0], [640, 480], [0, 480]]),
        np.float32([[100, 50], [540, 30], [600, 450], [40, 430]])
    )
    img_tilted = cv2.warpPerspective(img, M_tilt, (640, 480))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "08_tilted.png"), img_tilted)

    # Koreksi: dari perspektif miring kembali ke tegak
    # Titik sumber = 4 sudut di gambar miring
    pts_tilted = np.float32([[100, 50], [540, 30], [600, 450], [40, 430]])
    # Titik tujuan = persegi panjang tegak
    pts_corrected = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

    H_correct = cv2.getPerspectiveTransform(pts_tilted, pts_corrected)
    img_corrected = cv2.warpPerspective(img_tilted, H_correct, (640, 480))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "08_corrected.png"), img_corrected)
    print(f"  Perspektif dikoreksi (bird's eye view)")

    # ============================================================
    # 4. Transformasi Titik dengan Homography
    # ============================================================
    print("\n--- 4. Transformasi Titik ---")

    # Titik-titik yang ingin ditransformasi
    test_points = np.float32([[[80, 0]], [[320, 240]], [[560, 480]]])

    # cv2.perspectiveTransform mentransformasi titik (bukan gambar)
    transformed = cv2.perspectiveTransform(test_points, H_exact)

    for orig, trans in zip(test_points, transformed):
        ox, oy = orig[0]
        tx, ty = trans[0]
        print(f"  ({ox:.0f}, {oy:.0f}) → ({tx:.1f}, {ty:.1f})")

    # ============================================================
    # 5. Dekomposisi Homography
    # ============================================================
    print("\n--- 5. Dekomposisi Homography ---")

    # Buat intrinsic camera matrix sederhana
    fx, fy = 500, 500
    cx_cam, cy_cam = 320, 240
    K = np.float64([[fx, 0, cx_cam], [0, fy, cy_cam], [0, 0, 1]])

    # cv2.decomposeHomographyMat mendekomposisi H menjadi R, t, n
    retval, rotations, translations, normals = cv2.decomposeHomographyMat(H_exact, K)
    print(f"  Jumlah solusi: {retval}")
    for i in range(min(retval, 2)):
        print(f"  Solusi {i+1}:")
        print(f"    Rotation angle: {np.degrees(cv2.Rodrigues(rotations[i])[0].flatten())}")
        print(f"    Translation: {translations[i].flatten()}")

    # ============================================================
    # 6. Visualisasi lengkap
    # ============================================================
    print("\n--- 6. Visualisasi ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original
    axes[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0][0].set_title("Original")

    # Warped
    axes[0][1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[0][1].set_title("Warped (H exact)")

    # Tilted
    axes[0][2].imshow(cv2.cvtColor(img_tilted, cv2.COLOR_BGR2RGB))
    axes[0][2].set_title("Tilted (Perspektif)")

    # Corrected
    axes[1][0].imshow(cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB))
    axes[1][0].set_title("Corrected (Bird's Eye)")

    # Feature matches plot
    ax_match = axes[1][1]
    inlier_m = mask.ravel() == 1
    ax_match.scatter(pts_src_many[inlier_m, 0], pts_src_many[inlier_m, 1],
                     c='green', s=30, label=f'Inlier ({np.sum(inlier_m)})')
    ax_match.scatter(pts_src_many[~inlier_m, 0], pts_src_many[~inlier_m, 1],
                     c='red', s=30, label=f'Outlier ({np.sum(~inlier_m)})')
    ax_match.set_title("RANSAC Inlier/Outlier")
    ax_match.legend()
    ax_match.grid(True, alpha=0.3)

    # Reprojection error
    ax_err = axes[1][2]
    pts_src_h2 = np.hstack([pts_src_many, np.ones((n_pairs, 1))])
    pts_reproj = (H_ransac @ pts_src_h2.T).T
    pts_reproj = pts_reproj[:, :2] / pts_reproj[:, 2:3]
    errors = np.sqrt(np.sum((pts_reproj - pts_dst_many) ** 2, axis=1))
    ax_err.bar(range(n_pairs), errors, color=['green' if m else 'red' for m in mask.ravel()])
    ax_err.axhline(5.0, color='orange', linestyle='--', label='RANSAC threshold')
    ax_err.set_title("Reprojection Error")
    ax_err.set_xlabel("Pair Index")
    ax_err.set_ylabel("Error (pixel)")
    ax_err.legend()

    for ax in axes.flat:
        ax.axis('off') if not ax.get_xlabel() else None

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "08_homography_full.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Disimpan: {output_path}")

    # ============================================================
    # 7. Metode estimasi: RANSAC vs LMEDS vs RHO
    # ============================================================
    print("\n--- 7. Perbandingan Metode Estimasi ---")

    methods = {
        'RANSAC': cv2.RANSAC,
        'LMEDS': cv2.LMEDS,
        'RHO': cv2.RHO,
    }

    for name, method in methods.items():
        H_m, mask_m = cv2.findHomography(pts_src_many, pts_dst_many, method, 5.0)
        n_in = np.sum(mask_m) if mask_m is not None else 0

        # Hitung reprojection error
        if H_m is not None:
            pts_h = np.hstack([pts_src_many, np.ones((n_pairs, 1))])
            pts_r = (H_m @ pts_h.T).T
            pts_r = pts_r[:, :2] / pts_r[:, 2:3]
            mean_err = np.mean(np.sqrt(np.sum((pts_r - pts_dst_many)**2, axis=1)))
        else:
            mean_err = float('inf')

        print(f"  {name:8s}: inlier={n_in:2d}, mean_reproj_err={mean_err:.2f}")

    print("\n" + "=" * 60)
    print("PERCOBAAN 08 SELESAI")
    print("=" * 60)



if __name__ == "__main__":
    main()
