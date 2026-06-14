

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 14: GRAPH CUT SEGMENTATION
    ==========================================================================
    Graph Cut menggunakan MRF (Markov Random Field) untuk segmentasi gambar
    dengan meminimalkan fungsi energi. OpenCV menyediakan GrabCut yang
    mengimplementasikan graph cut untuk segmentasi foreground/background.

    Fungsi utama:
    - cv2.grabCut()          : segmentasi foreground/bg menggunakan graph cut
    - cv2.GC_INIT_WITH_RECT  : inisialisasi dengan bounding rectangle
    - cv2.GC_INIT_WITH_MASK  : inisialisasi dengan mask manual
    - cv2.watershed()        : segmentasi berbasis watershed (marker-based)
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
    print("PERCOBAAN 14: GRAPH CUT SEGMENTATION")
    print("=" * 60)

    np.random.seed(42)

    # ============================================================
    # 1. Memuat gambar
    # ============================================================
    print("\n--- 1. Memuat Gambar ---")

    img_path = os.path.join(IMAGE_DIR, "koin.png")
    if not os.path.exists(img_path):
        print("[ERROR] img tidak ditemukan. Jalankan download_image.py!"); exit()

    img = cv2.imread(img_path)
    print(f"  Gambar: {img.shape}")

    # ============================================================
    # 2. GrabCut dengan Rectangle
    # ============================================================
    print("\n--- 2. GrabCut dengan Rectangle ---")

    # Siapkan mask dan model
    # mask: 0=BG certain, 1=FG certain, 2=BG probable, 3=FG probable
    mask = np.zeros(img.shape[:2], np.uint8)

    # bgdModel dan fgdModel: array internal untuk GMM (Gaussian Mixture Model)
    # Harus berukuran (1, 65) float64
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Rectangle (x, y, w, h) yang mengelilingi objek foreground
    h_img, w_img = img.shape[:2]
    rect = (int(w_img*0.1), int(h_img*0.1), int(w_img*0.8), int(h_img*0.8))
    print(f"  Rectangle: {rect}")

    # cv2.grabCut iterasi graph cut optimization
    # GC_INIT_WITH_RECT: inisialisasi dari rectangle
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Konversi mask: 0,2 → background (0), 1,3 → foreground (1)
    mask_fg = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Terapkan mask ke gambar
    img_grabcut_rect = img * mask_fg[:, :, np.newaxis]

    cv2.imwrite(os.path.join(OUTPUT_DIR, "14_grabcut_rect.png"), img_grabcut_rect)
    print(f"  Piksel foreground: {np.sum(mask_fg)}")

    # ============================================================
    # 3. GrabCut dengan Mask (iterasi lanjutan)
    # ============================================================
    print("\n--- 3. GrabCut dengan Mask ---")

    # Buat mask manual (mark yang pasti foreground dan background)
    mask_manual = np.zeros(img.shape[:2], np.uint8)
    mask_manual[:] = cv2.GC_PR_BGD  # probable background

    # Mark center sebagai probable foreground
    cy, cx = h_img // 2, w_img // 2
    cv2.circle(mask_manual, (cx, cy), min(cx, cy) // 2, int(cv2.GC_PR_FGD), -1)
    # Mark center yang lebih kecil sebagai certain foreground
    cv2.circle(mask_manual, (cx, cy), min(cx, cy) // 4, int(cv2.GC_FGD), -1)
    # Mark tepi sebagai certain background
    mask_manual[:20, :] = cv2.GC_BGD
    mask_manual[-20:, :] = cv2.GC_BGD
    mask_manual[:, :20] = cv2.GC_BGD
    mask_manual[:, -20:] = cv2.GC_BGD

    bgdModel2 = np.zeros((1, 65), np.float64)
    fgdModel2 = np.zeros((1, 65), np.float64)

    # GC_INIT_WITH_MASK: inisialisasi dari mask
    cv2.grabCut(img, mask_manual, None, bgdModel2, fgdModel2, 5, cv2.GC_INIT_WITH_MASK)

    mask_fg2 = np.where((mask_manual == 2) | (mask_manual == 0), 0, 1).astype('uint8')
    img_grabcut_mask = img * mask_fg2[:, :, np.newaxis]

    cv2.imwrite(os.path.join(OUTPUT_DIR, "14_grabcut_mask.png"), img_grabcut_mask)
    print(f"  Piksel foreground: {np.sum(mask_fg2)}")

    # ============================================================
    # 4. Iterasi GrabCut dan konvergensi
    # ============================================================
    print("\n--- 4. Iterasi GrabCut ---")

    results_iter = []
    fg_counts = []

    for n_iter in [1, 3, 5, 10, 20]:
        mask_i = np.zeros(img.shape[:2], np.uint8)
        bgd_i = np.zeros((1, 65), np.float64)
        fgd_i = np.zeros((1, 65), np.float64)

        cv2.grabCut(img, mask_i, rect, bgd_i, fgd_i, n_iter, cv2.GC_INIT_WITH_RECT)

        mask_fg_i = np.where((mask_i == 2) | (mask_i == 0), 0, 1).astype('uint8')
        fg_count = np.sum(mask_fg_i)
        fg_counts.append(fg_count)
        results_iter.append(img * mask_fg_i[:, :, np.newaxis])

        print(f"  Iterasi {n_iter:2d}: fg_pixels={fg_count}")

    # ============================================================
    # 5. Watershed Segmentation
    # ============================================================
    print("\n--- 5. Watershed Segmentation ---")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal dengan morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background: dilasi dari opening
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground: erosi agresif atau distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # background = 1, bukan 0
    markers[unknown == 255] = 0  # unknown = 0

    # cv2.watershed: segmentasi berbasis marker
    # Boundary ditandai dengan -1
    markers_ws = cv2.watershed(img, markers.copy())

    # Tandai boundary
    img_watershed = img.copy()
    img_watershed[markers_ws == -1] = [0, 0, 255]  # merah untuk boundary

    cv2.imwrite(os.path.join(OUTPUT_DIR, "14_watershed.png"), img_watershed)
    print(f"  Jumlah region: {markers_ws.max()}")

    # ============================================================
    # 6. Energy minimization sederhana (MRF konsep)
    # ============================================================
    print("\n--- 6. MRF Energy Minimization (ICM) ---")

    # Buat gambar noisy sederhana untuk segmentasi biner
    h_small, w_small = 100, 100
    clean = np.zeros((h_small, w_small), dtype=np.float64)
    clean[30:70, 30:70] = 1.0  # kotak putih
    noisy = clean + np.random.normal(0, 0.5, clean.shape)

    # ICM (Iterated Conditional Modes)
    # Energi: E = sum(data_term) + lambda * sum(smoothness_term)
    # data_term: (pixel - label)^2
    # smoothness_term: penalti jika tetangga berbeda label

    labels = (noisy > 0.5).astype(np.float64)
    lambda_smooth = 2.0

    for iteration in range(20):
        new_labels = labels.copy()
        changed = 0

        for y in range(1, h_small - 1):
            for x in range(1, w_small - 1):
                # Hitung energi untuk label 0 dan 1
                neighbors = [labels[y-1, x], labels[y+1, x],
                            labels[y, x-1], labels[y, x+1]]

                # Energi label=0
                e0 = (noisy[y, x] - 0) ** 2 + lambda_smooth * sum(n != 0 for n in neighbors)
                # Energi label=1
                e1 = (noisy[y, x] - 1) ** 2 + lambda_smooth * sum(n != 1 for n in neighbors)

                best = 0 if e0 < e1 else 1
                if best != labels[y, x]:
                    changed += 1
                new_labels[y, x] = best

        labels = new_labels
        if changed == 0:
            print(f"  Konvergen pada iterasi {iteration + 1}")
            break
        if iteration % 5 == 0:
            print(f"  Iterasi {iteration}: {changed} piksel berubah")

    # ============================================================
    # 7. Visualisasi gabungan
    # ============================================================
    print("\n--- 7. Visualisasi Gabungan ---")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(cv2.cvtColor(img_grabcut_rect, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("GrabCut (Rect)")

    axes[0, 2].imshow(cv2.cvtColor(img_grabcut_mask, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("GrabCut (Mask)")

    axes[0, 3].imshow(cv2.cvtColor(img_watershed, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title("Watershed")

    axes[1, 0].imshow(noisy, cmap='gray')
    axes[1, 0].set_title("Noisy Image")

    axes[1, 1].imshow(clean, cmap='gray')
    axes[1, 1].set_title("Clean (Ground Truth)")

    axes[1, 2].imshow(labels, cmap='gray')
    axes[1, 2].set_title("ICM Result")

    # Perbandingan iterasi GrabCut
    axes[1, 3].plot([1, 3, 5, 10, 20], fg_counts, 'bo-')
    axes[1, 3].set_xlabel("Iterasi")
    axes[1, 3].set_ylabel("Piksel Foreground")
    axes[1, 3].set_title("GrabCut Convergence")

    for ax in axes.flat[:7]:
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "14_graph_cut_all.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Disimpan: {output_path}")

    print("\n" + "=" * 60)
    print("PERCOBAAN 14 SELESAI")
    print("=" * 60)



if __name__ == "__main__":
    main()
