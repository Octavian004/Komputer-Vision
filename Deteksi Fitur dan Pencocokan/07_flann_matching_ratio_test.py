"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
File    : 07_flann_matching_ratio_test.py
Topik   : FLANN-based Matching
Deskripsi:
    Mempelajari FlannBasedMatcher sebagai alternatif cepat dari BFMatcher,
    termasuk konfigurasi untuk SIFT (KDTree) dan ORB (LSH), perbandingan
    kecepatan, serta visualisasi inlier/outlier menggunakan RANSAC.
=============================================================================
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Path konfigurasi ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def muat_gambar_pasangan():
    """
    Memuat atau membuat sepasang gambar sintetis.
    Gambar pertama: adegan solid dengan bentuk geometris.
    Gambar kedua  : rotasi 25° dan scale 0.85 dari gambar pertama.
    Mengembalikan : (gray1, gray2, img1_bgr, img2_bgr)
    """
    for nama in ["chess.png", "box.png", "simple.png", "scene.png"]:
        jalur = os.path.join(IMAGE_DIR, nama)
        if os.path.exists(jalur):
            img = cv2.imread(jalur)
            if img is not None:
                h, w  = img.shape[:2]
                M     = cv2.getRotationMatrix2D((w//2, h//2), 30, 0.9)
                img2  = cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT)
                return (cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                        img, img2)

    H, W = 480, 640
    img1 = np.full((H, W, 3), (60, 30, 10), dtype=np.uint8)
    cv2.rectangle(img1, (80, 60),   (250, 200), (200, 100,  50), -1)
    cv2.circle   (img1, (420, 150),  90,        (50,  180, 220), -1)
    pts = np.array([[300, 350], [500, 420], [200, 430]], np.int32)
    cv2.fillPoly (img1, [pts],               (80,  220, 100))
    cv2.rectangle(img1, (50, 300),   (180, 430),(180,  60, 200), -1)
    cv2.circle   (img1, (560, 360),   60,       (230, 200,  40), -1)
    noise = np.random.randint(0, 30, img1.shape, dtype=np.uint8)
    img1  = cv2.add(img1, noise)

    M    = cv2.getRotationMatrix2D((W//2, H//2), 25, 0.85)
    img2 = cv2.warpAffine(img1, M, (W, H), borderMode=cv2.BORDER_REFLECT)

    return (cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
            img1, img2)


# ─────────────────────────────────────────────────────────────────────────────
def demo_flann_sift_kdtree(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 1: FlannBasedMatcher dengan SIFT menggunakan indeks KDTree.
    Parameter: trees=5, checks=50. Ratio test Lowe (0.7) diterapkan.
    """
    print("\n[Demo 1] FLANN + SIFT (KDTree index, trees=5, checks=50) + ratio test")

    # Detektor SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"  Keypoint SIFT — gambar-1: {len(kp1)}, gambar-2: {len(kp2)}")

    if des1 is None or des2 is None:
        print("  [SKIP] Deskriptor kosong.")
        return

    # Konfigurasi FLANN untuk deskriptor float (SIFT/SURF) → gunakan KDTree
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # lebih banyak checks → lebih akurat
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    pasang = flann.knnMatch(des1, des2, k=2)

    # Ratio test Lowe: simpan hanya match yang jelas terbaik
    RATIO = 0.7
    match_baik = [[m] for m, n in pasang
                  if len([m, n]) == 2 and m.distance < RATIO * n.distance]
    print(f"  Match setelah ratio test ({RATIO}): {len(match_baik)}")

    gb = cv2.drawMatchesKnn(
        img1_bgr, kp1, img2_bgr, kp2, match_baik[:40], None,
        matchColor=(0, 200, 100), singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(cv2.cvtColor(gb, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"FLANN + SIFT (KDTree, trees=5, checks=50) + ratio test (ratio={RATIO})\n"
        f"Keypoint: {len(kp1)} | {len(kp2)} — Match survive: {len(match_baik)}",
        fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_demo1_flann_sift_kdtree.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_flann_orb_lsh(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 2: FlannBasedMatcher dengan ORB menggunakan indeks LSH.
    Parameter: table_number=6, key_size=12, multi_probe_level=1.
    """
    print("\n[Demo 2] FLANN + ORB (LSH index, table=6, key=12, probe=1)")

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    print(f"  Keypoint ORB — gambar-1: {len(kp1)}, gambar-2: {len(kp2)}")

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("  [SKIP] Deskriptor tidak cukup.")
        return

    # Konfigurasi FLANN untuk deskriptor biner (ORB/BRISK) → gunakan LSH
    FLANN_INDEX_LSH = 6
    index_params  = dict(algorithm=FLANN_INDEX_LSH,
                         table_number=6,      # jumlah hash table
                         key_size=12,         # panjang kunci hash
                         multi_probe_level=1) # multi-probe untuk akurasi lebih
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Konversi ke uint8 jika belum
    des1 = np.uint8(des1)
    des2 = np.uint8(des2)

    pasang = flann.knnMatch(des1, des2, k=2)

    RATIO = 0.75
    match_baik = []
    for p in pasang:
        if len(p) == 2:
            m, n = p
            if m.distance < RATIO * n.distance:
                match_baik.append([m])

    print(f"  Match setelah ratio test ({RATIO}): {len(match_baik)}")

    gb = cv2.drawMatchesKnn(
        img1_bgr, kp1, img2_bgr, kp2, match_baik[:40], None,
        matchColor=(120, 60, 220), singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(cv2.cvtColor(gb, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"FLANN + ORB (LSH, table=6, key=12, probe=1) + ratio test (ratio={RATIO})\n"
        f"Keypoint: {len(kp1)} | {len(kp2)} — Match survive: {len(match_baik)}",
        fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_demo2_flann_orb_lsh.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_perbandingan_kecepatan(gray1, gray2):
    """
    Demo 3: Perbandingan kecepatan BFMatcher vs FLANN pada berbagai
    jumlah keypoint N = 100, 300, 500, 800.
    """
    print("\n[Demo 3] Perbandingan kecepatan BFMatcher vs FLANN")

    n_list   = [100, 300, 500, 800]
    t_bf     = []
    t_flann  = []

    for N in n_list:
        orb = cv2.ORB_create(nfeatures=N)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            t_bf.append(0); t_flann.append(0)
            continue

        des1 = np.uint8(des1)
        des2 = np.uint8(des2)

        # Ukur waktu BFMatcher
        ULANG = 5
        start = time.time()
        for _ in range(ULANG):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            bf.knnMatch(des1, des2, k=2)
        rata_bf = (time.time() - start) / ULANG * 1000   # ms
        t_bf.append(rata_bf)

        # Ukur waktu FLANN (LSH untuk ORB)
        FLANN_INDEX_LSH = 6
        index_params  = dict(algorithm=FLANN_INDEX_LSH,
                             table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        start = time.time()
        for _ in range(ULANG):
            flann.knnMatch(des1, des2, k=2)
        rata_fl = (time.time() - start) / ULANG * 1000   # ms
        t_flann.append(rata_fl)

        print(f"  N={N:4d}: BF={rata_bf:.2f}ms, FLANN={rata_fl:.2f}ms "
              f"(FLANN {rata_bf/rata_fl:.1f}x {'lebih cepat' if rata_fl<rata_bf else 'lebih lambat'})")

    # Plot perbandingan waktu
    x     = np.arange(len(n_list))
    lebar = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bar1 = ax.bar(x - lebar/2, t_bf,    lebar, label="BFMatcher",  color="#e74c3c")
    bar2 = ax.bar(x + lebar/2, t_flann, lebar, label="FLANN",      color="#3498db")

    ax.set_xlabel("Jumlah Keypoint (N)", fontsize=11)
    ax.set_ylabel("Waktu Rata-rata (ms)", fontsize=11)
    ax.set_title("Perbandingan Kecepatan: BFMatcher vs FLANN (ORB)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_list])
    ax.legend()
    ax.bar_label(bar1, fmt="%.1f", fontsize=8)
    ax.bar_label(bar2, fmt="%.1f", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_demo3_kecepatan_bf_vs_flann.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_inlier_outlier_ransac(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 4: Visualisasi inlier (hijau) vs outlier (merah) menggunakan
    FLANN matching + RANSAC Homography.
    """
    print("\n[Demo 4] Visualisasi inlier/outlier dengan FLANN + RANSAC")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("  [SKIP] Keypoint tidak cukup.")
        return

    # FLANN KDTree untuk SIFT
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50))
    pasang = flann.knnMatch(des1, des2, k=2)

    # Ratio test untuk mendapatkan kandidat match yang andal
    RATIO  = 0.75
    match_andal = [m for m, n in pasang
                   if len([m, n]) == 2 and m.distance < RATIO * n.distance]

    if len(match_andal) < 4:
        print(f"  [SKIP] Match terlalu sedikit: {len(match_andal)}")
        return

    # Ambil koordinat titik untuk RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in match_andal]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in match_andal]).reshape(-1, 1, 2)

    # Estimasi Homography + RANSAC → mask inlier/outlier
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel() if mask is not None else np.zeros(len(match_andal))

    inlier_match  = [m for m, flag in zip(match_andal, mask) if flag]
    outlier_match = [m for m, flag in zip(match_andal, mask) if not flag]

    persen_inlier = len(inlier_match) / len(match_andal) * 100
    print(f"  Total match andal : {len(match_andal)}")
    print(f"  Inlier RANSAC     : {len(inlier_match)} ({persen_inlier:.1f}%)")
    print(f"  Outlier RANSAC    : {len(outlier_match)} ({100-persen_inlier:.1f}%)")

    # Gambar dengan warna berbeda: hijau=inlier, merah=outlier
    H1, W1 = img1_bgr.shape[:2]
    H2, W2 = img2_bgr.shape[:2]
    kanvas  = np.zeros((max(H1, H2), W1 + W2, 3), dtype=np.uint8)
    kanvas[:H1, :W1] = img1_bgr
    kanvas[:H2, W1:] = img2_bgr

    # Gambar garis inlier (hijau)
    for m in inlier_match:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = (int(kp2[m.trainIdx].pt[0]) + W1,
                int(kp2[m.trainIdx].pt[1]))
        cv2.line(kanvas, pt1, pt2, (0, 220, 0), 1)
        cv2.circle(kanvas, pt1, 3, (0, 220, 0), -1)
        cv2.circle(kanvas, pt2, 3, (0, 220, 0), -1)

    # Gambar garis outlier (merah)
    for m in outlier_match:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = (int(kp2[m.trainIdx].pt[0]) + W1,
                int(kp2[m.trainIdx].pt[1]))
        cv2.line(kanvas, pt1, pt2, (0, 0, 220), 1)
        cv2.circle(kanvas, pt1, 3, (0, 0, 220), -1)
        cv2.circle(kanvas, pt2, 3, (0, 0, 220), -1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(cv2.cvtColor(kanvas, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"FLANN + RANSAC — Inlier: {len(inlier_match)} (hijau, "
        f"{persen_inlier:.1f}%) | "
        f"Outlier: {len(outlier_match)} (merah, {100-persen_inlier:.1f}%)",
        fontsize=11)
    ax.axis("off")

    # Legend manual
    from matplotlib.patches import Patch
    legend = [Patch(facecolor="green", label=f"Inlier ({len(inlier_match)})"),
              Patch(facecolor="red",   label=f"Outlier ({len(outlier_match)})")]
    ax.legend(handles=legend, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_demo4_inlier_outlier_ransac.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Modul 04 — 07: FLANN-based Matching & Ratio Test")
    print("=" * 60)

    gray1, gray2, img1_bgr, img2_bgr = muat_gambar_pasangan()
    print(f"Ukuran gambar-1: {gray1.shape}, gambar-2: {gray2.shape}")

    demo_flann_sift_kdtree      (gray1, gray2, img1_bgr, img2_bgr)
    demo_flann_orb_lsh          (gray1, gray2, img1_bgr, img2_bgr)
    demo_perbandingan_kecepatan (gray1, gray2)
    demo_inlier_outlier_ransac  (gray1, gray2, img1_bgr, img2_bgr)

    print("\nSemua demo selesai. Output disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
