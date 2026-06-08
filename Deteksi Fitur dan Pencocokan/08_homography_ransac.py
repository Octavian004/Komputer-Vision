"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
File    : 08_homography_ransac.py
Topik   : Homography dan RANSAC
Deskripsi:
    Mempelajari estimasi Homography menggunakan RANSAC, transformasi perspektif
    dengan warpPerspective, visualisasi inlier RANSAC, dan pengaruh parameter
    ransacReprojThreshold terhadap jumlah inlier serta presisi.
=============================================================================
"""

import os
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
    Gambar pertama: scene dengan bentuk geometris.
    Gambar kedua  : transformasi perspektif dari gambar pertama.
    Mengembalikan : (gray1, gray2, img1_bgr, img2_bgr)
    """
    for nama in ["chess.png", "box.png", "simple.png", "scene.png"]:
        jalur = os.path.join(IMAGE_DIR, nama)
        if os.path.exists(jalur):
            img = cv2.imread(jalur)
            if img is not None:
                h, w  = img.shape[:2]
                M     = cv2.getRotationMatrix2D((w//2, h//2), 20, 0.9)
                img2  = cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT)
                return (cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                        img, img2)

    # Gambar sintetis dengan noise tekstur sehingga SIFT bekerja baik
    H, W = 480, 640
    img1 = np.full((H, W, 3), (40, 20, 80), dtype=np.uint8)
    cv2.rectangle(img1, (50,  40),  (280, 220), (180, 100,  60), -1)
    cv2.circle   (img1, (450, 140),  100,       (60,  200, 180), -1)
    pts = np.array([[150, 350], [480, 400], [100, 430]], np.int32)
    cv2.fillPoly (img1, [pts],               (220, 80,  80))
    cv2.rectangle(img1, (350, 280),  (600, 440),(60,  120, 220), -1)
    noise = np.random.randint(0, 25, img1.shape, dtype=np.uint8)
    img1  = cv2.add(img1, noise)

    # Transformasi perspektif ringan untuk img2
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[20, 30], [W-10, 15], [W-30, H-20], [10, H-10]])
    M   = cv2.getPerspectiveTransform(src, dst)
    img2 = cv2.warpPerspective(img1, M, (W, H), borderMode=cv2.BORDER_REFLECT)

    return (cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
            img1, img2)


# ─────────────────────────────────────────────────────────────────────────────
def demo_homography_dasar(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 1: Estimasi Homography dasar dengan cv2.findHomography + RANSAC.
    Menampilkan matriks H dan statistik inlier/outlier.
    """
    print("\n[Demo 1] cv2.findHomography + RANSAC (reprojThreshold=5.0) — dasar")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("  [SKIP] Keypoint tidak cukup.")
        return

    # FLANN untuk mendapatkan kandidat match
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))
    pasang = flann.knnMatch(des1, des2, k=2)
    match_andal = [m for m, n in pasang
                   if len([m, n]) == 2 and m.distance < 0.75 * n.distance]

    if len(match_andal) < 4:
        print(f"  [SKIP] Match terlalu sedikit: {len(match_andal)}")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in match_andal]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in match_andal]).reshape(-1, 1, 2)

    # Estimasi Homography menggunakan RANSAC
    H_mat, mask = cv2.findHomography(src_pts, dst_pts,
                                     cv2.RANSAC, ransacReprojThreshold=5.0)

    if H_mat is None:
        print("  [SKIP] Homography tidak dapat diestimasi.")
        return

    mask       = mask.ravel()
    n_inlier   = int(mask.sum())
    n_outlier  = int((1 - mask).sum())
    pct_inlier = n_inlier / len(mask) * 100

    print(f"  Total match    : {len(match_andal)}")
    print(f"  Inlier RANSAC  : {n_inlier} ({pct_inlier:.1f}%)")
    print(f"  Outlier RANSAC : {n_outlier}")
    print(f"  Matriks Homography H:\n{H_mat}")

    # Visualisasi: gambar asli + kanan gambar transformed
    H_h, W_h = gray1.shape[:2]
    img1_warped = cv2.warpPerspective(img1_bgr, H_mat, (W_h, H_h))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img1_bgr,    cv2.COLOR_BGR2RGB))
    axes[0].set_title("Gambar Asal (img1)", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(img2_bgr,    cv2.COLOR_BGR2RGB))
    axes[1].set_title("Gambar Target (img2)", fontsize=11)
    axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(img1_warped, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f"img1 → warpPerspective(H)\n"
        f"Inlier: {n_inlier}/{len(mask)} ({pct_inlier:.1f}%)",
        fontsize=11)
    axes[2].axis("off")

    plt.suptitle("Homography Dasar — RANSAC (reprojThreshold=5.0)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_demo1_homography_dasar.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_warp_perspective(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 2: Penggunaan cv2.warpPerspective untuk transformasi gambar
    menggunakan Homography yang sudah diestimasi.
    """
    print("\n[Demo 2] cv2.warpPerspective — transformasi gambar dengan Homography")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4:
        print("  [SKIP] Keypoint tidak cukup.")
        return

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))
    pasang = flann.knnMatch(des1, des2, k=2)
    match_andal = [m for m, n in pasang
                   if len([m, n]) == 2 and m.distance < 0.75 * n.distance]

    if len(match_andal) < 4:
        print("  [SKIP] Match tidak cukup.")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in match_andal]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in match_andal]).reshape(-1, 1, 2)
    H_mat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H_mat is None:
        print("  [SKIP] Homography gagal.")
        return

    H_h, W_h   = img2_bgr.shape[:2]
    H_inv       = np.linalg.inv(H_mat)   # inverse untuk arah kebalikan

    # Warp img1 ke ruang koordinat img2
    img1_ke_img2 = cv2.warpPerspective(img1_bgr, H_mat, (W_h, H_h),
                                       borderValue=(128, 128, 128))
    # Warp img2 ke ruang koordinat img1 (inverse)
    img2_ke_img1 = cv2.warpPerspective(img2_bgr, H_inv, (W_h, H_h),
                                       borderValue=(128, 128, 128))
    # Overlay (blending) untuk membandingkan
    overlay = cv2.addWeighted(img2_bgr, 0.5, img1_ke_img2, 0.5, 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, gambar, judul in zip(
            axes.ravel(),
            [img1_bgr, img2_bgr, img1_ke_img2, overlay],
            ["img1 (asal)", "img2 (target)",
             "img1 → warpPerspective(H) → ruang img2",
             "Overlay (img2 × 0.5 + warped × 0.5)"]):
        ax.imshow(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
        ax.set_title(judul, fontsize=10)
        ax.axis("off")

    plt.suptitle("cv2.warpPerspective — Transformasi Gambar dengan Homography",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_demo2_warp_perspective.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_visualisasi_inlier_ransac(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 3: Visualisasi inlier/outlier RANSAC — hitung persentase inlier.
    Inlier ditampilkan dengan garis hijau, outlier dengan garis merah.
    """
    print("\n[Demo 3] Visualisasi inlier RANSAC — mask dari findHomography")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4:
        print("  [SKIP] Keypoint tidak cukup.")
        return

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))
    pasang = flann.knnMatch(des1, des2, k=2)
    match_andal = [m for m, n in pasang
                   if len([m, n]) == 2 and m.distance < 0.75 * n.distance]

    if len(match_andal) < 4:
        print("  [SKIP] Match tidak cukup.")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in match_andal]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in match_andal]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask    = mask.ravel() if mask is not None else np.zeros(len(match_andal))

    # Pisahkan inlier dan outlier berdasarkan mask RANSAC
    inlier_m  = [m for m, f in zip(match_andal, mask) if f]
    outlier_m = [m for m, f in zip(match_andal, mask) if not f]
    pct = len(inlier_m) / max(len(match_andal), 1) * 100

    print(f"  Kandidat match  : {len(match_andal)}")
    print(f"  Inlier RANSAC   : {len(inlier_m)} ({pct:.1f}%)")
    print(f"  Outlier RANSAC  : {len(outlier_m)} ({100-pct:.1f}%)")

    # Gambar kanvas gabungan
    H1, W1 = img1_bgr.shape[:2]
    H2, W2 = img2_bgr.shape[:2]
    kanvas = np.zeros((max(H1, H2), W1+W2, 3), dtype=np.uint8)
    kanvas[:H1, :W1] = img1_bgr
    kanvas[:H2, W1:] = img2_bgr

    # Inlier → hijau
    for m in inlier_m:
        p1 = tuple(map(int, kp1[m.queryIdx].pt))
        p2 = (int(kp2[m.trainIdx].pt[0])+W1, int(kp2[m.trainIdx].pt[1]))
        cv2.line(kanvas, p1, p2, (0, 220, 0),  1, cv2.LINE_AA)
        cv2.circle(kanvas, p1, 4, (0, 255, 0), -1)
        cv2.circle(kanvas, p2, 4, (0, 255, 0), -1)

    # Outlier → merah
    for m in outlier_m:
        p1 = tuple(map(int, kp1[m.queryIdx].pt))
        p2 = (int(kp2[m.trainIdx].pt[0])+W1, int(kp2[m.trainIdx].pt[1]))
        cv2.line(kanvas, p1, p2, (0, 0, 220),  1, cv2.LINE_AA)
        cv2.circle(kanvas, p1, 4, (0, 0, 255), -1)
        cv2.circle(kanvas, p2, 4, (0, 0, 255), -1)

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(cv2.cvtColor(kanvas, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"Visualisasi Inlier/Outlier RANSAC\n"
        f"Inlier: {len(inlier_m)} ({pct:.1f}%)  |  "
        f"Outlier: {len(outlier_m)} ({100-pct:.1f}%)",
        fontsize=12)
    ax.axis("off")
    legend = [Patch(facecolor="green", label=f"Inlier ({len(inlier_m)})"),
              Patch(facecolor="red",   label=f"Outlier ({len(outlier_m)})")]
    ax.legend(handles=legend, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_demo3_inlier_outlier_ransac.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_pengaruh_ransac_threshold(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 4: Pengaruh ransacReprojThreshold (1/3/5/10) terhadap jumlah inlier
    dan presisi matching. Ditampilkan sebagai subplot 2×2.
    """
    print("\n[Demo 4] Pengaruh ransacReprojThreshold (1/3/5/10) — subplot 2×2")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4:
        print("  [SKIP] Keypoint tidak cukup.")
        return

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))
    pasang = flann.knnMatch(des1, des2, k=2)
    match_andal = [m for m, n in pasang
                   if len([m, n]) == 2 and m.distance < 0.75 * n.distance]

    if len(match_andal) < 4:
        print("  [SKIP] Match tidak cukup.")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in match_andal]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in match_andal]).reshape(-1, 1, 2)

    threshold_list = [1.0, 3.0, 5.0, 10.0]
    hasil          = []   # (threshold, n_inlier, pct)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, thresh in zip(axes.ravel(), threshold_list):
        H_mat, mask = cv2.findHomography(src_pts, dst_pts,
                                         cv2.RANSAC,
                                         ransacReprojThreshold=thresh)
        if mask is None:
            hasil.append((thresh, 0, 0))
            ax.set_title(f"threshold={thresh} — GAGAL")
            ax.axis("off")
            continue

        mask     = mask.ravel()
        n_in     = int(mask.sum())
        pct      = n_in / len(mask) * 100
        hasil.append((thresh, n_in, pct))
        print(f"  threshold={thresh:4.1f}: inlier={n_in}/{len(mask)} ({pct:.1f}%)")

        # Gambar kanvas dengan warna inlier/outlier
        H1, W1 = img1_bgr.shape[:2]
        H2, W2 = img2_bgr.shape[:2]
        kanvas  = np.zeros((max(H1, H2), W1+W2, 3), dtype=np.uint8)
        kanvas[:H1, :W1] = img1_bgr
        kanvas[:H2, W1:] = img2_bgr

        for m, f in zip(match_andal, mask):
            p1   = tuple(map(int, kp1[m.queryIdx].pt))
            p2   = (int(kp2[m.trainIdx].pt[0])+W1, int(kp2[m.trainIdx].pt[1]))
            warna = (0, 200, 0) if f else (0, 0, 200)
            cv2.line(kanvas, p1, p2, warna, 1, cv2.LINE_AA)

        ax.imshow(cv2.cvtColor(kanvas, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"ransacReprojThreshold = {thresh}\n"
            f"Inlier: {n_in}/{len(mask)} ({pct:.1f}%)",
            fontsize=11)
        ax.axis("off")

    plt.suptitle(
        "Pengaruh ransacReprojThreshold terhadap Jumlah Inlier\n"
        "(hijau = inlier, merah = outlier)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_demo4_ransac_threshold.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Modul 04 — 08: Homography dan RANSAC")
    print("=" * 60)

    gray1, gray2, img1_bgr, img2_bgr = muat_gambar_pasangan()
    print(f"Ukuran gambar-1: {gray1.shape}, gambar-2: {gray2.shape}")

    demo_homography_dasar          (gray1, gray2, img1_bgr, img2_bgr)
    demo_warp_perspective          (gray1, gray2, img1_bgr, img2_bgr)
    demo_visualisasi_inlier_ransac (gray1, gray2, img1_bgr, img2_bgr)
    demo_pengaruh_ransac_threshold (gray1, gray2, img1_bgr, img2_bgr)

    print("\nSemua demo selesai. Output disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
