"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
File    : 06_brute_force_matching.py
Topik   : Brute Force Matching dengan BFMatcher
Deskripsi:
    Mempelajari cara mencocokkan deskriptor fitur menggunakan BFMatcher OpenCV,
    termasuk knnMatch + Lowe ratio test, pengaruh threshold, serta perbandingan
    cross-check matching versus ratio test.
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
    Memuat atau membuat sepasang gambar sintetis untuk demonstrasi matching.
    Gambar pertama : adegan warna solid dengan berbagai bentuk geometris.
    Gambar kedua   : gambar pertama yang telah dirotasi 25° dan di-scale 0.85.
    Mengembalikan  : (gray1, gray2, img1_bgr, img2_bgr)
    """
    # ── Coba muat dari direktori image ────────────────────────────────────────
    for nama in ["chess.png", "box.png", "simple.png", "scene.png"]:
        jalur = os.path.join(IMAGE_DIR, nama)
        if os.path.exists(jalur):
            img = cv2.imread(jalur)
            if img is not None:
                h, w  = img.shape[:2]
                pusat = (w // 2, h // 2)
                M     = cv2.getRotationMatrix2D(pusat, 30, 0.9)
                img2  = cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT)
                return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                        img, img2)

    # ── Buat gambar sintetis jika tidak ada file gambar ───────────────────────
    H, W = 480, 640
    img1 = np.full((H, W, 3), (60, 30, 10), dtype=np.uint8)
    cv2.rectangle(img1, (80,  60),  (250, 200), (200, 100,  50), -1)
    cv2.circle   (img1, (420, 150),  90,        (50,  180, 220), -1)
    pts = np.array([[300, 350], [500, 420], [200, 430]], np.int32)
    cv2.fillPoly (img1, [pts],               (80,  220, 100))
    cv2.rectangle(img1, (50,  300),  (180, 430),(180,  60, 200), -1)
    cv2.circle   (img1, (560, 360),   60,       (230, 200,  40), -1)
    # Tambahkan noise tekstur agar detektor menemukan cukup keypoint
    noise = np.random.randint(0, 30, img1.shape, dtype=np.uint8)
    img1  = cv2.add(img1, noise)

    # Buat gambar kedua dengan rotasi 25° dan scale 0.85
    pusat = (W // 2, H // 2)
    M     = cv2.getRotationMatrix2D(pusat, 25, 0.85)
    img2  = cv2.warpAffine(img1, M, (W, H), borderMode=cv2.BORDER_REFLECT)

    return (cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
            img1, img2)


# ─────────────────────────────────────────────────────────────────────────────
def demo_bfmatcher_orb_dasar(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 1: BFMatcher dengan ORB menggunakan NORM_HAMMING.
    Semua DMatch diurutkan berdasarkan jarak (ascending) lalu top-30 ditampilkan.
    """
    print("\n[Demo 1] BFMatcher ORB — urutkan & tampilkan top-30 DMatch")

    # Deteksi dan komputasi deskriptor ORB
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    print(f"  Keypoint gambar-1: {len(kp1)}, gambar-2: {len(kp2)}")

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("  [SKIP] Deskriptor kosong.")
        return

    # BFMatcher dengan NORM_HAMMING — cocok untuk deskriptor biner ORB
    bf          = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    semua_match = bf.match(des1, des2)

    # Urutkan berdasarkan jarak; semakin kecil → semakin mirip
    semua_match = sorted(semua_match, key=lambda m: m.distance)
    top30       = semua_match[:30]
    print(f"  Total match    : {len(semua_match)}")
    print(f"  Jarak min      : {semua_match[0].distance:.2f}")
    print(f"  Jarak max      : {semua_match[-1].distance:.2f}")
    print(f"  Jarak rata-rata top-30: {np.mean([m.distance for m in top30]):.2f}")

    # Gambar match menggunakan drawMatches
    gambar_match = cv2.drawMatches(
        img1_bgr, kp1, img2_bgr, kp2, top30, None,
        matchColor=(0, 255, 0),        # garis hijau untuk match
        singlePointColor=(255, 0, 0),  # titik biru untuk keypoint tidak match
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(cv2.cvtColor(gambar_match, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"BFMatcher ORB (NORM_HAMMING) — Top-30 DMatch\n"
        f"Total: {len(semua_match)} match | "
        f"Jarak rata-rata top-30: {np.mean([m.distance for m in top30]):.2f}",
        fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_demo1_bfmatcher_orb_top30.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_knn_match_ratio_test(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 2: BFMatcher knnMatch(k=2) + Lowe ratio test (ratio=0.75).
    Membedakan secara visual antara match yang lulus vs gagal ratio test.
    """
    print("\n[Demo 2] knnMatch(k=2) + Lowe ratio test (ratio=0.75)")

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("  [SKIP] Deskriptor tidak cukup.")
        return

    bf         = cv2.BFMatcher(cv2.NORM_HAMMING)
    # knnMatch mengembalikan 2 kandidat terbaik untuk setiap deskriptor kueri
    pasang_knn = bf.knnMatch(des1, des2, k=2)

    RATIO        = 0.75
    match_baik   = []   # match yang lulus ratio test
    match_buruk  = []   # match yang gagal ratio test (ambigu)

    for pasang in pasang_knn:
        if len(pasang) < 2:
            continue
        m, n = pasang
        if m.distance < RATIO * n.distance:
            # Jarak terpendek jauh lebih kecil dari kedua terpendek → andal
            match_baik.append([m])
        else:
            # Kedua kandidat terlalu dekat satu sama lain → ambigu
            match_buruk.append([m])

    print(f"  Total pasang knn   : {len(pasang_knn)}")
    print(f"  Lulus ratio test   : {len(match_baik)} match")
    print(f"  Gagal ratio test   : {len(match_buruk)} match")

    # Visualisasi match baik (hijau) dan match buruk (merah)
    gb_baik  = cv2.drawMatchesKnn(
        img1_bgr, kp1, img2_bgr, kp2, match_baik[:30], None,
        matchColor=(0, 220, 0), singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    gb_buruk = cv2.drawMatchesKnn(
        img1_bgr, kp1, img2_bgr, kp2, match_buruk[:30], None,
        matchColor=(0, 0, 220), singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    axes[0].imshow(cv2.cvtColor(gb_baik,  cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"✔ Lulus Ratio Test (ratio={RATIO}) — {len(match_baik)} match",
        fontsize=11, color="green")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(gb_buruk, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"✘ Gagal Ratio Test (ambigu) — {len(match_buruk)} match",
        fontsize=11, color="red")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_demo2_knnmatch_ratio_test.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_pengaruh_ratio_threshold(gray1, gray2):
    """
    Demo 3: Pengaruh nilai ratio threshold (0.5 / 0.6 / 0.75 / 0.9) terhadap
    jumlah match yang survive setelah Lowe ratio test.
    """
    print("\n[Demo 3] Pengaruh nilai ratio threshold terhadap jumlah match survive")

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("  [SKIP] Deskriptor tidak cukup.")
        return

    bf         = cv2.BFMatcher(cv2.NORM_HAMMING)
    pasang_knn = bf.knnMatch(des1, des2, k=2)
    pasang_valid = [p for p in pasang_knn if len(p) == 2]

    # Daftar nilai threshold yang diuji
    threshold_list = [0.5, 0.6, 0.75, 0.9]
    jumlah_survive = []
    persentase     = []

    for rasio in threshold_list:
        survive = sum(
            1 for p in pasang_valid
            if p[0].distance < rasio * p[1].distance
        )
        persen = (survive / len(pasang_valid) * 100) if pasang_valid else 0
        jumlah_survive.append(survive)
        persentase.append(persen)
        print(f"  Ratio={rasio}: {survive}/{len(pasang_valid)} survive ({persen:.1f}%)")

    # Bar chart perbandingan jumlah dan persentase match survive
    warna = ["#e74c3c", "#e67e22", "#27ae60", "#3498db"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar([str(t) for t in threshold_list], jumlah_survive,
                color=warna, edgecolor="black", width=0.5)
    axes[0].set_xlabel("Ratio Threshold", fontsize=11)
    axes[0].set_ylabel("Jumlah Match Survive", fontsize=11)
    axes[0].set_title("Jumlah Match vs Ratio Threshold", fontsize=12)
    for i, v in enumerate(jumlah_survive):
        axes[0].text(i, v + max(jumlah_survive) * 0.02,
                     str(v), ha="center", fontsize=11, fontweight="bold")

    axes[1].bar([str(t) for t in threshold_list], persentase,
                color=warna, edgecolor="black", width=0.5)
    axes[1].set_xlabel("Ratio Threshold", fontsize=11)
    axes[1].set_ylabel("Persentase Survive (%)", fontsize=11)
    axes[1].set_title("% Match Survive vs Ratio Threshold", fontsize=12)
    axes[1].set_ylim(0, 110)
    for i, v in enumerate(persentase):
        axes[1].text(i, v + 2, f"{v:.1f}%",
                     ha="center", fontsize=11, fontweight="bold")

    plt.suptitle(
        "Pengaruh Ratio Threshold pada Lowe Ratio Test (ORB + BFMatcher)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_demo3_pengaruh_ratio_threshold.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_crosscheck_vs_ratio_test(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 4: Cross-check matching vs ratio test — membandingkan presisi vs recall.
    Cross-check: match valid hanya jika simetris (A→B dan B→A sama).
    Ratio test : lebih banyak match ditemukan namun mungkin ada false positive.
    """
    print("\n[Demo 4] Cross-check matching vs Ratio test")

    orb = cv2.ORB_create(nfeatures=800)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("  [SKIP] Deskriptor tidak cukup.")
        return

    # ── Metode 1: Cross-check matching ───────────────────────────────────────
    bf_cross    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    match_cross = sorted(bf_cross.match(des1, des2), key=lambda m: m.distance)

    # ── Metode 2: Ratio test (threshold=0.75) ─────────────────────────────────
    bf_knn      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pasang_knn  = bf_knn.knnMatch(des1, des2, k=2)
    RATIO       = 0.75
    match_ratio = [p[0] for p in pasang_knn
                   if len(p) == 2 and p[0].distance < RATIO * p[1].distance]

    print(f"  Cross-check → {len(match_cross)} match")
    print(f"  Ratio test  → {len(match_ratio)} match")
    if match_cross:
        print(f"  Jarak rata-rata cross-check: "
              f"{np.mean([m.distance for m in match_cross]):.2f}")
    if match_ratio:
        print(f"  Jarak rata-rata ratio test : "
              f"{np.mean([m.distance for m in match_ratio]):.2f}")

    TOP = 25
    gb_cross = cv2.drawMatches(
        img1_bgr, kp1, img2_bgr, kp2, match_cross[:TOP], None,
        matchColor=(0, 220, 120), singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    gb_ratio = cv2.drawMatches(
        img1_bgr, kp1, img2_bgr, kp2, match_ratio[:TOP], None,
        matchColor=(60, 120, 240), singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    axes[0].imshow(cv2.cvtColor(gb_cross, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"Cross-Check Matching — {len(match_cross)} match total "
        f"(ditampilkan top-{TOP})\n"
        "Presisi tinggi, recall lebih rendah",
        fontsize=11, color="#116611")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(gb_ratio, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Ratio Test (ratio={RATIO}) — {len(match_ratio)} match total "
        f"(ditampilkan top-{TOP})\n"
        "Recall lebih tinggi, mungkin ada false positive",
        fontsize=11, color="#114488")
    axes[1].axis("off")

    fig.text(0.5, 0.01,
             f"Selisih jumlah match: {abs(len(match_ratio)-len(match_cross))} "
             f"(ratio test lebih banyak {len(match_ratio)-len(match_cross):+d})",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "06_demo4_crosscheck_vs_ratio.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Modul 04 — 06: Brute Force Matching dengan BFMatcher")
    print("=" * 60)

    gray1, gray2, img1_bgr, img2_bgr = muat_gambar_pasangan()
    print(f"Ukuran gambar-1: {gray1.shape}, gambar-2: {gray2.shape}")

    demo_bfmatcher_orb_dasar     (gray1, gray2, img1_bgr, img2_bgr)
    demo_knn_match_ratio_test    (gray1, gray2, img1_bgr, img2_bgr)
    demo_pengaruh_ratio_threshold(gray1, gray2)
    demo_crosscheck_vs_ratio_test(gray1, gray2, img1_bgr, img2_bgr)

    print("\nSemua demo selesai. Output disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
