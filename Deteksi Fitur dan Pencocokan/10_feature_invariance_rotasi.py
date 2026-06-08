"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
File    : 10_feature_invariance_rotasi.py
Topik   : Feature Invariansi Terhadap Rotasi
Deskripsi:
    Menguji ketahanan detektor fitur (SIFT, ORB, AKAZE) terhadap berbagai
    sudut rotasi, membandingkan jumlah keypoint dan rasio match yang survive,
    serta memvisualisasikan orientasi deskriptor dengan quiver plot.
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
    Memuat satu gambar referensi dari IMAGE_DIR atau membuat gambar sintetis.
    Mengembalikan (gray_ref, img_bgr_ref).
    """
    for nama in ["chess.png", "box.png", "simple.png", "scene.png"]:
        jalur = os.path.join(IMAGE_DIR, nama)
        if os.path.exists(jalur):
            img = cv2.imread(jalur)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img

    H, W = 480, 640
    img = np.full((H, W, 3), (40, 20, 60), dtype=np.uint8)
    cv2.rectangle(img, (60,  50),  (260, 220), (180, 100, 60), -1)
    cv2.circle   (img, (430, 150), 100,        (60, 200, 180), -1)
    pts = np.array([[200, 330], [500, 390], [120, 420]], np.int32)
    cv2.fillPoly (img, [pts],               (220, 80,  80))
    cv2.rectangle(img, (360, 280),  (600, 440),(60, 120, 220), -1)
    cv2.circle   (img, (150, 380),   55,       (220, 200, 40), -1)
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    img   = cv2.add(img, noise)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img


def rotasi_gambar(gray, sudut_derajat):
    """Merotasi gambar gray sejumlah sudut_derajat, return gray hasil rotasi."""
    H, W  = gray.shape[:2]
    pusat = (W // 2, H // 2)
    M     = cv2.getRotationMatrix2D(pusat, sudut_derajat, 1.0)
    return cv2.warpAffine(gray, M, (W, H), borderMode=cv2.BORDER_REFLECT)


# ─────────────────────────────────────────────────────────────────────────────
def demo_keypoint_per_sudut(gray_ref, img_bgr):
    """
    Demo 1: Jumlah keypoint SIFT pada gambar yang dirotasi 0°/45°/90°/135°/180°.
    """
    print("\n[Demo 1] Keypoint SIFT pada berbagai sudut rotasi (0°–180°)")

    sudut_list = [0, 45, 90, 135, 180]
    sift       = cv2.SIFT_create()
    jumlah_kp  = []

    fig, axes = plt.subplots(1, len(sudut_list), figsize=(16, 4))
    for ax, sudut in zip(axes, sudut_list):
        gray_rot = rotasi_gambar(gray_ref, sudut)
        kp, _    = sift.detectAndCompute(gray_rot, None)
        jumlah_kp.append(len(kp))

        # Gambar keypoint pada gambar rotasi
        gb = cv2.drawKeypoints(
            cv2.cvtColor(gray_rot, cv2.COLOR_GRAY2BGR), kp, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(gb, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Rotasi {sudut}°\n{len(kp)} keypoint", fontsize=10)
        ax.axis("off")

    plt.suptitle("Keypoint SIFT pada Berbagai Sudut Rotasi", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_demo1_keypoint_per_sudut.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    for s, n in zip(sudut_list, jumlah_kp):
        print(f"  Rotasi {s:3d}°: {n} keypoint")
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_match_survive_per_rotasi(gray_ref, img_bgr):
    """
    Demo 2: Rasio match SIFT yang survive vs sudut rotasi (setiap 15° dari 0–180°).
    Menunjukkan kurva invariansi terhadap rotasi.
    """
    print("\n[Demo 2] % match SIFT survive vs sudut rotasi (0°–180°, step 15°)")

    sift       = cv2.SIFT_create()
    kp0, des0  = sift.detectAndCompute(gray_ref, None)
    flann      = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))

    sudut_list = list(range(0, 181, 15))
    pct_list   = []

    for sudut in sudut_list:
        gray_rot    = rotasi_gambar(gray_ref, sudut)
        kp1, des1   = sift.detectAndCompute(gray_rot, None)

        if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
            pct_list.append(0)
            continue

        pasang      = flann.knnMatch(des0, des1, k=2)
        match_baik  = [m for m, n in pasang
                       if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        pct         = len(match_baik) / max(len(kp0), 1) * 100
        pct_list.append(pct)
        print(f"  Rotasi {sudut:3d}°: {len(match_baik)}/{len(kp0)} match "
              f"({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sudut_list, pct_list, "o-", color="#27ae60", linewidth=2,
            markersize=7, label="SIFT match survive")
    ax.fill_between(sudut_list, pct_list, alpha=0.15, color="#27ae60")
    ax.set_xlabel("Sudut Rotasi (°)", fontsize=12)
    ax.set_ylabel("% Match Survive", fontsize=12)
    ax.set_title("Match SIFT vs Sudut Rotasi (0°–180°, step 15°)", fontsize=13,
                 fontweight="bold")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, max(pct_list) * 1.15 + 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_demo2_match_survive_rotasi.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_bandingkan_detektor_rotasi(gray_ref, img_bgr):
    """
    Demo 3: Perbandingan ORB vs SIFT vs AKAZE terhadap rotasi.
    Line plot jumlah match untuk setiap detektor pada 0°–180° (step 30°).
    """
    print("\n[Demo 3] Perbandingan ORB vs SIFT vs AKAZE — invariansi rotasi")

    sudut_list = list(range(0, 181, 30))

    # Konfigurasi detektor/deskriptor dan matcher
    detektor_info = {
        "SIFT" : (cv2.SIFT_create(),
                  cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),
                                        dict(checks=50)),
                  0.75, "float"),
        "ORB"  : (cv2.ORB_create(nfeatures=1000),
                  cv2.BFMatcher(cv2.NORM_HAMMING),
                  0.75, "uint8"),
        "AKAZE": (cv2.AKAZE_create(),
                  cv2.BFMatcher(cv2.NORM_HAMMING),
                  0.80, "uint8"),
    }
    warna = {"SIFT": "#27ae60", "ORB": "#e74c3c", "AKAZE": "#3498db"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for nama, (det, matcher, rasio, tipe) in detektor_info.items():
        kp0, des0 = det.detectAndCompute(gray_ref, None)
        if des0 is None or len(kp0) == 0:
            continue
        if tipe == "uint8":
            des0 = np.uint8(des0)

        match_list = []
        for sudut in sudut_list:
            gray_rot   = rotasi_gambar(gray_ref, sudut)
            kp1, des1  = det.detectAndCompute(gray_rot, None)
            if des1 is None or len(kp1) < 2:
                match_list.append(0); continue
            if tipe == "uint8":
                des1 = np.uint8(des1)

            try:
                pasang     = matcher.knnMatch(des0, des1, k=2)
                match_baik = [m for p in pasang
                              if len(p) == 2 and p[0].distance < rasio * p[1].distance
                              for m in [p[0]]]
                match_list.append(len(match_baik))
            except Exception:
                match_list.append(0)

        ax.plot(sudut_list, match_list, "o-", color=warna[nama], linewidth=2,
                markersize=7, label=nama)
        print(f"  {nama:5s}: {match_list}")

    ax.set_xlabel("Sudut Rotasi (°)", fontsize=12)
    ax.set_ylabel("Jumlah Match Survive", fontsize=12)
    ax.set_title("Perbandingan Detektor: Invariansi Terhadap Rotasi",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 180)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_demo3_bandingkan_detektor_rotasi.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_quiver_orientasi_keypoint(gray_ref, img_bgr):
    """
    Demo 4: Visualisasi orientasi keypoint SIFT dengan quiver plot.
    Panjang panah proporsional dengan ukuran keypoint.
    """
    print("\n[Demo 4] Quiver plot orientasi descriptor SIFT")

    sift     = cv2.SIFT_create()
    kp, _    = sift.detectAndCompute(gray_ref, None)

    # Batasi jumlah keypoint agar plot tidak terlalu padat
    MAX_KP = 80
    kp_sample = sorted(kp, key=lambda k: k.response, reverse=True)[:MAX_KP]

    # Ekstrak posisi, orientasi, dan ukuran
    xs = np.array([k.pt[0] for k in kp_sample])
    ys = np.array([k.pt[1] for k in kp_sample])
    # Orientasi dalam radian (cv2 menyimpan derajat negatif → konversi)
    angles = np.deg2rad([k.angle for k in kp_sample])
    sizes  = np.array([k.size for k in kp_sample])
    # Komponen vektor arah
    U = np.cos(angles) * sizes * 0.3
    V = np.sin(angles) * sizes * 0.3

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Kiri: gambar asli dengan keypoint
    gb = cv2.drawKeypoints(
        img_bgr, kp_sample, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes[0].imshow(cv2.cvtColor(gb, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Keypoint SIFT dengan Rich Flags\n"
                      f"({len(kp_sample)} ditampilkan dari {len(kp)} total)",
                      fontsize=11)
    axes[0].axis("off")

    # Kanan: quiver plot orientasi
    H, W = gray_ref.shape[:2]
    axes[1].imshow(gray_ref, cmap="gray")
    q = axes[1].quiver(xs, ys, U, -V,   # negatif V karena sumbu y terbalik
                       color="lime", scale=1, scale_units="xy",
                       angles="xy", width=0.003, headwidth=4, headlength=5)
    axes[1].set_title(f"Quiver Plot Orientasi Deskriptor SIFT\n"
                      f"Panjang panah ∝ ukuran keypoint",
                      fontsize=11)
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].axis("off")

    plt.suptitle("Visualisasi Orientasi Keypoint SIFT", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_demo4_quiver_orientasi_keypoint.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Modul 04 — 10: Feature Invariansi Terhadap Rotasi")
    print("=" * 60)

    gray_ref, img_bgr = muat_gambar_pasangan()
    print(f"Ukuran gambar referensi: {gray_ref.shape}")

    demo_keypoint_per_sudut         (gray_ref, img_bgr)
    demo_match_survive_per_rotasi   (gray_ref, img_bgr)
    demo_bandingkan_detektor_rotasi (gray_ref, img_bgr)
    demo_quiver_orientasi_keypoint  (gray_ref, img_bgr)

    print("\nSemua demo selesai. Output disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
