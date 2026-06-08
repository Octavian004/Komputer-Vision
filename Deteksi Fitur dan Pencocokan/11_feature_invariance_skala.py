"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
File    : 11_feature_invariance_skala.py
Topik   : Feature Invariansi Terhadap Skala
Deskripsi:
    Menguji ketahanan detektor fitur (SIFT, ORB, AKAZE) terhadap perubahan
    skala gambar, membandingkan jumlah keypoint dan match survive di berbagai
    skala, serta memvisualisasikan scale-space dengan keypoint multi-skala.
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
    Memuat gambar dari IMAGE_DIR atau membuat gambar sintetis.
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


def skalakan_gambar(gray, faktor):
    """Mengubah ukuran gambar gray dengan faktor skala, kemudian resize ke ukuran asal."""
    H, W      = gray.shape[:2]
    H_baru    = max(1, int(H * faktor))
    W_baru    = max(1, int(W * faktor))
    gray_kecil = cv2.resize(gray, (W_baru, H_baru))
    # Kembalikan ke ukuran semula agar muat satu kanvas
    return cv2.resize(gray_kecil, (W, H))


# ─────────────────────────────────────────────────────────────────────────────
def demo_keypoint_per_skala(gray_ref, img_bgr):
    """
    Demo 1: Jumlah keypoint SIFT pada scale 0.25 / 0.5 / 1.0 / 2.0.
    """
    print("\n[Demo 1] Keypoint SIFT pada berbagai level skala")

    skala_list = [0.25, 0.5, 1.0, 2.0]
    sift       = cv2.SIFT_create()
    H, W       = gray_ref.shape[:2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, skala in zip(axes.ravel(), skala_list):
        # Resize gambar sesuai skala
        H_s    = max(1, int(H * skala))
        W_s    = max(1, int(W * skala))
        gray_s = cv2.resize(gray_ref, (W_s, H_s))
        kp, _  = sift.detectAndCompute(gray_s, None)

        # Gambar keypoint pada gambar yang discale
        gb_s   = cv2.cvtColor(gray_s, cv2.COLOR_GRAY2BGR)
        gb_s   = cv2.drawKeypoints(
            gb_s, kp, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(gb_s, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"Scale = {skala}x  ({W_s}×{H_s} px)\n"
            f"Keypoint: {len(kp)}",
            fontsize=10)
        ax.axis("off")

    plt.suptitle("Keypoint SIFT pada Berbagai Level Skala", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_demo1_keypoint_per_skala.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    for skala in skala_list:
        H_s = max(1, int(H * skala))
        W_s = max(1, int(W * skala))
        g   = cv2.resize(gray_ref, (W_s, H_s))
        kp, _ = sift.detectAndCompute(g, None)
        print(f"  Scale={skala}: {len(kp)} keypoint pada {W_s}×{H_s}")
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_match_survive_per_skala(gray_ref, img_bgr):
    """
    Demo 2: Jumlah match SIFT: gambar original vs berbagai faktor skala (0.3–3.0).
    """
    print("\n[Demo 2] Match SIFT survive vs faktor skala (0.3x – 3.0x)")

    sift       = cv2.SIFT_create()
    kp0, des0  = sift.detectAndCompute(gray_ref, None)
    H, W       = gray_ref.shape[:2]

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))

    skala_list = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    match_list = []

    for skala in skala_list:
        H_s    = max(1, int(H * skala))
        W_s    = max(1, int(W * skala))
        gray_s = cv2.resize(gray_ref, (W_s, H_s))
        kp1, des1 = sift.detectAndCompute(gray_s, None)

        if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
            match_list.append(0); continue

        pasang     = flann.knnMatch(des0, des1, k=2)
        match_baik = [m for m, n in pasang
                      if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        match_list.append(len(match_baik))
        print(f"  Scale={skala}: {len(match_baik)} match  "
              f"({len(match_baik)/max(len(kp0),1)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(skala_list, match_list, "s-", color="#e74c3c", linewidth=2,
            markersize=8, label="SIFT match survive")
    ax.fill_between(skala_list, match_list, alpha=0.15, color="#e74c3c")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.6, label="Skala asli (1.0x)")
    ax.set_xlabel("Faktor Skala", fontsize=12)
    ax.set_ylabel("Jumlah Match Survive", fontsize=12)
    ax.set_title("Match SIFT vs Faktor Skala (0.3x – 3.0x)", fontsize=13,
                 fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_demo2_match_survive_skala.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_bandingkan_detektor_skala(gray_ref, img_bgr):
    """
    Demo 3: Perbandingan ORB vs SIFT vs AKAZE terhadap perubahan skala.
    Line plot jumlah match untuk setiap detektor.
    """
    print("\n[Demo 3] Perbandingan ORB vs SIFT vs AKAZE — invariansi skala")

    H, W       = gray_ref.shape[:2]
    skala_list = [0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2]

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
        for skala in skala_list:
            H_s    = max(1, int(H * skala))
            W_s    = max(1, int(W * skala))
            gray_s = cv2.resize(gray_ref, (W_s, H_s))
            kp1, des1 = det.detectAndCompute(gray_s, None)
            if des1 is None or len(kp1) < 2:
                match_list.append(0); continue
            if tipe == "uint8":
                des1 = np.uint8(des1)
            try:
                pasang     = matcher.knnMatch(des0, des1, k=2)
                match_baik = [p[0] for p in pasang
                              if len(p) == 2 and p[0].distance < rasio * p[1].distance]
                match_list.append(len(match_baik))
            except Exception:
                match_list.append(0)

        ax.plot(skala_list, match_list, "o-", color=warna[nama], linewidth=2,
                markersize=7, label=nama)
        print(f"  {nama:5s}: {match_list}")

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="Skala asli")
    ax.set_xlabel("Faktor Skala", fontsize=12)
    ax.set_ylabel("Jumlah Match Survive", fontsize=12)
    ax.set_title("Perbandingan Detektor: Invariansi Terhadap Skala",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_demo3_bandingkan_detektor_skala.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_visualisasi_scale_space(gray_ref, img_bgr):
    """
    Demo 4: Visualisasi scale-space — keypoint dari berbagai level sigma
    digambar dengan lingkaran berukuran proporsional terhadap sigma.
    """
    print("\n[Demo 4] Visualisasi scale-space keypoint SIFT (lingkaran proporsional)")

    sift      = cv2.SIFT_create()
    kp_semua, _ = sift.detectAndCompute(gray_ref, None)

    # Kelompokkan keypoint ke dalam 4 kelompok berdasarkan ukuran
    ukuran_arr  = np.array([k.size for k in kp_semua])
    if len(ukuran_arr) == 0:
        print("  [SKIP] Tidak ada keypoint.")
        return

    batas = np.percentile(ukuran_arr, [25, 50, 75, 100])
    label = ["σ kecil (0–25%)", "σ sedang (25–50%)",
             "σ besar (50–75%)", "σ sangat besar (75–100%)"]
    warna = ["#3498db", "#27ae60", "#f39c12", "#e74c3c"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    batas_bawah = 0

    for ax, batas_atas, lab, col in zip(axes.ravel(), batas, label, warna):
        kelompok = [k for k in kp_semua
                    if batas_bawah <= k.size <= batas_atas]
        batas_bawah = batas_atas

        gb = cv2.cvtColor(gray_ref, cv2.COLOR_GRAY2BGR)
        # Gambar lingkaran manual dengan radius proporsional
        for k in kelompok[:100]:   # batasi 100 agar tidak terlalu padat
            cx, cy = int(k.pt[0]), int(k.pt[1])
            r      = int(k.size / 2)
            cv2.circle(gb, (cx, cy), max(r, 2), (0, 200, 255), 1, cv2.LINE_AA)
            cv2.circle(gb, (cx, cy), 2,          (0, 0, 255), -1)

        ax.imshow(cv2.cvtColor(gb, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{lab}\n"
            f"Ukuran: {batas_atas:.1f}px | "
            f"{len(kelompok)} keypoint",
            fontsize=10, color=col)
        ax.axis("off")

    plt.suptitle("Visualisasi Scale-Space: Keypoint SIFT per Level σ\n"
                 "(Diameter lingkaran ∝ ukuran keypoint)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_demo4_scale_space_visualization.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Modul 04 — 11: Feature Invariansi Terhadap Skala")
    print("=" * 60)

    gray_ref, img_bgr = muat_gambar_pasangan()
    print(f"Ukuran gambar referensi: {gray_ref.shape}")

    demo_keypoint_per_skala        (gray_ref, img_bgr)
    demo_match_survive_per_skala   (gray_ref, img_bgr)
    demo_bandingkan_detektor_skala (gray_ref, img_bgr)
    demo_visualisasi_scale_space   (gray_ref, img_bgr)

    print("\nSemua demo selesai. Output disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
