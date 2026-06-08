"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
File    : 12_feature_invariance_iluminasi.py
Topik   : Feature Invariansi Terhadap Iluminasi
Deskripsi:
    Menguji ketahanan detektor SIFT terhadap perubahan kecerahan (brightness),
    kontras (contrast), noise Gaussian, serta kombinasi ketiganya dalam bentuk
    heatmap matrix kondisi pencahayaan vs jumlah match.
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
    cv2.circle   (img, (150, 380),   55,       (220, 200,  40), -1)
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    img   = cv2.add(img, noise)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img


def terapkan_brightness(gray, beta):
    """Mengubah kecerahan gambar: pixel += beta, dijepit ke [0,255]."""
    return np.clip(gray.astype(np.int32) + beta, 0, 255).astype(np.uint8)


def terapkan_kontras(gray, alpha):
    """Mengubah kontras gambar: pixel *= alpha, dijepit ke [0,255]."""
    return np.clip(gray.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def tambah_noise_gaussian(gray, sigma):
    """Menambahkan Gaussian noise dengan standar deviasi sigma."""
    if sigma == 0:
        return gray.copy()
    noise = np.random.normal(0, sigma, gray.shape)
    return np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def match_sift(gray0, gray1, rasio=0.75):
    """
    Menghitung jumlah match SIFT antara gray0 dan gray1 menggunakan
    FLANN + ratio test. Mengembalikan jumlah match yang survive.
    """
    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(gray0, None)
    kp1, des1 = sift.detectAndCompute(gray1, None)

    if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
        return 0

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))
    pasang = flann.knnMatch(des0, des1, k=2)
    return sum(1 for m, n in pasang
               if len([m, n]) == 2 and m.distance < rasio * n.distance)


# ─────────────────────────────────────────────────────────────────────────────
def demo_keypoint_brightness(gray_ref):
    """
    Demo 1: Jumlah keypoint SIFT pada berbagai tingkat kecerahan
    (beta = -100, -50, 0, +50, +100).
    """
    print("\n[Demo 1] Keypoint SIFT vs brightness (beta= -100/-50/0/+50/+100)")

    beta_list  = [-100, -50, 0, 50, 100]
    sift       = cv2.SIFT_create()
    jumlah_kp  = []

    fig, axes = plt.subplots(1, len(beta_list), figsize=(16, 4))
    for ax, beta in zip(axes, beta_list):
        gray_mod = terapkan_brightness(gray_ref, beta)
        kp, _    = sift.detectAndCompute(gray_mod, None)
        jumlah_kp.append(len(kp))

        gb = cv2.drawKeypoints(
            cv2.cvtColor(gray_mod, cv2.COLOR_GRAY2BGR), kp, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(gb, cv2.COLOR_BGR2RGB))
        tanda = "+" if beta > 0 else ("")
        ax.set_title(f"β={tanda}{beta}\n{len(kp)} keypoint", fontsize=10)
        ax.axis("off")

    plt.suptitle("Keypoint SIFT pada Berbagai Tingkat Kecerahan (Beta)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_demo1_keypoint_brightness.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    for b, n in zip(beta_list, jumlah_kp):
        print(f"  beta={b:+4d}: {n} keypoint")
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_match_kontras(gray_ref):
    """
    Demo 2: Jumlah match SIFT pada perubahan kontras
    (alpha = 0.5 / 0.75 / 1.0 / 1.5 / 2.0).
    """
    print("\n[Demo 2] Match SIFT vs perubahan kontras (alpha 0.5–2.0)")

    alpha_list = [0.5, 0.75, 1.0, 1.5, 2.0]
    match_list = []

    for alpha in alpha_list:
        gray_mod  = terapkan_kontras(gray_ref, alpha)
        n_match   = match_sift(gray_ref, gray_mod)
        match_list.append(n_match)
        print(f"  alpha={alpha}: {n_match} match")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot kiri: contoh gambar per alpha
    for i, alpha in enumerate(alpha_list):
        gray_mod = terapkan_kontras(gray_ref, alpha)
    # Hanya tampilkan 3 contoh agar tidak terlalu banyak
    contoh_alpha = [0.5, 1.0, 2.0]
    for idx, a in enumerate(contoh_alpha):
        g = terapkan_kontras(gray_ref, a)
        axes[0].imshow(g, cmap="gray", alpha=0.6 if idx > 0 else 1.0)

    # Bar chart jumlah match
    warna_bar = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9", "#8e44ad"]
    axes[1].bar([str(a) for a in alpha_list], match_list,
                color=warna_bar, edgecolor="black")
    axes[1].set_xlabel("Alpha (kontras)", fontsize=11)
    axes[1].set_ylabel("Jumlah Match Survive", fontsize=11)
    axes[1].set_title("Match SIFT vs Perubahan Kontras (alpha)", fontsize=12)
    for i, v in enumerate(match_list):
        axes[1].text(i, v + max(match_list)*0.02, str(v),
                     ha="center", fontsize=11, fontweight="bold")

    # Ganti subplot kiri dengan contoh gambar
    axes[0].clear()
    for idx, a in enumerate([0.5, 1.0, 2.0]):
        g = terapkan_kontras(gray_ref, a)
        axes[0].imshow(g, cmap="gray", extent=[idx*220, (idx+1)*220-10, 0, 200],
                       aspect="auto")
    axes[0].set_xlim(0, 660)
    axes[0].set_ylim(0, 200)
    axes[0].set_title("Contoh: alpha=0.5 | 1.0 | 2.0", fontsize=11)
    axes[0].axis("off")

    plt.suptitle("Ketahanan SIFT Terhadap Perubahan Kontras",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_demo2_match_kontras.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_resistansi_noise(gray_ref):
    """
    Demo 3: Jumlah match SIFT pada gambar dengan Gaussian noise
    (sigma = 0 / 10 / 30 / 50).
    """
    print("\n[Demo 3] Resistansi terhadap Gaussian noise (sigma 0/10/30/50)")

    sigma_list = [0, 10, 30, 50]
    match_list = []

    fig, axes = plt.subplots(2, len(sigma_list), figsize=(16, 7))

    for col, sigma in enumerate(sigma_list):
        gray_noise = tambah_noise_gaussian(gray_ref, sigma)
        n_match    = match_sift(gray_ref, gray_noise)
        match_list.append(n_match)

        # Baris atas: gambar dengan noise
        axes[0][col].imshow(gray_noise, cmap="gray")
        axes[0][col].set_title(f"σ = {sigma}\n(PSNR ~{10*np.log10(255**2/max(sigma**2,0.01)):.0f}dB)",
                               fontsize=10)
        axes[0][col].axis("off")

        # Baris bawah: bar match
        axes[1][col].bar([f"σ={sigma}"], [n_match],
                         color="#3498db" if sigma == 0 else "#e74c3c")
        axes[1][col].set_ylim(0, max(match_list + [1]) * 1.2)
        axes[1][col].set_ylabel("Match")
        axes[1][col].text(0, n_match + 1, str(n_match), ha="center",
                          fontsize=12, fontweight="bold")

    plt.suptitle("Resistansi SIFT terhadap Gaussian Noise",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_demo3_resistansi_noise.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    for s, m in zip(sigma_list, match_list):
        print(f"  sigma={s:2d}: {m} match")
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_heatmap_combined(gray_ref):
    """
    Demo 4: Heatmap matrix 3×3 — kombinasi brightness (baris) dan
    noise+blur (kolom) vs jumlah match SIFT.
    """
    print("\n[Demo 4] Heatmap: brightness × noise+blur kondisi → jumlah match")

    # Baris: tingkat brightness
    brightness_vals = [-60, 0, +60]
    brightness_label = ["Gelap (β=-60)", "Normal (β=0)", "Terang (β=+60)"]

    # Kolom: kombinasi noise sigma + blur kernel
    kondisi_cols = [
        (0,  0,   "Bersih\n(σ=0, blur=0)"),
        (20, 3,   "Noise σ=20\nBlur k=3"),
        (40, 5,   "Noise σ=40\nBlur k=5"),
    ]

    matriks = np.zeros((3, 3), dtype=int)

    for r, beta in enumerate(brightness_vals):
        gray_b = terapkan_brightness(gray_ref, beta)
        for c, (sigma, blur_k, _) in enumerate(kondisi_cols):
            gray_mod = tambah_noise_gaussian(gray_b, sigma)
            if blur_k > 0:
                gray_mod = cv2.GaussianBlur(gray_mod, (blur_k, blur_k), 0)
            n_match = match_sift(gray_ref, gray_mod)
            matriks[r, c] = n_match
            print(f"  [β={beta:+3d}, σ={sigma:2d}, blur={blur_k}] → {n_match} match")

    # Gambar heatmap
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matriks, cmap="YlOrRd_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Jumlah Match Survive")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([c[2] for c in kondisi_cols], fontsize=10)
    ax.set_yticklabels(brightness_label, fontsize=10)
    ax.set_xlabel("Kondisi Noise + Blur", fontsize=11)
    ax.set_ylabel("Kondisi Kecerahan (Brightness)", fontsize=11)
    ax.set_title("Heatmap: Jumlah Match SIFT\nvs Kombinasi Brightness × Noise+Blur",
                 fontsize=12, fontweight="bold")

    # Anotasi nilai di tiap sel
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(matriks[r, c]),
                    ha="center", va="center", fontsize=14,
                    fontweight="bold",
                    color="white" if matriks[r, c] < matriks.max() * 0.4 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_demo4_heatmap_combined.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Modul 04 — 12: Feature Invariansi Terhadap Iluminasi")
    print("=" * 60)

    gray_ref, img_bgr = muat_gambar_pasangan()
    print(f"Ukuran gambar referensi: {gray_ref.shape}")

    demo_keypoint_brightness(gray_ref)
    demo_match_kontras      (gray_ref)
    demo_resistansi_noise   (gray_ref)
    demo_heatmap_combined   (gray_ref)

    print("\nSemua demo selesai. Output disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
