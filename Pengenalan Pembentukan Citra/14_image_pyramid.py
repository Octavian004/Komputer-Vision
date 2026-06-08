"""
==========================================================================
 PERCOBAAN 14 — IMAGE PYRAMID
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Membangun dan memahami Gaussian Pyramid dan Laplacian Pyramid
           sebagai representasi multi-skala gambar.
 Konsep  :
   - GAUSSIAN PYRAMID: setiap level di-blur (5×5 Gaussian) lalu
     di-downsample 2× menggunakan cv2.pyrDown().
     Digunakan untuk: deteksi fitur multi-skala, image blending.
   - LAPLACIAN PYRAMID: menyimpan DETAIL yang hilang antar level Gaussian.
     L_i = G_i − upsample(G_{i+1})
     Digunakan untuk: image blending tanpa seam, kompresi gambar.
   - Rekonstruksi: L_n → L_{n-1} → ... → L_0 menghasilkan gambar asli.
   - cv2.pyrDown()  : blur + downsample (÷2 setiap dimensi).
   - cv2.pyrUp()    : upsample (×2) + blur (bukan invers pyrDown!).
 Catatan : Laplacian pyramid sangat berguna untuk seamless image blending
           karena menggabungkan di domain frekuensi berbeda per level.
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def baca_gambar(nama: str, target_size=None) -> np.ndarray:
    path = os.path.join(IMAGE_DIR, nama)
    img = cv2.imread(path)
    if img is None:
        h, w = (256, 256) if target_size is None else target_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, 20):
            cv2.line(img, (0, y), (w, y), (150, 150, 150), 1)
        cv2.circle(img, (w//4, h//2), h//3, (0, 120, 200), -1)
        cv2.rectangle(img, (w//2, h//4), (3*w//4, 3*h//4), (200, 80, 0), -1)
    if target_size:
        img = cv2.resize(img, target_size[::-1])  # (w, h)
    return img


# ── Bangun Gaussian Pyramid ────────────────────────────────────────────────────
def buat_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """
    Membangun Gaussian Pyramid dengan pyrDown() sebanyak `levels` kali.
    Setiap level berukuran ½ dari level sebelumnya.
    """
    pyramid = [img.copy()]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


# ── Bangun Laplacian Pyramid ──────────────────────────────────────────────────
def buat_laplacian_pyramid(gaussian_pyr: list) -> list:
    """
    Membangun Laplacian Pyramid dari Gaussian Pyramid:
    L_i = G_i − pyrUp(G_{i+1})
    Level terakhir = level terkecil dari Gaussian Pyramid.
    """
    laplacian = []
    for i in range(len(gaussian_pyr) - 1):
        G_curr = gaussian_pyr[i]
        G_next = gaussian_pyr[i + 1]
        G_up   = cv2.pyrUp(G_next, dstsize=(G_curr.shape[1], G_curr.shape[0]))
        L      = cv2.subtract(G_curr, G_up)
        laplacian.append(L)
    laplacian.append(gaussian_pyr[-1])  # level terkecil
    return laplacian


# ── Rekonstruksi dari Laplacian Pyramid ──────────────────────────────────────
def rekonstruksi(laplacian_pyr: list) -> np.ndarray:
    """
    Merekonstruksi gambar asli dari Laplacian Pyramid:
    Mulai dari level terkecil, add L_i dan pyrUp ke level berikutnya.
    """
    img = laplacian_pyr[-1].copy()
    for L in reversed(laplacian_pyr[:-1]):
        img = cv2.add(cv2.pyrUp(img, dstsize=(L.shape[1], L.shape[0])), L)
    return img


# ── Demo 1: Gaussian Pyramid ──────────────────────────────────────────────────
def demo_gaussian_pyramid(img):
    """Visualisasi Gaussian Pyramid 4 level."""
    levels = 4
    gp = buat_gaussian_pyramid(img, levels)

    fig, axes = plt.subplots(1, levels + 1, figsize=(18, 4))
    for i, (ax, level) in enumerate(zip(axes, gp)):
        ax.imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Level {i}\n{level.shape[1]}×{level.shape[0]}")
        ax.axis("off")

    plt.suptitle("Percobaan 14a — Gaussian Pyramid (4 level)", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "14_gaussian_pyramid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")
    return gp


# ── Demo 2: Laplacian Pyramid ─────────────────────────────────────────────────
def demo_laplacian_pyramid(gp):
    """Visualisasi Laplacian Pyramid — setiap level = detail yang tersimpan."""
    lp = buat_laplacian_pyramid(gp)

    fig, axes = plt.subplots(1, len(lp), figsize=(18, 4))
    for i, (ax, level) in enumerate(zip(axes, lp)):
        # Normalisasi untuk tampilan (nilai bisa negatif)
        vis = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX)
        ax.imshow(cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_BGR2RGB))
        label = f"L{i} (detail)" if i < len(lp)-1 else f"L{i} (base)"
        ax.set_title(f"{label}\n{level.shape[1]}×{level.shape[0]}")
        ax.axis("off")

    plt.suptitle("Percobaan 14b — Laplacian Pyramid (detail tiap level)", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "14_laplacian_pyramid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")

    # Verifikasi rekonstruksi
    rekons = rekonstruksi(lp)
    psnr   = cv2.PSNR(gp[0], rekons)
    print(f"  PSNR rekonstruksi: {psnr:.2f} dB  (> 40 dB = sangat baik)")
    return lp


# ── Demo 3: Laplacian Pyramid Blending ────────────────────────────────────────
def demo_pyramid_blending(img_a, img_b):
    """
    Pyramid blending: menggabungkan dua gambar secara seamless.
    Algoritma:
    1. Bangun LP dari gambar A dan B.
    2. Blend pada setiap level menggunakan mask.
    3. Rekonstruksi dari LP gabungan.
    Hasilnya jauh lebih halus dibanding blending langsung.
    """
    h, w = img_a.shape[:2]
    # Resize agar sama
    img_b_r = cv2.resize(img_b, (w, h))

    levels = 4
    gp_a = buat_gaussian_pyramid(img_a, levels)
    gp_b = buat_gaussian_pyramid(img_b_r, levels)
    lp_a = buat_laplacian_pyramid(gp_a)
    lp_b = buat_laplacian_pyramid(gp_b)

    # Blend: setengah kiri dari A, setengah kanan dari B
    blended_lp = []
    for la, lb in zip(lp_a, lp_b):
        mid = la.shape[1] // 2
        blended = np.hstack([la[:, :mid], lb[:, mid:]])
        blended_lp.append(blended)

    pyramid_blend = rekonstruksi(blended_lp)

    # Perbandingan dengan blend langsung (cut & paste)
    direct_blend = np.hstack([img_a[:, :w//2], img_b_r[:, w//2:]])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, im, title in zip(axes,
                              [img_a, img_b_r, direct_blend, pyramid_blend],
                              ["Gambar A", "Gambar B", "Direct Blend\n(seam terlihat)",
                               "Pyramid Blend\n(seamless)"]):
        ax.imshow(cv2.cvtColor(im.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 14c — Pyramid Blending vs Direct Blend", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "14_pyramid_blending.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 14: IMAGE PYRAMID")
    print("=" * 60)

    # Pastikan ukuran bisa dibagi 2 sebanyak 4 kali (multiple of 16)
    img = baca_gambar("foto_kucing.jpg", target_size=(240, 320))
    img2 = baca_gambar("kota.jpg",       target_size=(240, 320))

    print(f"Ukuran gambar: {img.shape[1]}×{img.shape[0]}")

    print("\n[1] Gaussian Pyramid")
    gp = demo_gaussian_pyramid(img)

    print("\n[2] Laplacian Pyramid & Rekonstruksi")
    lp = demo_laplacian_pyramid(gp)

    print("\n[3] Pyramid Blending (seamless)")
    demo_pyramid_blending(img, img2)

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
