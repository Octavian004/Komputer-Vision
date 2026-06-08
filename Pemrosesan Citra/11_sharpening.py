"""
Modul 03 - Pemrosesan Citra (Image Processing)
Topik  : Sharpening dan Unsharp Masking
Tujuan : Memahami cara mempertajam gambar menggunakan kernel sharpening,
         Unsharp Masking (USM), dan Laplacian sharpening. Membandingkan
         efek berbagai kekuatan penajaman (nilai alpha) pada gambar.
Fungsi : cv2.filter2D(), cv2.addWeighted(), cv2.Laplacian(),
         cv2.GaussianBlur() untuk menghasilkan mask blur pada USM
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Path direktori ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Fungsi bantu ────────────────────────────────────────────────────────────
def muat_atau_buat_gambar():
    """Memuat gambar dari IMAGE_DIR/kota.jpg; jika tidak ada, buat gambar sintetis."""
    jalur = os.path.join(IMAGE_DIR, "kota.jpg")
    if os.path.exists(jalur):
        img = cv2.imread(jalur)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Gambar sintetis: tepi tajam + gradien + teks – ideal untuk uji sharpening
    kanvas = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(300):
        kanvas[i, :200] = [int(i * 0.7), 60, 180]
        kanvas[i, 200:] = [180, 60, int(i * 0.7)]
    cv2.circle(kanvas, (200, 150), 90, (255, 230, 50), -1)
    cv2.rectangle(kanvas, (30, 30), (130, 100), (255, 100, 100), 2)
    cv2.putText(kanvas, "SHARPENING", (75, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return kanvas


# ── Demo 1: Kernel sharpening [[0,-1,0],[-1,5,-1],[0,-1,0]] ─────────────────
def demo_kernel_sharpening(img_rgb):
    """Menerapkan berbagai varian kernel sharpening dan membandingkan hasilnya."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Kernel sharpening standar (Laplacian + pusat = 5)
    k_sharp = np.array([[0, -1,  0],
                        [-1,  5, -1],
                        [0, -1,  0]], dtype=np.float32)
    # Kernel sharpening kuat (diagonal juga dikenakan)
    k_kuat  = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]], dtype=np.float32)
    # Kernel sharpening ringan
    k_ringan = np.array([[0, -0.5, 0],
                         [-0.5, 3, -0.5],
                         [0, -0.5, 0]], dtype=np.float32)

    def terapkan(kern):
        h = cv2.filter2D(abu.astype(np.float32), -1, kern)
        return np.clip(h, 0, 255).astype(np.uint8)

    h_std   = terapkan(k_sharp)
    h_kuat  = terapkan(k_kuat)
    h_ringan = terapkan(k_ringan)

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, img, jud in zip(axs,
            [abu, h_ringan, h_std, h_kuat],
            ["Asli", "Sharp Ringan (×3)", "Sharp Standar (×5)", "Sharp Kuat (×9)"]):
        ax.imshow(img, cmap='gray'); ax.set_title(jud); ax.axis('off')

    plt.suptitle("Demo 1 – Kernel Sharpening", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_kernel_sharpening.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 2: Unsharp Masking (USM) ───────────────────────────────────────────
def demo_unsharp_masking(img_rgb):
    """Menerapkan Unsharp Masking: sharp = original + α × (original − blur)."""
    abu  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Buat blur sebagai 'mask' komponen frekuensi rendah
    blur  = cv2.GaussianBlur(abu, (15, 15), 0).astype(np.float32)
    mask  = abu - blur     # komponen frekuensi tinggi (detail)

    alpha = 1.5            # kekuatan penambahan detail
    usm   = np.clip(abu + alpha * mask, 0, 255).astype(np.uint8)

    # Gunakan cv2.addWeighted sebagai cara alternatif USM
    # sharp = (1+α)×original − α×blur  ≡  addWeighted(orig, 1+α, blur, -α, 0)
    usm_aw = cv2.addWeighted(abu.astype(np.uint8), 1 + alpha,
                             blur.astype(np.uint8), -alpha, 0)

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, img, jud in zip(axs,
            [abu.astype(np.uint8),
             np.clip(mask + 128, 0, 255).astype(np.uint8),
             usm, usm_aw],
            ["Asli", f"Mask (detail α={alpha})",
             f"USM Manual α={alpha}", f"USM addWeighted α={alpha}"]):
        ax.imshow(img, cmap='gray'); ax.set_title(jud); ax.axis('off')

    plt.suptitle("Demo 2 – Unsharp Masking (USM)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_unsharp_masking.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 3: Laplacian sharpening ────────────────────────────────────────────
def demo_laplacian_sharpening(img_rgb):
    """Menggunakan Laplacian sebagai detektor detail untuk proses penajaman."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Hitung Laplacian (turunan orde dua)
    lap = cv2.Laplacian(abu, cv2.CV_64F, ksize=3)

    # Penajaman: original − Laplacian (tanda minus karena Laplacian bertanda negatif)
    sharp_lap = np.clip(abu.astype(np.float64) - lap, 0, 255).astype(np.uint8)

    # Laplacian dengan kernel 5×5 (mendeteksi area lebih luas)
    lap5   = cv2.Laplacian(abu, cv2.CV_64F, ksize=5)
    sharp5 = np.clip(abu.astype(np.float64) - lap5, 0, 255).astype(np.uint8)

    # Visualisasi Laplacian: geser ke 128 agar nilai negatif terlihat
    lap_vis  = np.clip(lap  + 128, 0, 255).astype(np.uint8)
    lap5_vis = np.clip(lap5 + 128, 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, img, jud in zip(axs[0],
            [abu, lap_vis, lap5_vis],
            ["Asli", "Laplacian ksize=3", "Laplacian ksize=5"]):
        ax.imshow(img, cmap='gray'); ax.set_title(jud); ax.axis('off')
    for ax, img, jud in zip(axs[1],
            [abu, sharp_lap, sharp5],
            ["Asli", "Sharp Laplacian k=3", "Sharp Laplacian k=5"]):
        ax.imshow(img, cmap='gray'); ax.set_title(jud); ax.axis('off')

    plt.suptitle("Demo 3 – Laplacian Sharpening", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_laplacian_sharpening.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 4: Perbandingan kekuatan sharpening (alpha) ────────────────────────
def demo_kekuatan_sharpening(img_rgb):
    """Menampilkan efek berbagai nilai alpha (0.5, 1.0, 2.0, 4.0) pada USM."""
    abu  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(abu, (11, 11), 0).astype(np.float32)
    mask = abu - blur    # detail frekuensi tinggi

    alpha_list = [0.5, 1.0, 2.0, 4.0]
    hasil_list = []
    for a in alpha_list:
        h = np.clip(abu + a * mask, 0, 255).astype(np.uint8)
        hasil_list.append(h)

    # Tampilkan gambar dan profil intensitas horizontal di baris tengah
    fig, axs = plt.subplots(2, len(alpha_list) + 1, figsize=(15, 7))
    axs[0, 0].imshow(abu.astype(np.uint8), cmap='gray')
    axs[0, 0].set_title("Asli"); axs[0, 0].axis('off')
    tengah = int(abu.shape[0] // 2)
    axs[1, 0].plot(abu[tengah, :], color='black'); axs[1, 0].set_title("Profil Asli")

    for i, (a, h) in enumerate(zip(alpha_list, hasil_list), start=1):
        axs[0, i].imshow(h, cmap='gray')
        axs[0, i].set_title(f"α={a}"); axs[0, i].axis('off')
        axs[1, i].plot(h[tengah, :], color='tomato')
        axs[1, i].set_title(f"Profil α={a}")
        axs[1, i].set_ylim(0, 255)

    plt.suptitle("Demo 4 – Perbandingan Kekuatan Sharpening (Alpha)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_kekuatan_sharpening.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Fungsi utama ─────────────────────────────────────────────────────────────
def main():
    print("=== Modul 03 | Sharpening dan Unsharp Masking ===")
    img = muat_atau_buat_gambar()
    print(f"Ukuran gambar: {img.shape}")

    demo_kernel_sharpening(img)
    demo_unsharp_masking(img)
    demo_laplacian_sharpening(img)
    demo_kekuatan_sharpening(img)

    print(f"Output disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
