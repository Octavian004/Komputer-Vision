"""
Modul 03 - Pemrosesan Citra (Image Processing)
Topik  : Deteksi Tepi Sobel
Tujuan : Memahami cara mendeteksi tepi gambar menggunakan filter Sobel
         (turunan pertama) dalam arah X dan Y, menghitung magnitude dan
         arah gradien, serta membandingkan Sobel dengan Scharr dan berbagai
         ukuran kernel. Visualisasi menggunakan heatmap dan quiver plot.
Fungsi : cv2.Sobel(src, cv2.CV_64F, dx, dy, ksize),
         cv2.Scharr(), np.arctan2(), np.hypot()
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

    # Gambar sintetis: bentuk geometris dengan tepi tajam untuk uji Sobel
    kanvas = np.zeros((300, 400, 3), dtype=np.uint8)
    # Latar gradien abu-abu
    for i in range(300):
        v = int(i * 0.5)
        kanvas[i, :] = [v, v, v]
    cv2.rectangle(kanvas, (40,  40), (160, 130), (220,  80,  80), -1)
    cv2.circle(kanvas, (280, 160), 90, (80, 200, 220), -1)
    cv2.line(kanvas, (0, 200), (400, 200), (255, 255, 100), 3)
    cv2.ellipse(kanvas, (200, 240), (70, 40), 30, 0, 360, (150, 255, 150), -1)
    cv2.putText(kanvas, "SOBEL EDGE", (85, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return kanvas


# ── Demo 1: Sobel X dan Sobel Y ─────────────────────────────────────────────
def demo_sobel_x_dan_y(img_rgb):
    """Menghitung gradien horizontal (Gx) dan vertikal (Gy) menggunakan cv2.Sobel()."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Sobel arah X: mendeteksi tepi vertikal
    sobel_x = cv2.Sobel(abu, cv2.CV_64F, 1, 0, ksize=3)
    # Sobel arah Y: mendeteksi tepi horizontal
    sobel_y = cv2.Sobel(abu, cv2.CV_64F, 0, 1, ksize=3)

    # Normalisasi untuk visualisasi (geser ke 128 agar tanda negatif tampak)
    def norm_vis(g):
        return np.clip(np.abs(g), 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].imshow(abu, cmap='gray');         axs[0].set_title("Asli (Grayscale)")
    axs[1].imshow(norm_vis(sobel_x), cmap='gray'); axs[1].set_title("Sobel Gx (tepi vertikal)")
    axs[2].imshow(norm_vis(sobel_y), cmap='gray'); axs[2].set_title("Sobel Gy (tepi horizontal)")
    for ax in axs: ax.axis('off')

    # Tunjukkan kernel Sobel 3×3 untuk referensi
    print("Kernel Sobel Gx:\n", cv2.Sobel(np.eye(3), cv2.CV_64F, 1, 0, ksize=3))
    print("Kernel Sobel Gy:\n", cv2.Sobel(np.eye(3), cv2.CV_64F, 0, 1, ksize=3))

    plt.suptitle("Demo 1 – Sobel Gx dan Gy", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_sobel_x_dan_y.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 2: Magnitude dan Arah Gradien ─────────────────────────────────────
def demo_magnitude_dan_arah(img_rgb):
    """Menghitung dan memvisualisasikan magnitude dan arah gradien sebagai heatmap."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(abu, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(abu, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude: M = √(Gx² + Gy²)
    magnitude = np.hypot(gx, gy)
    # Arah gradien: θ = arctan(Gy / Gx), dalam derajat
    arah = np.degrees(np.arctan2(gy, gx))

    # Normalisasi ke [0, 1] untuk colormap
    mag_norm  = magnitude / magnitude.max()
    arah_norm = (arah + 180) / 360.0    # geser dari [-180,180] ke [0,1]

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    axs[0].imshow(abu, cmap='gray');         axs[0].set_title("Asli")
    im1 = axs[1].imshow(mag_norm, cmap='hot');  axs[1].set_title("Magnitude (heatmap)")
    im2 = axs[2].imshow(arah_norm, cmap='hsv'); axs[2].set_title("Arah Gradien °(HSV)")
    plt.colorbar(im1, ax=axs[1], fraction=0.04)
    plt.colorbar(im2, ax=axs[2], fraction=0.04)

    # Quiver plot: anak panah gradien pada sampel grid 20×20
    h, w = abu.shape
    step = 15
    Y, X = np.mgrid[step//2:h:step, step//2:w:step]
    U = gx[Y, X]; V = gy[Y, X]
    axs[3].imshow(mag_norm, cmap='gray')
    axs[3].quiver(X, Y, U, -V, magnitude[Y, X],
                  cmap='plasma', scale=2000, width=0.003)
    axs[3].set_title("Quiver (arah & magnitudo)")
    for ax in axs: ax.axis('off')

    plt.suptitle("Demo 2 – Magnitude dan Arah Gradien Sobel", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_magnitude_dan_arah.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 3: Scharr filter sebagai alternatif ksize=3 ────────────────────────
def demo_scharr_vs_sobel(img_rgb):
    """Membandingkan Scharr dengan Sobel ksize=3 untuk akurasi rotasional lebih baik."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Sobel ksize=3: koefisien [1,2,1; 0,0,0; -1,-2,-1]
    sobel_gx = cv2.Sobel(abu, cv2.CV_64F, 1, 0, ksize=3)
    sobel_gy = cv2.Sobel(abu, cv2.CV_64F, 0, 1, ksize=3)
    mag_sobel = np.hypot(sobel_gx, sobel_gy)

    # Scharr: koefisien [-3,0,3; -10,0,10; -3,0,3] – akurasi isotropik lebih baik
    scharr_gx = cv2.Scharr(abu, cv2.CV_64F, 1, 0)
    scharr_gy = cv2.Scharr(abu, cv2.CV_64F, 0, 1)
    mag_scharr = np.hypot(scharr_gx, scharr_gy)

    # Normalisasi ke 0-255 untuk tampilan
    def norm255(m):
        return (255 * m / m.max()).astype(np.uint8)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, img, jud in zip(axs[0],
            [abu, norm255(np.abs(sobel_gx)), norm255(mag_sobel)],
            ["Asli", "Sobel Gx k=3", "Magnitude Sobel k=3"]):
        ax.imshow(img, cmap='gray'); ax.set_title(jud); ax.axis('off')
    for ax, img, jud in zip(axs[1],
            [abu, norm255(np.abs(scharr_gx)), norm255(mag_scharr)],
            ["Asli", "Scharr Gx", "Magnitude Scharr"]):
        ax.imshow(img, cmap='gray'); ax.set_title(jud); ax.axis('off')

    plt.suptitle("Demo 3 – Scharr vs Sobel ksize=3", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_scharr_vs_sobel.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 4: Perbandingan ksize Sobel (3, 5, 7) ──────────────────────────────
def demo_perbandingan_ksize_sobel(img_rgb):
    """Menunjukkan pengaruh ukuran kernel Sobel terhadap ketebalan dan detail tepi."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ksize_list = [1, 3, 5, 7]   # ksize=1 → filter Prewitt sederhana

    hasil_mag = []
    for k in ksize_list:
        gx = cv2.Sobel(abu, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(abu, cv2.CV_64F, 0, 1, ksize=k)
        mag = np.hypot(gx, gy)
        mag_norm = (255 * mag / mag.max()).astype(np.uint8)
        hasil_mag.append(mag_norm)

    # Baris atas: magnitude grayscale; baris bawah: magnitude heatmap
    fig, axs = plt.subplots(2, len(ksize_list), figsize=(15, 7))
    for i, (k, m) in enumerate(zip(ksize_list, hasil_mag)):
        axs[0, i].imshow(m, cmap='gray')
        axs[0, i].set_title(f"Magnitude ksize={k}"); axs[0, i].axis('off')
        axs[1, i].imshow(m, cmap='inferno')
        axs[1, i].set_title(f"Heatmap ksize={k}");  axs[1, i].axis('off')

    # Profil intensitas horizontal di baris tengah pada semua ksize
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    tengah = abu.shape[0] // 2
    warna = ['black', 'steelblue', 'tomato', 'seagreen']
    for k, m, c in zip(ksize_list, hasil_mag, warna):
        ax2.plot(m[tengah, :], color=c, label=f"ksize={k}", linewidth=1.5)
    ax2.set_xlabel("Posisi piksel horizontal"); ax2.set_ylabel("Magnitude")
    ax2.set_title("Profil Magnitude di Baris Tengah")
    ax2.legend(); ax2.set_ylim(0, 255)

    plt.tight_layout()
    for f, nama in [(fig, "12_perbandingan_ksize_sobel.png"),
                    (fig2, "12_profil_magnitude_ksize.png")]:
        f.savefig(os.path.join(OUTPUT_DIR, nama), dpi=150, bbox_inches="tight")
    plt.show()


# ── Fungsi utama ─────────────────────────────────────────────────────────────
def main():
    print("=== Modul 03 | Deteksi Tepi Sobel ===")
    img = muat_atau_buat_gambar()
    print(f"Ukuran gambar: {img.shape}")

    demo_sobel_x_dan_y(img)
    demo_magnitude_dan_arah(img)
    demo_scharr_vs_sobel(img)
    demo_perbandingan_ksize_sobel(img)

    print(f"Output disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
