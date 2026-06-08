"""
Modul 03 - Pemrosesan Citra (Image Processing)
Topik  : Gaussian Blur
Tujuan : Memahami filter Gaussian dan distribusi Gaussian 2D, serta
         membandingkannya dengan blur kotak (box filter). Mempelajari
         pengaruh ukuran kernel dan nilai sigma terhadap tingkat kehalusan,
         serta implementasi separable Gaussian untuk efisiensi komputasi.
Fungsi : cv2.GaussianBlur(), cv2.blur(), cv2.boxFilter(),
         cv2.getGaussianKernel(), cv2.sepFilter2D()
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

    # Buat pola kotak-kotak tajam agar efek blur mudah terlihat
    kanvas = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(6):
        for j in range(8):
            warna = (np.random.randint(50, 255),
                     np.random.randint(50, 255),
                     np.random.randint(50, 255))
            cv2.rectangle(kanvas, (j*50, i*50), (j*50+49, i*50+49), warna, -1)
    # Tambahkan tepi tajam di tengah untuk visualisasi blur
    cv2.line(kanvas, (200, 0), (200, 300), (255, 255, 255), 3)
    cv2.putText(kanvas, "GAUSSIAN BLUR", (60, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return kanvas


# ── Demo 1: GaussianBlur vs blur vs boxFilter ────────────────────────────────
def demo_perbandingan_metode_blur(img_rgb):
    """Membandingkan tiga metode blur OpenCV pada ukuran kernel yang sama (15×15)."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    k = 15          # ukuran kernel yang sama untuk perbandingan adil

    # GaussianBlur: bobot mengikuti distribusi normal (Gaussian)
    g_blur  = cv2.GaussianBlur(abu, (k, k), 0)
    # blur: rata-rata sederhana semua piksel dalam jendela (box filter)
    b_blur  = cv2.blur(abu, (k, k))
    # boxFilter: mirip blur, tapi bisa dinormalisasi atau tidak
    bx_blur = cv2.boxFilter(abu, -1, (k, k), normalize=True)

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    gambar_dan_judul = [
        (abu,     "Asli"),
        (g_blur,  f"GaussianBlur k={k}"),
        (b_blur,  f"blur (rata-rata) k={k}"),
        (bx_blur, f"boxFilter k={k}"),
    ]
    for ax, (img, judul) in zip(axs, gambar_dan_judul):
        ax.imshow(img, cmap='gray'); ax.set_title(judul); ax.axis('off')

    plt.suptitle("Demo 1 – GaussianBlur vs blur vs boxFilter", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_perbandingan_metode_blur.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 2: Pengaruh ukuran kernel Gaussian ──────────────────────────────────
def demo_pengaruh_ukuran_kernel(img_rgb):
    """Menunjukkan perbedaan visual dan profil intensitas untuk ukuran kernel berbeda."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ukuran = [3, 7, 15, 31]

    fig, axs = plt.subplots(2, len(ukuran) + 1, figsize=(15, 7))

    # Baris atas: gambar hasil blur
    axs[0, 0].imshow(abu, cmap='gray'); axs[0, 0].set_title("Asli"); axs[0, 0].axis('off')
    for i, k in enumerate(ukuran, start=1):
        h = cv2.GaussianBlur(abu, (k, k), 0)
        axs[0, i].imshow(h, cmap='gray')
        axs[0, i].set_title(f"k={k}"); axs[0, i].axis('off')

    # Baris bawah: profil horizontal intensitas piksel di baris tengah
    tengah = abu.shape[0] // 2
    axs[1, 0].plot(abu[tengah, :], color='black')
    axs[1, 0].set_title("Profil Asli"); axs[1, 0].set_ylim(0, 255)
    for i, k in enumerate(ukuran, start=1):
        h = cv2.GaussianBlur(abu, (k, k), 0)
        axs[1, i].plot(h[tengah, :], color='steelblue')
        axs[1, i].set_title(f"Profil k={k}"); axs[1, i].set_ylim(0, 255)

    plt.suptitle("Demo 2 – Pengaruh Ukuran Kernel Gaussian", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_pengaruh_ukuran_kernel.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 3: Pengaruh nilai sigma ────────────────────────────────────────────
def demo_pengaruh_sigma(img_rgb):
    """Menampilkan bagaimana sigma mengontrol 'keluasan' distribusi Gaussian."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sigma_list = [0.5, 1.0, 2.0, 5.0]
    k = 31   # kernel cukup besar agar sigma rendah maupun tinggi terlihat

    fig, axs = plt.subplots(2, len(sigma_list) + 1, figsize=(15, 7))

    # Baris atas: gambar hasil blur berbagai sigma
    axs[0, 0].imshow(abu, cmap='gray'); axs[0, 0].set_title("Asli"); axs[0, 0].axis('off')
    for i, s in enumerate(sigma_list, start=1):
        h = cv2.GaussianBlur(abu, (k, k), s)
        axs[0, i].imshow(h, cmap='gray')
        axs[0, i].set_title(f"σ={s}, k={k}"); axs[0, i].axis('off')

    # Baris bawah: bentuk kernel 1D Gaussian untuk setiap sigma
    x = np.linspace(-k // 2, k // 2, k)
    axs[1, 0].axis('off')
    for i, s in enumerate(sigma_list, start=1):
        # Hitung distribusi Gaussian 1D secara manual
        g1d = np.exp(-x**2 / (2 * s**2))
        g1d /= g1d.sum()                             # normalisasi agar jumlah = 1
        axs[1, i].plot(x, g1d, color='tomato')
        axs[1, i].set_title(f"Kernel 1D σ={s}")
        axs[1, i].set_xlabel("Posisi piksel")

    plt.suptitle("Demo 3 – Pengaruh Nilai Sigma pada Gaussian Blur", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_pengaruh_sigma.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 4: Separable Gaussian (1D × 1D = 2D) ───────────────────────────────
def demo_separable_gaussian(img_rgb):
    """Mendemonstrasikan sifat separabilitas: filter 1D berurutan ≡ filter 2D."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    k, sigma = 15, 3.0

    # Dapatkan kernel 1D dari OpenCV
    kernel_1d = cv2.getGaussianKernel(k, sigma)   # bentuk (k, 1)
    kernel_2d = kernel_1d @ kernel_1d.T            # outer product → kernel 2D

    # sepFilter2D: terapkan filter baris dulu, lalu kolom (efisiensi O(k) vs O(k²))
    hasil_sep = cv2.sepFilter2D(abu, -1, kernel_1d, kernel_1d)
    # GaussianBlur biasa sebagai referensi
    hasil_std = cv2.GaussianBlur(abu, (k, k), sigma)
    # Selisih absolut untuk verifikasi kesamaan
    selisih   = np.abs(hasil_sep.astype(int) - hasil_std.astype(int)).astype(np.uint8)

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, img, judul in zip(axs,
            [kernel_2d, hasil_std, hasil_sep, selisih],
            ["Kernel 2D (heatmap)", f"GaussianBlur k={k}", f"sepFilter2D k={k}", "Selisih Absolut"]):
        if judul.startswith("Kernel"):
            ax.imshow(img, cmap='hot', interpolation='nearest')
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(judul); ax.axis('off')

    plt.suptitle("Demo 4 – Separable Gaussian: 1D × 1D = 2D", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_separable_gaussian.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # Cetak ringkasan numeris
    print(f"  Kernel 1D ukuran : {kernel_1d.shape}")
    print(f"  Kernel 2D ukuran : {kernel_2d.shape}")
    print(f"  Max selisih |sep - std| : {selisih.max()} piksel")


# ── Fungsi utama ─────────────────────────────────────────────────────────────
def main():
    print("=== Modul 03 | Gaussian Blur ===")
    img = muat_atau_buat_gambar()
    print(f"Ukuran gambar: {img.shape}")

    demo_perbandingan_metode_blur(img)
    demo_pengaruh_ukuran_kernel(img)
    demo_pengaruh_sigma(img)
    demo_separable_gaussian(img)

    print(f"Output disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
