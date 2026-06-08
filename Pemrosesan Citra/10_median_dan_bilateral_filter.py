"""
Modul 03 - Pemrosesan Citra (Image Processing)
Topik  : Median Filter dan Bilateral Filter
Tujuan : Memahami perbedaan filter berbasis urutan (median) dan filter yang
         mempertahankan tepi (edge-preserving) menggunakan bilateral filter.
         Membandingkan efektivitas masing-masing terhadap jenis noise berbeda.
Fungsi : cv2.medianBlur(), cv2.bilateralFilter(), pembuatan salt-pepper noise
         dan Gaussian noise secara manual.
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

    # Pola bertekstur: gradien latar + lingkaran + pola kotak
    kanvas = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(300):
        kanvas[i, :] = [int(i * 0.6), 100, 200 - int(i * 0.5)]
    cv2.circle(kanvas, (200, 150), 100, (255, 200, 50), -1)
    for x in range(0, 400, 40):
        cv2.line(kanvas, (x, 0), (x, 300), (80, 80, 80), 1)
    cv2.putText(kanvas, "FILTER NOISE", (80, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return kanvas


def tambah_salt_pepper_noise(img, kepadatan=0.05):
    """Menambahkan salt-and-pepper noise: piksel acak menjadi 0 atau 255."""
    noisy = img.copy()
    total = img.size
    n_salt   = int(total * kepadatan / 2)
    n_pepper = int(total * kepadatan / 2)

    # Koordinat salt (putih = 255)
    coords_s = [np.random.randint(0, d, n_salt) for d in img.shape]
    noisy[coords_s[0], coords_s[1]] = 255 if img.ndim == 2 else [255, 255, 255]

    # Koordinat pepper (hitam = 0)
    coords_p = [np.random.randint(0, d, n_pepper) for d in img.shape]
    noisy[coords_p[0], coords_p[1]] = 0
    return noisy.astype(np.uint8)


def tambah_gaussian_noise(img, std=25):
    """Menambahkan Gaussian noise dengan standar deviasi yang ditentukan."""
    noise  = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy  = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


# ── Demo 1: medianBlur pada salt-pepper noise ────────────────────────────────
def demo_median_pada_salt_pepper(img_rgb):
    """Menunjukkan efektivitas medianBlur terhadap salt-and-pepper noise."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    np.random.seed(42)
    noisy  = tambah_salt_pepper_noise(abu, kepadatan=0.08)
    median = cv2.medianBlur(noisy, 5)            # ksize harus bilangan ganjil
    gauss  = cv2.GaussianBlur(noisy, (5, 5), 0) # Gaussian sebagai pembanding

    # Hitung PSNR untuk membandingkan kualitas restorasi
    def psnr(ref, test):
        mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
        return 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, img, judul in zip(axs,
            [abu, noisy, median, gauss],
            ["Asli", "Salt-Pepper (8%)",
             f"medianBlur k=5\nPSNR={psnr(abu, median):.1f}dB",
             f"GaussianBlur k=5\nPSNR={psnr(abu, gauss):.1f}dB"]):
        ax.imshow(img, cmap='gray'); ax.set_title(judul); ax.axis('off')

    plt.suptitle("Demo 1 – medianBlur pada Salt-Pepper Noise", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_median_salt_pepper.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 2: Perbandingan ukuran kernel medianBlur ────────────────────────────
def demo_perbandingan_ukuran_median(img_rgb):
    """Memperlihatkan perubahan hasil medianBlur untuk ukuran jendela 3, 5, 9, 15."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    np.random.seed(0)
    noisy = tambah_salt_pepper_noise(abu, kepadatan=0.10)

    ukuran = [3, 5, 9, 15]
    fig, axs = plt.subplots(1, len(ukuran) + 1, figsize=(15, 4))
    axs[0].imshow(noisy, cmap='gray'); axs[0].set_title("Noisy 10%"); axs[0].axis('off')

    for ax, k in zip(axs[1:], ukuran):
        hasil = cv2.medianBlur(noisy, k)
        ax.imshow(hasil, cmap='gray')
        ax.set_title(f"Median k={k}")
        ax.axis('off')

    plt.suptitle("Demo 2 – Perbandingan Ukuran Kernel medianBlur", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_perbandingan_ukuran_median.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 3: bilateralFilter – edge-preserving ───────────────────────────────
def demo_bilateral_filter(img_rgb):
    """Mendemonstrasikan bilateralFilter: menghaluskan area datar tapi mempertahankan tepi."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    np.random.seed(1)
    noisy = tambah_gaussian_noise(abu, std=20)

    # d=9: diameter jendela; sigmaColor: rentang warna; sigmaSpace: lebar spasial
    bilateral_ketat  = cv2.bilateralFilter(noisy, d=9, sigmaColor=25,  sigmaSpace=25)
    bilateral_sedang = cv2.bilateralFilter(noisy, d=9, sigmaColor=75,  sigmaSpace=75)
    bilateral_kuat   = cv2.bilateralFilter(noisy, d=9, sigmaColor=150, sigmaSpace=150)

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, img, judul in zip(axs,
            [noisy, bilateral_ketat, bilateral_sedang, bilateral_kuat],
            ["Noisy (Gaussian σ=20)",
             "Bilateral σC=25",
             "Bilateral σC=75",
             "Bilateral σC=150"]):
        ax.imshow(img, cmap='gray'); ax.set_title(judul); ax.axis('off')

    plt.suptitle("Demo 3 – bilateralFilter: Edge-Preserving", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_bilateral_filter.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 4: Gaussian vs Median vs Bilateral pada noise campuran ──────────────
def demo_perbandingan_tiga_filter(img_rgb):
    """Membandingkan ketiga filter pada gambar dengan dua jenis noise sekaligus."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    np.random.seed(7)

    # Tambahkan salt-pepper dan Gaussian noise secara berurutan
    noisy = tambah_salt_pepper_noise(abu, kepadatan=0.05)
    noisy = tambah_gaussian_noise(noisy, std=15)

    hasil_gauss    = cv2.GaussianBlur(noisy, (9, 9), 2)
    hasil_median   = cv2.medianBlur(noisy, 5)
    hasil_bilateral = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)

    # Komputasi selisih terhadap gambar asli (error map)
    def err_map(ref, test):
        return np.abs(ref.astype(int) - test.astype(int)).astype(np.uint8)

    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    judul_atas = ["Asli", "Noisy", "Gaussian k=9", "Median k=5", "Bilateral"]
    imgs_atas  = [abu, noisy, hasil_gauss, hasil_median, hasil_bilateral]

    # Baris atas: gambar hasil filter
    for i, (img, jud) in enumerate(zip(imgs_atas[:4], judul_atas[:4])):
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].set_title(jud); axs[0, i].axis('off')

    # Baris bawah: peta error (semakin gelap = semakin baik)
    axs[1, 0].imshow(abu, cmap='gray'); axs[1, 0].set_title("Referensi"); axs[1, 0].axis('off')
    for i, (hasil, label) in enumerate(zip(
            [hasil_gauss, hasil_median, hasil_bilateral],
            ["Error Gaussian", "Error Median", "Error Bilateral"]), start=1):
        axs[1, i].imshow(err_map(abu, hasil), cmap='hot')
        axs[1, i].set_title(label); axs[1, i].axis('off')

    plt.suptitle("Demo 4 – Perbandingan Gaussian vs Median vs Bilateral", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_perbandingan_tiga_filter.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Fungsi utama ─────────────────────────────────────────────────────────────
def main():
    print("=== Modul 03 | Median dan Bilateral Filter ===")
    img = muat_atau_buat_gambar()
    print(f"Ukuran gambar: {img.shape}")

    demo_median_pada_salt_pepper(img)
    demo_perbandingan_ukuran_median(img)
    demo_bilateral_filter(img)
    demo_perbandingan_tiga_filter(img)

    print(f"Output disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
