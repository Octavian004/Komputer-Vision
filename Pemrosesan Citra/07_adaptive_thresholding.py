"""
=============================================================
Modul 03 - Pemrosesan Citra: Adaptive Thresholding
=============================================================
Adaptive thresholding menghitung nilai threshold secara lokal
untuk setiap piksel berdasarkan sekitarnya (neighborhood). Metode
ini jauh lebih efektif daripada global thresholding untuk gambar
dengan pencahayaan tidak merata (non-uniform illumination).

Topik:
  1. ADAPTIVE_THRESH_MEAN_C vs ADAPTIVE_THRESH_GAUSSIAN_C
  2. Variasi blockSize (11, 21, 51)
  3. Variasi konstanta C
  4. Perbandingan global vs adaptive pada gambar pencahayaan tidak merata

Referensi: cv2.adaptiveThreshold
=============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Konfigurasi direktori ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_PATH = os.path.join(IMAGE_DIR, "kota.jpg")


def buat_gambar_gradien_pencahayaan(h=400, w=600):
    """Buat gambar sintetis dengan gradien pencahayaan tidak merata."""
    # Buat pola teks/objek yang seragam di seluruh gambar
    pola = np.zeros((h, w), dtype=np.uint8)
    # Objek-objek gelap yang teratur (simulasi teks atau tekstur)
    for r in range(30, h - 30, 40):
        for c in range(30, w - 30, 50):
            pola[r:r+20, c:c+30] = 80

    # Buat gradien pencahayaan tidak merata (lebih terang di kiri, gelap di kanan)
    gradien = np.linspace(220, 40, w, dtype=np.float32)
    gradien = np.tile(gradien, (h, 1))

    # Tambahkan gradien vertikal (lebih terang di atas)
    gradien_v = np.linspace(1.2, 0.7, h, dtype=np.float32).reshape(h, 1)
    gradien   = np.clip(gradien * gradien_v, 0, 255).astype(np.uint8)

    # Gabungkan pola dengan latar gradien
    gambar = np.where(pola > 0, pola, gradien).astype(np.uint8)
    return gambar


def muat_atau_buat_gambar():
    """Muat gambar dari IMAGE_DIR; buat gambar sintetis jika tidak ditemukan."""
    if os.path.exists(IMAGE_PATH):
        img  = cv2.imread(IMAGE_PATH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"[INFO] Gambar dimuat dari: {IMAGE_PATH}")
        return gray

    # --- Buat gambar sintetis dengan pencahayaan tidak merata ---
    print("[INFO] Gambar tidak ditemukan. Membuat gambar sintetis.")
    return buat_gambar_gradien_pencahayaan()


def demo_mean_vs_gaussian(gray):
    """Demo 1: Perbandingan ADAPTIVE_THRESH_MEAN_C vs ADAPTIVE_THRESH_GAUSSIAN_C."""
    # Parameter adaptive threshold:
    # - blockSize: ukuran neighborhood (harus ganjil, >= 3)
    # - C: konstanta yang dikurangkan dari rata-rata/Gaussian
    block = 11
    C     = 2

    # Metode 1: MEAN - threshold = rata-rata piksel di neighborhood - C
    hasil_mean = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block, C
    )

    # Metode 2: GAUSSIAN - threshold = rata-rata terbobot Gaussian neighborhood - C
    # Memberikan bobot lebih pada piksel dekat dengan pusat
    hasil_gauss = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, C
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Demo 1: Mean vs Gaussian Adaptive Threshold (blockSize={block}, C={C})",
                 fontsize=14, fontweight="bold")

    axes[0].imshow(gray,         cmap="gray"); axes[0].set_title("Gambar Asli");              axes[0].axis("off")
    axes[1].imshow(hasil_mean,   cmap="gray"); axes[1].set_title("ADAPTIVE_THRESH_MEAN_C");   axes[1].axis("off")
    axes[2].imshow(hasil_gauss,  cmap="gray"); axes[2].set_title("ADAPTIVE_THRESH_GAUSSIAN_C"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_01_mean_vs_gaussian.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 1: Mean vs Gaussian adaptive threshold.")


def demo_variasi_blocksize(gray):
    """Demo 2: Pengaruh variasi blockSize terhadap hasil adaptive threshold."""
    # blockSize menentukan seberapa besar area lokal yang digunakan:
    # - blockSize kecil: detail lokal tinggi, sensitif noise
    # - blockSize besar: lebih smooth, detail lokal berkurang
    block_sizes = [7, 11, 21, 31, 51, 101]
    C = 3

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Demo 2: Variasi blockSize pada Adaptive Threshold (Gaussian, C=3)",
                 fontsize=14, fontweight="bold")

    for i, bs in enumerate(block_sizes):
        baris, kol = divmod(i, 3)
        # blockSize harus ganjil
        hasil = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs, C
        )
        axes[baris, kol].imshow(hasil, cmap="gray")
        axes[baris, kol].set_title(f"blockSize = {bs}")
        axes[baris, kol].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_02_variasi_blocksize.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 2: variasi blockSize.")


def demo_variasi_konstanta_c(gray):
    """Demo 3: Pengaruh variasi konstanta C terhadap hasil adaptive threshold."""
    # Konstanta C dikurangkan dari threshold lokal:
    # - C besar: threshold turun, lebih banyak piksel jadi putih (lebih terang)
    # - C kecil/negatif: threshold naik, lebih banyak piksel jadi hitam
    nilai_c = [-5, 0, 2, 5, 10, 20]
    block   = 15

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Demo 3: Variasi Konstanta C (Gaussian, blockSize=15)",
                 fontsize=14, fontweight="bold")

    for i, c_val in enumerate(nilai_c):
        baris, kol = divmod(i, 3)
        hasil = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, c_val
        )
        axes[baris, kol].imshow(hasil, cmap="gray")
        axes[baris, kol].set_title(f"C = {c_val}")
        axes[baris, kol].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_03_variasi_c.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 3: variasi konstanta C.")


def demo_global_vs_adaptive_gradien(gray):
    """Demo 4: Perbandingan global vs adaptive pada gambar pencahayaan tidak merata."""
    # Buat gambar dengan gradien pencahayaan tidak merata
    # (gunakan sintetis agar perbedaannya jelas terlihat)
    gambar_gradien = buat_gambar_gradien_pencahayaan()

    # Metode 1: Global threshold (Otsu) - gagal pada pencahayaan tidak merata
    ret_otsu, hasil_global = cv2.threshold(
        gambar_gradien, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Metode 2: Adaptive threshold Mean - adaptif per region
    hasil_mean = cv2.adaptiveThreshold(
        gambar_gradien, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        21, 5
    )

    # Metode 3: Adaptive threshold Gaussian - lebih halus
    hasil_gauss = cv2.adaptiveThreshold(
        gambar_gradien, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 5
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Demo 4: Global vs Adaptive Threshold - Pencahayaan Tidak Merata",
                 fontsize=14, fontweight="bold")

    # Tampilkan gambar dengan pencahayaan gradien dan tiga hasil threshold
    axes[0, 0].imshow(gambar_gradien, cmap="gray")
    axes[0, 0].set_title("Gambar Gradien Pencahayaan\n(Kiri Terang -> Kanan Gelap)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(hasil_global, cmap="gray")
    axes[0, 1].set_title(f"Global Otsu (thresh={ret_otsu:.0f})\nGagal di Area Campuran")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(hasil_mean, cmap="gray")
    axes[1, 0].set_title("Adaptive Mean (block=21, C=5)\nAdaptif per Region")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hasil_gauss, cmap="gray")
    axes[1, 1].set_title("Adaptive Gaussian (block=21, C=5)\nHasil Paling Halus")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_04_global_vs_adaptive.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 4: global vs adaptive pada pencahayaan tidak merata.")


def main():
    """Fungsi utama: jalankan semua demo adaptive thresholding."""
    print("=" * 60)
    print("  MODUL 03 - ADAPTIVE THRESHOLDING")
    print("=" * 60)

    # Muat gambar atau buat sintetis jika tidak tersedia
    gray = muat_atau_buat_gambar()

    # Jalankan empat demo secara berurutan
    demo_mean_vs_gaussian(gray)
    demo_variasi_blocksize(gray)
    demo_variasi_konstanta_c(gray)
    demo_global_vs_adaptive_gradien(gray)

    print(f"\n[SELESAI] Semua output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
