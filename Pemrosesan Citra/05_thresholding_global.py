"""
=============================================================
Modul 03 - Pemrosesan Citra: Thresholding Global
=============================================================
Thresholding global mengkonversi gambar grayscale menjadi gambar
biner menggunakan nilai ambang batas (threshold) yang sama untuk
seluruh gambar. Teknik ini efektif untuk gambar dengan pencahayaan
yang seragam.

Topik:
  1. Manual threshold dengan berbagai nilai (cv2.threshold)
  2. Perbandingan jenis threshold: BINARY, BINARY_INV, TRUNC,
     TOZERO, TOZERO_INV
  3. Otsu's threshold - deteksi threshold otomatis
  4. Pengaruh noise terhadap hasil thresholding

Referensi: cv2.threshold, cv2.THRESH_OTSU
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


def muat_atau_buat_gambar():
    """Muat gambar grayscale; buat gambar sintetis jika tidak ditemukan."""
    if os.path.exists(IMAGE_PATH):
        img  = cv2.imread(IMAGE_PATH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"[INFO] Gambar dimuat dari: {IMAGE_PATH}")
        return gray

    # --- Buat gambar sintetis dengan dua puncak distribusi (bimodal) ---
    # Gambar bimodal ideal untuk demonstrasi Otsu threshold
    print("[INFO] Gambar tidak ditemukan. Membuat gambar sintetis.")
    h, w = 400, 600
    gray = np.zeros((h, w), dtype=np.uint8)
    # Latar belakang terang
    gray[:] = 200
    # Objek gelap di atas latar belakang terang
    gray[50:150,  50:200]  = 60
    gray[200:350, 150:400] = 80
    gray[80:250,  380:560] = 50
    gray[300:380, 400:580] = 70
    # Tambahkan gradasi halus di tepi
    for i in range(10):
        val = int(200 - (200 - 60) * i / 10)
        if 50 - i >= 0:
            gray[50 - i, 50:200] = val
    return gray


def demo_threshold_manual(gray):
    """Demo 1: Pengaruh nilai threshold manual pada hasil binerisasi."""
    # Uji beberapa nilai threshold untuk melihat pengaruhnya
    nilai_thresh = [50, 100, 127, 150, 200]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Demo 1: Manual Threshold dengan Berbagai Nilai",
                 fontsize=14, fontweight="bold")

    # Panel pertama: gambar asli dan histogramnya
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Gambar Grayscale Asli")
    axes[0, 0].axis("off")

    for i, thresh_val in enumerate(nilai_thresh):
        baris = (i + 1) // 3
        kol   = (i + 1) % 3
        # cv2.threshold: THRESH_BINARY - piksel >= thresh -> 255, lainnya -> 0
        ret, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        axes[baris, kol].imshow(binary, cmap="gray")
        axes[baris, kol].set_title(f"Threshold = {thresh_val}")
        axes[baris, kol].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_01_threshold_manual.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 1: manual threshold.")


def demo_jenis_threshold(gray):
    """Demo 2: Perbandingan berbagai jenis (tipe) threshold OpenCV."""
    # Nilai threshold tunggal untuk membandingkan semua jenis
    thresh_val = 127

    # Definisi semua jenis threshold yang tersedia
    jenis = [
        ("THRESH_BINARY",     cv2.THRESH_BINARY,     "piksel >= T -> 255, lainnya -> 0"),
        ("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV, "piksel >= T -> 0, lainnya -> 255"),
        ("THRESH_TRUNC",      cv2.THRESH_TRUNC,      "piksel >= T -> T, lainnya tetap"),
        ("THRESH_TOZERO",     cv2.THRESH_TOZERO,     "piksel >= T tetap, lainnya -> 0"),
        ("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV, "piksel >= T -> 0, lainnya tetap"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Demo 2: Berbagai Jenis Threshold (nilai = {thresh_val})",
                 fontsize=14, fontweight="bold")

    # Panel pertama: gambar asli
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Gambar Asli")
    axes[0, 0].axis("off")

    for i, (nama, tp, ket) in enumerate(jenis):
        baris = (i + 1) // 3
        kol   = (i + 1) % 3
        _, hasil = cv2.threshold(gray, thresh_val, 255, tp)
        axes[baris, kol].imshow(hasil, cmap="gray")
        axes[baris, kol].set_title(f"{nama}\n{ket}", fontsize=8)
        axes[baris, kol].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_02_jenis_threshold.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 2: jenis-jenis threshold.")


def demo_otsu_threshold(gray):
    """Demo 3: Otsu's thresholding - pemilihan threshold optimal otomatis."""
    # Otsu mencari nilai threshold yang meminimalkan varians intra-kelas
    # Sangat efektif untuk gambar dengan distribusi bimodal

    # Threshold manual untuk perbandingan
    _, biner_manual = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Otsu: threshold dipilih otomatis berdasarkan histogram
    ret_otsu, biner_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Hitung histogram untuk menampilkan posisi threshold Otsu
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Demo 3: Otsu Threshold (nilai otomatis = {ret_otsu:.0f})",
                 fontsize=14, fontweight="bold")

    # Histogram dengan garis threshold
    axes[0].plot(hist, color="steelblue")
    axes[0].axvline(x=127,      color="red",   linestyle="--", label="Manual (127)")
    axes[0].axvline(x=ret_otsu, color="green", linestyle="-",  label=f"Otsu ({ret_otsu:.0f})")
    axes[0].set_title("Histogram & Posisi Threshold")
    axes[0].set_xlim([0, 256])
    axes[0].set_xlabel("Intensitas")
    axes[0].set_ylabel("Frekuensi")
    axes[0].legend()

    # Hasil threshold manual
    axes[1].imshow(biner_manual, cmap="gray")
    axes[1].set_title("Threshold Manual (127)")
    axes[1].axis("off")

    # Hasil Otsu
    axes[2].imshow(biner_otsu, cmap="gray")
    axes[2].set_title(f"Threshold Otsu ({ret_otsu:.0f})\nDeteksi Otomatis")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_03_otsu_threshold.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Nilai threshold Otsu yang terdeteksi: {ret_otsu:.0f}")
    print("[SELESAI] Demo 3: Otsu threshold.")


def demo_pengaruh_noise(gray):
    """Demo 4: Pengaruh noise terhadap hasil global thresholding."""
    # Simulasikan noise dalam gambar
    noise_ringan = np.random.normal(0, 10, gray.shape).astype(np.int16)
    noise_sedang = np.random.normal(0, 30, gray.shape).astype(np.int16)
    noise_berat  = np.random.normal(0, 60, gray.shape).astype(np.int16)

    gray_noise_ringan = np.clip(gray.astype(np.int16) + noise_ringan, 0, 255).astype(np.uint8)
    gray_noise_sedang = np.clip(gray.astype(np.int16) + noise_sedang, 0, 255).astype(np.uint8)
    gray_noise_berat  = np.clip(gray.astype(np.int16) + noise_berat,  0, 255).astype(np.uint8)

    variasi = [
        ("Tanpa Noise",    gray),
        ("Noise Ringan (σ=10)", gray_noise_ringan),
        ("Noise Sedang (σ=30)", gray_noise_sedang),
        ("Noise Berat  (σ=60)", gray_noise_berat),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Demo 4: Pengaruh Noise pada Global Thresholding (Otsu)",
                 fontsize=14, fontweight="bold")

    for kol, (nama, img_var) in enumerate(variasi):
        # Terapkan Otsu threshold pada setiap variasi noise
        ret, hasil = cv2.threshold(img_var, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Baris atas: gambar dengan noise
        axes[0, kol].imshow(img_var, cmap="gray")
        axes[0, kol].set_title(f"{nama}", fontsize=9)
        axes[0, kol].axis("off")

        # Baris bawah: hasil threshold
        axes[1, kol].imshow(hasil, cmap="gray")
        axes[1, kol].set_title(f"Otsu = {ret:.0f}", fontsize=9)
        axes[1, kol].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_04_pengaruh_noise.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 4: pengaruh noise.")


def main():
    """Fungsi utama: jalankan semua demo thresholding global."""
    print("=" * 60)
    print("  MODUL 03 - THRESHOLDING GLOBAL")
    print("=" * 60)

    # Muat gambar atau buat sintetis jika tidak tersedia
    gray = muat_atau_buat_gambar()

    # Jalankan empat demo secara berurutan
    demo_threshold_manual(gray)
    demo_jenis_threshold(gray)
    demo_otsu_threshold(gray)
    demo_pengaruh_noise(gray)

    print(f"\n[SELESAI] Semua output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
