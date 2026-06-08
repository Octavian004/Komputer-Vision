"""
=============================================================
Modul 03 - Pemrosesan Citra: Histogram Equalization
=============================================================
Histogram equalization adalah teknik untuk meningkatkan kontras
gambar dengan meratakan distribusi intensitas piksel. Teknik ini
berguna untuk gambar yang terlalu gelap, terlalu terang, atau
kontrasnya rendah.

Topik:
  1. Kalkulasi dan visualisasi histogram asli (calcHist)
  2. Equalisasi histogram grayscale dengan equalizeHist
  3. Equalisasi warna via channel L pada ruang warna LAB
  4. Perbandingan underexposed vs normal vs overexposed

Referensi: OpenCV docs - Histogram Equalization
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
    """Muat gambar dari IMAGE_DIR; buat gambar sintetis jika tidak ditemukan."""
    if os.path.exists(IMAGE_PATH):
        img = cv2.imread(IMAGE_PATH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"[INFO] Gambar dimuat dari: {IMAGE_PATH}")
        return img, gray

    # --- Buat gambar sintetis dengan kontras rendah (distribusi intensitas sempit) ---
    print("[INFO] Gambar tidak ditemukan. Membuat gambar sintetis.")
    h, w = 400, 600
    # Gradien horizontal dengan nilai 60-180 (kontras rendah, tidak memanfaatkan 0-255 penuh)
    x = np.linspace(60, 180, w, dtype=np.float32)
    gray = np.tile(x, (h, 1)).astype(np.uint8)
    # Tambahkan blok untuk memperkaya variasi intensitas
    gray[80:180,  50:200]  = 70
    gray[150:280, 250:420] = 150
    gray[50:120,  420:560] = 130
    # Buat versi berwarna sintetis (BGR)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img[:, :, 0] = np.clip(img[:, :, 0].astype(int) - 20, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2].astype(int) + 20, 0, 255)
    return img, gray


def demo_histogram_asli(gray):
    """Demo 1: Menghitung dan memvisualisasikan histogram asli gambar."""
    # Hitung histogram: 1 channel, 256 bin, rentang [0,256]
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Hitung CDF untuk memperlihatkan distribusi kumulatif intensitas
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Demo 1: Histogram Asli Gambar", fontsize=14, fontweight="bold")

    # Panel kiri: gambar grayscale asli
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Gambar Grayscale Asli")
    axes[0].axis("off")

    # Panel kanan: histogram dan CDF
    axes[1].plot(hist, color="steelblue", label="Histogram")
    axes[1].plot(cdf_normalized, color="orange", label="CDF (dinormalisasi)")
    axes[1].set_xlabel("Nilai Intensitas Piksel (0-255)")
    axes[1].set_ylabel("Frekuensi")
    axes[1].set_title("Histogram & CDF - Distribusi Intensitas")
    axes[1].legend()
    axes[1].set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_01_histogram_asli.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 1: histogram asli.")


def demo_equalisasi_grayscale(gray):
    """Demo 2: Equalisasi histogram grayscale dan perbandingan histogram."""
    # cv2.equalizeHist meratakan distribusi intensitas sehingga kontras meningkat
    gray_eq = cv2.equalizeHist(gray)

    # Hitung histogram sebelum dan sesudah untuk perbandingan visual
    hist_asli = cv2.calcHist([gray],    [0], None, [256], [0, 256])
    hist_eq   = cv2.calcHist([gray_eq], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Demo 2: Equalisasi Histogram Grayscale", fontsize=14, fontweight="bold")

    # Baris atas: gambar asli vs hasil equalisasi
    axes[0, 0].imshow(gray,    cmap="gray"); axes[0, 0].set_title("Gambar Asli");              axes[0, 0].axis("off")
    axes[0, 1].imshow(gray_eq, cmap="gray"); axes[0, 1].set_title("Setelah equalizeHist");    axes[0, 1].axis("off")

    # Baris bawah: histogram sebelum vs sesudah equalisasi
    axes[1, 0].plot(hist_asli, color="steelblue")
    axes[1, 0].set_title("Histogram Asli - Distribusi Sempit")
    axes[1, 0].set_xlim([0, 256]); axes[1, 0].set_xlabel("Intensitas"); axes[1, 0].set_ylabel("Frekuensi")

    axes[1, 1].plot(hist_eq, color="green")
    axes[1, 1].set_title("Histogram Setelah Equalisasi - Distribusi Merata")
    axes[1, 1].set_xlim([0, 256]); axes[1, 1].set_xlabel("Intensitas"); axes[1, 1].set_ylabel("Frekuensi")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_02_equalisasi_grayscale.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 2: equalisasi grayscale.")


def demo_equalisasi_warna_lab(img_bgr):
    """Demo 3: Equalisasi warna melalui channel L di ruang warna LAB."""
    # Konversi BGR -> LAB: L = luminansi, a/b = krominansi
    # Strategi: equalisasi hanya L agar warna tidak berubah
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(img_lab)

    # Equalisasi channel L (kecerahan) saja
    L_eq = cv2.equalizeHist(L)

    # Gabungkan L yang sudah diequalisasi dengan a dan b asli
    img_lab_eq = cv2.merge([L_eq, a, b])
    # Konversi kembali ke BGR
    img_hasil = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)

    # Konversi ke RGB agar warna ditampilkan benar oleh matplotlib
    img_asli_rgb  = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2RGB)
    img_hasil_rgb = cv2.cvtColor(img_hasil, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Demo 3: Equalisasi Warna via Channel L (Ruang Warna LAB)",
                 fontsize=14, fontweight="bold")

    axes[0].imshow(img_asli_rgb);  axes[0].set_title("Gambar Warna Asli");           axes[0].axis("off")
    axes[1].imshow(img_hasil_rgb); axes[1].set_title("Setelah Equalisasi LAB-L\n(Warna Terjaga, Kontras Meningkat)"); axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_03_equalisasi_warna_lab.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 3: equalisasi warna LAB.")


def demo_perbandingan_eksposur(gray):
    """Demo 4: Efek equalisasi pada gambar underexposed, normal, dan overexposed."""
    # Simulasikan tiga kondisi pencahayaan berbeda
    underexposed = np.clip(gray.astype(int) - 80, 0, 255).astype(np.uint8)
    normal       = gray.copy()
    overexposed  = np.clip(gray.astype(int) + 80, 0, 255).astype(np.uint8)

    variasi = [
        ("Underexposed (Terlalu Gelap)", underexposed),
        ("Normal",                        normal),
        ("Overexposed (Terlalu Terang)",   overexposed),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("Demo 4: Equalisasi pada Berbagai Kondisi Eksposur",
                 fontsize=14, fontweight="bold")

    for baris, (nama, img_var) in enumerate(variasi):
        # Terapkan equalisasi histogram pada setiap kondisi
        img_eq   = cv2.equalizeHist(img_var)
        hist_var = cv2.calcHist([img_var], [0], None, [256], [0, 256])
        hist_eq  = cv2.calcHist([img_eq],  [0], None, [256], [0, 256])

        # Kolom 1: gambar kondisi asli
        axes[baris, 0].imshow(img_var, cmap="gray", vmin=0, vmax=255)
        axes[baris, 0].set_title(f"{nama}\n(Asli)")
        axes[baris, 0].axis("off")

        # Kolom 2: hasil equalisasi
        axes[baris, 1].imshow(img_eq, cmap="gray", vmin=0, vmax=255)
        axes[baris, 1].set_title(f"{nama}\n(Setelah EQ)")
        axes[baris, 1].axis("off")

        # Kolom 3: perbandingan histogram
        axes[baris, 2].plot(hist_var, color="steelblue", label="Asli",       alpha=0.75)
        axes[baris, 2].plot(hist_eq,  color="green",     label="Setelah EQ", alpha=0.75)
        axes[baris, 2].set_title(f"Histogram - {nama}")
        axes[baris, 2].set_xlim([0, 256])
        axes[baris, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_04_perbandingan_eksposur.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 4: perbandingan eksposur.")


def main():
    """Fungsi utama: jalankan semua demo histogram equalization."""
    print("=" * 60)
    print("  MODUL 03 - HISTOGRAM EQUALIZATION")
    print("=" * 60)

    # Muat gambar dari disk atau buat sintetis jika tidak tersedia
    img_bgr, gray = muat_atau_buat_gambar()

    # Jalankan empat demo secara berurutan
    demo_histogram_asli(gray)
    demo_equalisasi_grayscale(gray)
    demo_equalisasi_warna_lab(img_bgr)
    demo_perbandingan_eksposur(gray)

    print(f"\n[SELESAI] Semua output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
