"""
=============================================================
Modul 03 - Pemrosesan Citra: CLAHE
=============================================================
CLAHE (Contrast Limited Adaptive Histogram Equalization) adalah
pengembangan dari histogram equalization yang bekerja secara lokal
pada tile-tile kecil gambar dan membatasi amplifikasi kontras.
CLAHE menghindari noise amplification yang terjadi pada equalizeHist.

Topik:
  1. Perbandingan Global EQ vs CLAHE
  2. Variasi clipLimit (1, 2, 4, 8, 20, 40)
  3. Variasi tileGridSize
  4. CLAHE pada gambar berwarna via ruang warna LAB

Referensi: cv2.createCLAHE, cv2.equalizeHist
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

    # --- Gambar sintetis: gradien dengan kontras rendah ---
    print("[INFO] Gambar tidak ditemukan. Membuat gambar sintetis.")
    h, w = 400, 600
    # Gradien intensitas sempit 50-160 agar efek CLAHE terlihat jelas
    x = np.linspace(50, 160, w, dtype=np.float32)
    gray = np.tile(x, (h, 1)).astype(np.uint8)
    # Tambah noise Gaussian ringan untuk uji amplifikasi noise
    noise = np.random.normal(0, 8, gray.shape).astype(np.int16)
    gray  = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Beberapa blok objek
    gray[60:160,  40:180]  = 80
    gray[180:320, 200:380] = 140
    gray[40:100,  400:560] = 120
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img, gray


def demo_global_eq_vs_clahe(gray):
    """Demo 1: Perbandingan Global Histogram Equalization vs CLAHE."""
    # Metode 1: equalizeHist - equalisasi global seluruh gambar
    gray_eq = cv2.equalizeHist(gray)

    # Metode 2: CLAHE - equalisasi adaptif per tile dengan batas kontras
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Hitung histogram ketiga gambar untuk perbandingan distribusi
    hist_asli  = cv2.calcHist([gray],       [0], None, [256], [0, 256])
    hist_eq    = cv2.calcHist([gray_eq],    [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([gray_clahe], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Demo 1: Global EQ vs CLAHE", fontsize=14, fontweight="bold")

    # Baris atas: gambar asli, EQ global, CLAHE
    axes[0, 0].imshow(gray,        cmap="gray"); axes[0, 0].set_title("Asli");               axes[0, 0].axis("off")
    axes[0, 1].imshow(gray_eq,     cmap="gray"); axes[0, 1].set_title("equalizeHist (Global)"); axes[0, 1].axis("off")
    axes[0, 2].imshow(gray_clahe,  cmap="gray"); axes[0, 2].set_title("CLAHE (clipLimit=2, tile=8x8)"); axes[0, 2].axis("off")

    # Baris bawah: histogram masing-masing
    axes[1, 0].plot(hist_asli,  color="steelblue"); axes[1, 0].set_title("Histogram Asli");       axes[1, 0].set_xlim([0, 256])
    axes[1, 1].plot(hist_eq,    color="red");       axes[1, 1].set_title("Histogram EQ Global");  axes[1, 1].set_xlim([0, 256])
    axes[1, 2].plot(hist_clahe, color="green");     axes[1, 2].set_title("Histogram CLAHE");      axes[1, 2].set_xlim([0, 256])

    for ax in axes[1]:
        ax.set_xlabel("Intensitas"); ax.set_ylabel("Frekuensi")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_01_global_eq_vs_clahe.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 1: global EQ vs CLAHE.")


def demo_variasi_clip_limit(gray):
    """Demo 2: Pengaruh variasi clipLimit terhadap hasil CLAHE."""
    # clipLimit mengontrol batas amplifikasi kontras:
    # - Nilai kecil: perubahan kecil, lebih dekat ke gambar asli
    # - Nilai besar: kontras tinggi, rawan noise amplification
    clip_limits = [1, 2, 4, 8, 20, 40]
    tile = (8, 8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Demo 2: Variasi clipLimit pada CLAHE (tileGridSize=8x8)",
                 fontsize=14, fontweight="bold")

    for i, cl in enumerate(clip_limits):
        baris, kol = divmod(i, 3)
        # Buat objek CLAHE dengan clipLimit berbeda
        clahe  = cv2.createCLAHE(clipLimit=cl, tileGridSize=tile)
        result = clahe.apply(gray)
        axes[baris, kol].imshow(result, cmap="gray")
        axes[baris, kol].set_title(f"clipLimit = {cl}")
        axes[baris, kol].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_02_variasi_clip_limit.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 2: variasi clipLimit.")


def demo_variasi_tile_grid(gray):
    """Demo 3: Pengaruh variasi tileGridSize terhadap hasil CLAHE."""
    # tileGridSize menentukan ukuran grid tile:
    # - Tile kecil: adaptasi lebih lokal, detail lebih tajam
    # - Tile besar: mendekati equalisasi global
    tile_sizes = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
    clip = 2.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Demo 3: Variasi tileGridSize pada CLAHE (clipLimit=2.0)",
                 fontsize=14, fontweight="bold")

    for i, ts in enumerate(tile_sizes):
        baris, kol = divmod(i, 3)
        clahe  = cv2.createCLAHE(clipLimit=clip, tileGridSize=ts)
        result = clahe.apply(gray)
        axes[baris, kol].imshow(result, cmap="gray")
        axes[baris, kol].set_title(f"tileGridSize = {ts[0]}x{ts[1]}")
        axes[baris, kol].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_03_variasi_tile_grid.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 3: variasi tileGridSize.")


def demo_clahe_warna_lab(img_bgr):
    """Demo 4: Penerapan CLAHE pada gambar berwarna via ruang warna LAB."""
    # Strategi: terapkan CLAHE hanya pada channel L (luminansi)
    # sehingga warna (channel a dan b) tidak terpengaruh
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(img_lab)

    # Untuk perbandingan: global EQ pada L
    L_eq = cv2.equalizeHist(L)

    # CLAHE pada L dengan parameter default yang baik
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)

    # Rekonstruksi gambar berwarna dari setiap metode
    img_eq_bgr    = cv2.cvtColor(cv2.merge([L_eq,    a, b]), cv2.COLOR_LAB2BGR)
    img_clahe_bgr = cv2.cvtColor(cv2.merge([L_clahe, a, b]), cv2.COLOR_LAB2BGR)

    # Konversi ke RGB untuk matplotlib
    asli_rgb  = cv2.cvtColor(img_bgr,       cv2.COLOR_BGR2RGB)
    eq_rgb    = cv2.cvtColor(img_eq_bgr,    cv2.COLOR_BGR2RGB)
    clahe_rgb = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Demo 4: CLAHE pada Gambar Berwarna (via Ruang Warna LAB)",
                 fontsize=14, fontweight="bold")

    axes[0].imshow(asli_rgb);  axes[0].set_title("Asli");               axes[0].axis("off")
    axes[1].imshow(eq_rgb);    axes[1].set_title("Global EQ (LAB-L)");  axes[1].axis("off")
    axes[2].imshow(clahe_rgb); axes[2].set_title("CLAHE (LAB-L)\nclipLimit=2, tile=8x8"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_04_clahe_warna_lab.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("[SELESAI] Demo 4: CLAHE warna LAB.")


def main():
    """Fungsi utama: jalankan semua demo CLAHE."""
    print("=" * 60)
    print("  MODUL 03 - CLAHE (Contrast Limited Adaptive HE)")
    print("=" * 60)

    # Muat gambar atau buat sintetis jika tidak tersedia
    img_bgr, gray = muat_atau_buat_gambar()

    # Jalankan empat demo secara berurutan
    demo_global_eq_vs_clahe(gray)
    demo_variasi_clip_limit(gray)
    demo_variasi_tile_grid(gray)
    demo_clahe_warna_lab(img_bgr)

    print(f"\n[SELESAI] Semua output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
