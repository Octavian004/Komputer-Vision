"""
=============================================================================
Modul 03 - Pemrosesan Citra (Image Processing)
Praktikum 17: Top-Hat dan Black-Hat Transform Mendalam
=============================================================================
Deskripsi:
    Mempelajari transformasi morfologi lanjutan yaitu Top-Hat dan Black-Hat
    yang berguna untuk mendeteksi objek kecil terang/gelap dan mengoreksi
    pencahayaan tidak merata (uneven illumination).

    - White Top-Hat  = gambar_asli - opening
      Menonjolkan objek CERAH yang lebih kecil dari structuring element
    - Black Top-Hat  = closing - gambar_asli
      Menonjolkan objek GELAP yang lebih kecil dari structuring element

Topik yang dibahas:
    1. White Top-Hat Transform (WTH)
    2. Black Top-Hat Transform (BTH)
    3. Koreksi pencahayaan tidak merata via Top-Hat background subtraction
    4. Pengaruh ukuran Structuring Element terhadap hasil Top-Hat

Referensi:
    - Gonzalez & Woods, "Digital Image Processing", 4th Edition
    - OpenCV Documentation: morphologyEx()
    - https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

Penulis  : Praktikum Komputer Vision
Tanggal  : 2026-03-02
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Konfigurasi direktori
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "..", "..", "Referensi", "images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: muat gambar atau buat gambar sintetis
# ---------------------------------------------------------------------------
def muat_atau_buat_gambar():
    """
    Memuat gambar kota.jpg dari IMAGE_DIR sebagai grayscale.
    Jika tidak ditemukan, membuat gambar sintetis yang terdiri dari:
      - Latar belakang gradien (simulasi pencahayaan tidak merata)
      - Titik-titik cerah kecil (bright spots)
      - Titik-titik gelap kecil (dark spots)
    Gambar sintetis ini ideal untuk mendemonstrasikan Top-Hat dan Black-Hat.

    Returns:
        tuple: (gambar_gray uint8, label_sumber str)
    """
    jalur_kota = os.path.join(IMAGE_DIR, "kota.jpg")
    if os.path.isfile(jalur_kota):
        gambar = cv2.imread(jalur_kota, cv2.IMREAD_GRAYSCALE)
        if gambar is not None:
            print(f"[INFO] Gambar dimuat dari: {jalur_kota}")
            gambar = cv2.resize(gambar, (512, 512))
            return gambar, "kota.jpg"

    print("[INFO] kota.jpg tidak ditemukan - membuat gambar sintetis.")

    tinggi, lebar = 512, 512

    # ---- Buat latar gradien 2D (mensimulasikan pencahayaan tidak merata) ----
    x = np.linspace(0, 255, lebar, dtype=np.float32)
    y = np.linspace(50, 200, tinggi, dtype=np.float32)
    gradien_x, gradien_y = np.meshgrid(x, y)
    latar = (gradien_x * 0.5 + gradien_y * 0.5).astype(np.uint8)

    gambar = latar.copy().astype(np.float32)
    np.random.seed(42)

    # ---- Tambahkan bright spots kecil (objek cerah lebih kecil dari SE) ----
    for _ in range(40):
        cx = np.random.randint(20, lebar - 20)
        cy = np.random.randint(20, tinggi - 20)
        radius = np.random.randint(3, 10)
        cv2.circle(gambar, (cx, cy), radius, 255, -1)

    # ---- Tambahkan dark spots kecil (objek gelap lebih kecil dari SE) ----
    for _ in range(40):
        cx = np.random.randint(20, lebar - 20)
        cy = np.random.randint(20, tinggi - 20)
        radius = np.random.randint(3, 10)
        cv2.circle(gambar, (cx, cy), radius, 0, -1)

    # ---- Tambahkan sedikit noise agar lebih realistis ----
    noise = np.random.normal(0, 5, gambar.shape).astype(np.float32)
    gambar = np.clip(gambar + noise, 0, 255).astype(np.uint8)

    return gambar, "sintetis"


# ---------------------------------------------------------------------------
# Demo 1: White Top-Hat Transform
# ---------------------------------------------------------------------------
def demo_white_tophat(gambar_gray, label_sumber):
    """
    Mendemonstrasikan White Top-Hat (WTH):
        WTH = gambar_asli - opening(gambar_asli, SE)

    Opening menghilangkan objek kecil yang LEBIH KECIL dari SE.
    WTH mengekstrak kembali objek-objek cerah kecil tersebut.

    Args:
        gambar_gray  : gambar grayscale uint8
        label_sumber : string label sumber gambar
    """
    print("\n[Demo 1] White Top-Hat Transform")

    # Structuring Element disk ukuran 21x21 piksel
    ukuran_se = 21
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ukuran_se, ukuran_se))

    # ---- Hitung opening: erosi diikuti dilasi ----
    opening = cv2.morphologyEx(gambar_gray, cv2.MORPH_OPEN, se)

    # ---- White Top-Hat manual = gambar asli - opening ----
    tophat_manual = cv2.subtract(gambar_gray, opening)

    # ---- Verifikasi dengan fungsi bawaan OpenCV MORPH_TOPHAT ----
    tophat_opencv = cv2.morphologyEx(gambar_gray, cv2.MORPH_TOPHAT, se)

    # ---- Visualisasi 6 panel ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Demo 1: White Top-Hat Transform\n"
        "WTH = Gambar Asli - Opening  =>  Menonjolkan Objek Cerah Kecil",
        fontsize=13, fontweight="bold"
    )

    axes[0, 0].imshow(gambar_gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Gambar Asli ({label_sumber})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(opening, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Opening (SE {ukuran_se}x{ukuran_se})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(tophat_manual, cmap="hot", vmin=0, vmax=255)
    axes[0, 2].set_title("WTH Manual = Asli - Opening")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(tophat_opencv, cmap="hot", vmin=0, vmax=255)
    axes[1, 0].set_title("WTH cv2.MORPH_TOPHAT (identik)")
    axes[1, 0].axis("off")

    # Selisih antara hasil manual dan OpenCV (harus sangat kecil atau nol)
    selisih = cv2.absdiff(tophat_manual, tophat_opencv)
    axes[1, 1].imshow(selisih, cmap="gray")
    axes[1, 1].set_title(f"Selisih Manual vs OpenCV\n(max={selisih.max()})")
    axes[1, 1].axis("off")

    # Histogram Top-Hat untuk melihat distribusi nilai intensitas
    axes[1, 2].hist(tophat_opencv.ravel(), bins=64, range=(0, 255),
                    color="tomato", edgecolor="darkred")
    axes[1, 2].set_title("Histogram WTH")
    axes[1, 2].set_xlabel("Intensitas Piksel")
    axes[1, 2].set_ylabel("Jumlah Piksel")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_demo1_white_tophat.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Tersimpan: 17_demo1_white_tophat.png")


# ---------------------------------------------------------------------------
# Demo 2: Black Top-Hat Transform
# ---------------------------------------------------------------------------
def demo_black_tophat(gambar_gray, label_sumber):
    """
    Mendemonstrasikan Black Top-Hat (BTH):
        BTH = closing(gambar_asli, SE) - gambar_asli

    Closing mengisi lubang gelap kecil yang LEBIH KECIL dari SE.
    BTH mengekstrak kembali objek-objek gelap kecil tersebut.

    Args:
        gambar_gray  : gambar grayscale uint8
        label_sumber : string label sumber gambar
    """
    print("\n[Demo 2] Black Top-Hat Transform")

    ukuran_se = 21
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ukuran_se, ukuran_se))

    # ---- Hitung closing: dilasi diikuti erosi ----
    closing = cv2.morphologyEx(gambar_gray, cv2.MORPH_CLOSE, se)

    # ---- Black Top-Hat manual = closing - gambar asli ----
    blackhat_manual = cv2.subtract(closing, gambar_gray)

    # ---- Verifikasi dengan fungsi bawaan OpenCV MORPH_BLACKHAT ----
    blackhat_opencv = cv2.morphologyEx(gambar_gray, cv2.MORPH_BLACKHAT, se)

    # ---- Visualisasi 6 panel ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Demo 2: Black Top-Hat Transform\n"
        "BTH = Closing - Gambar Asli  =>  Menonjolkan Objek Gelap Kecil",
        fontsize=13, fontweight="bold"
    )

    axes[0, 0].imshow(gambar_gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Gambar Asli ({label_sumber})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(closing, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Closing (SE {ukuran_se}x{ukuran_se})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(blackhat_manual, cmap="Blues", vmin=0, vmax=255)
    axes[0, 2].set_title("BTH Manual = Closing - Asli")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(blackhat_opencv, cmap="Blues", vmin=0, vmax=255)
    axes[1, 0].set_title("BTH cv2.MORPH_BLACKHAT (identik)")
    axes[1, 0].axis("off")

    # Selisih verifikasi manual vs OpenCV
    selisih = cv2.absdiff(blackhat_manual, blackhat_opencv)
    axes[1, 1].imshow(selisih, cmap="gray")
    axes[1, 1].set_title(f"Selisih Manual vs OpenCV\n(max={selisih.max()})")
    axes[1, 1].axis("off")

    # Histogram Black Top-Hat
    axes[1, 2].hist(blackhat_opencv.ravel(), bins=64, range=(0, 255),
                    color="steelblue", edgecolor="navy")
    axes[1, 2].set_title("Histogram BTH")
    axes[1, 2].set_xlabel("Intensitas Piksel")
    axes[1, 2].set_ylabel("Jumlah Piksel")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_demo2_black_tophat.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Tersimpan: 17_demo2_black_tophat.png")


# ---------------------------------------------------------------------------
# Demo 3: Koreksi Pencahayaan Tidak Merata via Top-Hat
# ---------------------------------------------------------------------------
def demo_koreksi_iluminasi(gambar_gray):
    """
    Mensimulasikan dan mengoreksi pencahayaan tidak merata menggunakan
    White Top-Hat sebagai teknik background subtraction:

    Langkah:
      1. Tambahkan gradien pencahayaan ke gambar (simulasi iluminasi tidak rata)
      2. Estimasi background menggunakan Opening dengan SE besar
      3. WTH = gambar tidak rata - background  (hapus variasi lambat)
      4. Stretch kontras ke 0-255
      5. Bandingkan dengan CLAHE sebagai metode alternatif

    Args:
        gambar_gray : gambar grayscale uint8
    """
    print("\n[Demo 3] Koreksi Pencahayaan Tidak Merata via Top-Hat")

    tinggi, lebar = gambar_gray.shape

    # ---- Buat gradien iluminasi horizontal untuk simulasi pencahayaan tidak rata ----
    gradien = np.tile(
        np.linspace(0, 120, lebar, dtype=np.float32), (tinggi, 1)
    )
    gambar_tidak_rata = np.clip(
        gambar_gray.astype(np.float32) + gradien, 0, 255
    ).astype(np.uint8)

    # ---- Estimasi background via opening (SE harus > ukuran fitur yang dipertahankan) ----
    ukuran_se = 51
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ukuran_se, ukuran_se))
    background = cv2.morphologyEx(gambar_tidak_rata, cv2.MORPH_OPEN, se)

    # ---- Top-Hat = gambar tidak rata - background ----
    tophat_koreksi = cv2.subtract(gambar_tidak_rata, background)

    # ---- Normalisasi kontras ke 0-255 untuk tampilan optimal ----
    terkoreksi_norm = cv2.normalize(
        tophat_koreksi, None, 0, 255, cv2.NORM_MINMAX
    )

    # ---- Pembanding: CLAHE (Contrast Limited Adaptive Histogram Equalization) ----
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    terkoreksi_clahe = clahe.apply(gambar_tidak_rata)

    # ---- Visualisasi 6 panel ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Demo 3: Koreksi Pencahayaan Tidak Merata\n"
        "Background Subtraction via White Top-Hat",
        fontsize=13, fontweight="bold"
    )

    axes[0, 0].imshow(gambar_gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Gambar Asli (pencahayaan merata)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gambar_tidak_rata, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Gambar + Gradien Iluminasi\n(simulasi tidak rata)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(background, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title(f"Background Diestimasi\n(Opening SE {ukuran_se}x{ukuran_se})")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(tophat_koreksi, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("WTH = Tidak Rata - Background\n(sebelum stretch)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(terkoreksi_norm, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("WTH + Stretch Kontras\n(hasil koreksi Top-Hat)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(terkoreksi_clahe, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("Pembanding: CLAHE\n(adaptive equalization)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_demo3_koreksi_iluminasi.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Tersimpan: 17_demo3_koreksi_iluminasi.png")


# ---------------------------------------------------------------------------
# Demo 4: Pengaruh Ukuran SE terhadap Hasil Top-Hat
# ---------------------------------------------------------------------------
def demo_pengaruh_ukuran_se(gambar_gray):
    """
    Membandingkan pengaruh ukuran Structuring Element (SE) terhadap
    hasil White Top-Hat pada gambar yang sama.

    SE yang diuji: 5x5, 15x15, 31x31, 61x61
    Semakin besar SE => semakin besar objek yang "tersaring" oleh opening
    => WTH menangkap fitur berukuran semakin besar.

    Args:
        gambar_gray : gambar grayscale uint8
    """
    print("\n[Demo 4] Pengaruh Ukuran SE terhadap Hasil Top-Hat")

    # Daftar ukuran SE yang akan dibandingkan
    ukuran_se_list = [5, 15, 31, 61]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Demo 4: Pengaruh Ukuran Structuring Element pada White Top-Hat\n"
        "SE lebih besar => fitur yang lebih besar ikut tertangkap",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, len(ukuran_se_list) + 1, figure=fig,
                           hspace=0.4, wspace=0.08)

    # Kolom pertama: gambar asli (span dua baris untuk referensi visual)
    ax_asli = fig.add_subplot(gs[:, 0])
    ax_asli.imshow(gambar_gray, cmap="gray", vmin=0, vmax=255)
    ax_asli.set_title("Gambar Asli", fontsize=11)
    ax_asli.set_xticks([]); ax_asli.set_yticks([])

    for idx, ukuran in enumerate(ukuran_se_list, start=1):
        # Buat SE berbentuk elips sesuai ukuran
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ukuran, ukuran))
        # Hitung White Top-Hat
        tophat = cv2.morphologyEx(gambar_gray, cv2.MORPH_TOPHAT, se)

        # Baris atas: hasil Top-Hat dengan colormap heat
        ax_wth = fig.add_subplot(gs[0, idx])
        ax_wth.imshow(tophat, cmap="hot", vmin=0, vmax=255)
        ax_wth.set_title(f"WTH SE={ukuran}x{ukuran}", fontsize=10)
        ax_wth.set_xticks([]); ax_wth.set_yticks([])

        # Baris bawah: histogram untuk perbandingan kuantitatif
        ax_hist = fig.add_subplot(gs[1, idx])
        ax_hist.hist(tophat.ravel(), bins=64, range=(0, 255),
                     color="tomato", edgecolor="darkred")
        rata = tophat.mean()
        # Garis putus-putus menunjukkan nilai rata-rata piksel
        ax_hist.axvline(rata, color="navy", linestyle="--", linewidth=1.2,
                        label=f"mu={rata:.1f}")
        ax_hist.legend(fontsize=8)
        ax_hist.set_xlabel("Intensitas", fontsize=8)
        ax_hist.tick_params(labelsize=7)

    plt.savefig(os.path.join(OUTPUT_DIR, "17_demo4_ukuran_se.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Tersimpan: 17_demo4_ukuran_se.png")


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------
def main():
    """
    Fungsi utama yang menjalankan seluruh demonstrasi Top-Hat dan Black-Hat.
    """
    print("=" * 70)
    print("  Praktikum 17: Top-Hat dan Black-Hat Transform Mendalam")
    print("=" * 70)
    print(f"  Output directory : {OUTPUT_DIR}")

    # Muat gambar dari disk atau buat gambar sintetis
    gambar_gray, label_sumber = muat_atau_buat_gambar()
    print(f"  Ukuran gambar    : {gambar_gray.shape[1]}x{gambar_gray.shape[0]} piksel")

    # Jalankan semua demo secara berurutan
    demo_white_tophat(gambar_gray, label_sumber)   # Demo 1: White Top-Hat
    demo_black_tophat(gambar_gray, label_sumber)   # Demo 2: Black Top-Hat
    demo_koreksi_iluminasi(gambar_gray)            # Demo 3: Koreksi iluminasi
    demo_pengaruh_ukuran_se(gambar_gray)           # Demo 4: Pengaruh ukuran SE

    print("\n" + "=" * 70)
    print("  Semua demo selesai. Hasil tersimpan di folder output/")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
