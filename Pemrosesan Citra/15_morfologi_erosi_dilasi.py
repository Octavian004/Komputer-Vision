"""
============================================================
Praktikum Komputer Vision – Modul 03
Topik   : Morfologi Erosi dan Dilasi
File    : 15_morfologi_erosi_dilasi.py
Deskripsi:
    Mendemonstrasikan operasi morfologi dasar — erosi dan dilasi —
    pada citra biner. Bahasan mencakup:
      1. Erosi (cv2.erode) dan Dilasi (cv2.dilate) dasar dengan kernel 3×3
      2. Berbagai bentuk kernel (RECT / ELLIPSE / CROSS) dan ukuran (3/5/9)
      3. Pengaruh iterasi berganda (1, 3, 5, 7 kali)
      4. Aplikasi nyata: erosi hapus noise kecil, dilasi sambungkan bagian putus
Penulis : Praktikum Komputer Vision
Tanggal : 2026-03-02
============================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Konfigurasi direktori ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "..", "..", "Referensi", "images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Fungsi utilitas ────────────────────────────────────────────────

def muat_atau_buat_gambar() -> np.ndarray:
    """
    Memuat citra 'kota.jpg' dari IMAGE_DIR jika tersedia.
    Jika tidak, membuat citra biner sintetis berisi kotak, lingkaran,
    dan noise titik-titik kecil sebagai gantinya.

    Returns
    -------
    np.ndarray
        Citra biner (0 atau 255) dengan dtype uint8, ukuran 300×400.
    """
    jalur_gambar = os.path.join(IMAGE_DIR, "kota.jpg")

    if os.path.isfile(jalur_gambar):
        # Muat citra nyata lalu ubah ke biner dengan Otsu thresholding
        bgr  = cv2.imread(jalur_gambar)
        abu  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        abu  = cv2.resize(abu, (400, 300))
        _, biner = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return biner

    # ── Buat citra biner sintetis ────────────────────────────────────────
    tinggi, lebar = 300, 400
    kanvas = np.zeros((tinggi, lebar), dtype=np.uint8)

    # Kotak besar
    cv2.rectangle(kanvas, (50, 50), (180, 150), 255, -1)
    # Lingkaran besar
    cv2.circle(kanvas, (290, 100), 60, 255, -1)
    # Kotak kecil (akan terpengaruh erosi)
    cv2.rectangle(kanvas, (50, 200), (130, 260), 255, -1)
    # Garis tipis (mudah terhapus oleh erosi)
    cv2.line(kanvas, (160, 200), (350, 200), 255, 2)
    # Tambahkan noise titik-titik kecil ← target erosi
    rng = np.random.default_rng(42)
    koordinat_noise = rng.integers(0, [lebar, tinggi], size=(200, 2))
    for x, y in koordinat_noise:
        kanvas[y, x] = 255

    return kanvas


def demo_erosi_dilasi_dasar(gambar_biner: np.ndarray) -> None:
    """
    Demo 1 – Erosi dan Dilasi Dasar.

    Menampilkan efek cv2.erode() dan cv2.dilate() menggunakan kernel
    kotak 3×3, serta memvisualisasikan kernel itu sendiri.

    Parameters
    ----------
    gambar_biner : np.ndarray
        Citra biner masukan (0 / 255, dtype uint8).
    """
    print("[Demo 1] Erosi dan Dilasi Dasar")

    # Definisikan kernel kotak 3×3
    kernel_3x3 = np.ones((3, 3), dtype=np.uint8)

    # Terapkan erosi: piksel foreground mengecil
    hasil_erosi  = cv2.erode(gambar_biner,  kernel_3x3, iterations=1)
    # Terapkan dilasi: piksel foreground membesar
    hasil_dilasi = cv2.dilate(gambar_biner, kernel_3x3, iterations=1)

    # Selisih untuk visualisasi perubahan
    selisih_erosi  = cv2.subtract(gambar_biner, hasil_erosi)   # piksel hilang
    selisih_dilasi = cv2.subtract(hasil_dilasi, gambar_biner)  # piksel bertambah

    fig, sumbu = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Demo 1 – Erosi dan Dilasi Dasar (Kernel 3×3)",
                 fontsize=14, fontweight="bold")

    # Baris atas: hasil operasi
    sumbu[0, 0].imshow(gambar_biner, cmap="gray")
    sumbu[0, 0].set_title("Citra Asli (Biner)")
    sumbu[0, 1].imshow(hasil_erosi,  cmap="gray")
    sumbu[0, 1].set_title("Setelah Erosi")
    sumbu[0, 2].imshow(hasil_dilasi, cmap="gray")
    sumbu[0, 2].set_title("Setelah Dilasi")

    # Baris bawah: visualisasi kernel 3×3 dan selisih piksel
    sumbu[1, 0].imshow(kernel_3x3, cmap="hot", vmin=0, vmax=1)
    sumbu[1, 0].set_title("Kernel 3×3 (Kotak)")
    for i in range(3):
        for j in range(3):
            sumbu[1, 0].text(j, i, str(kernel_3x3[i, j]),
                             ha="center", va="center", fontsize=14, color="cyan")
    sumbu[1, 1].imshow(selisih_erosi,  cmap="hot")
    sumbu[1, 1].set_title("Piksel Hilang (Erosi)")
    sumbu[1, 2].imshow(selisih_dilasi, cmap="hot")
    sumbu[1, 2].set_title("Piksel Bertambah (Dilasi)")

    # Tampilkan jumlah piksel putih sebagai keterangan
    n_asli   = int(np.count_nonzero(gambar_biner))
    n_erosi  = int(np.count_nonzero(hasil_erosi))
    n_dilasi = int(np.count_nonzero(hasil_dilasi))
    fig.text(0.5, 0.01,
             f"Piksel putih – Asli: {n_asli} | Erosi: {n_erosi} ({n_erosi - n_asli:+d}) "
             f"| Dilasi: {n_dilasi} ({n_dilasi - n_asli:+d})",
             ha="center", fontsize=10)

    for ax in sumbu.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_demo1_erosi_dilasi_dasar.png"), dpi=150)
    plt.show()
    print("  → Tersimpan: 15_demo1_erosi_dilasi_dasar.png\n")


def demo_berbagai_kernel(gambar_biner: np.ndarray) -> None:
    """
    Demo 2 – Berbagai Bentuk Kernel dan Ukuran.

    Membandingkan hasil erosi dan dilasi dengan tiga bentuk kernel
    (MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS) dan tiga ukuran (3, 5, 9).

    Parameters
    ----------
    gambar_biner : np.ndarray
        Citra biner masukan.
    """
    print("[Demo 2] Berbagai Kernel Shape dan Ukuran")

    # Tiga bentuk structuring element bawaan OpenCV
    bentuk_kernel = {
        "RECT":    cv2.MORPH_RECT,
        "ELLIPSE": cv2.MORPH_ELLIPSE,
        "CROSS":   cv2.MORPH_CROSS,
    }
    ukuran_list = [3, 5, 9]  # ukuran kernel yang dibandingkan

    fig, sumbu = plt.subplots(3, len(ukuran_list) * 2, figsize=(18, 10))
    fig.suptitle(
        "Demo 2 – Berbagai Bentuk Kernel dan Ukuran\n"
        "(Kolom Ganjil=Erosi, Kolom Genap=Dilasi)",
        fontsize=13, fontweight="bold"
    )

    for baris, (nama_bentuk, kode_bentuk) in enumerate(bentuk_kernel.items()):
        for kolom_dasar, ukuran in enumerate(ukuran_list):
            # Buat structuring element
            kernel = cv2.getStructuringElement(kode_bentuk, (ukuran, ukuran))

            # Terapkan erosi dan dilasi dengan kernel tersebut
            erosi  = cv2.erode(gambar_biner,  kernel, iterations=1)
            dilasi = cv2.dilate(gambar_biner, kernel, iterations=1)

            # Dua kolom per ukuran: erosi (ganjil) dan dilasi (genap)
            kol_erosi  = kolom_dasar * 2
            kol_dilasi = kolom_dasar * 2 + 1

            sumbu[baris, kol_erosi].imshow(erosi,  cmap="gray")
            sumbu[baris, kol_erosi].set_title(
                f"{nama_bentuk} {ukuran}×{ukuran}\n(Erosi)", fontsize=8)

            sumbu[baris, kol_dilasi].imshow(dilasi, cmap="gray")
            sumbu[baris, kol_dilasi].set_title(
                f"{nama_bentuk} {ukuran}×{ukuran}\n(Dilasi)", fontsize=8)

    for ax in sumbu.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_demo2_berbagai_kernel.png"), dpi=150)
    plt.show()
    print("  → Tersimpan: 15_demo2_berbagai_kernel.png\n")


def demo_pengaruh_iterasi(gambar_biner: np.ndarray) -> None:
    """
    Demo 3 – Pengaruh Iterasi Berganda.

    Menunjukkan bagaimana menambah jumlah iterasi (1, 3, 5, 7)
    memperkuat efek erosi maupun dilasi secara bertahap.

    Parameters
    ----------
    gambar_biner : np.ndarray
        Citra biner masukan.
    """
    print("[Demo 3] Pengaruh Iterasi Berganda")

    # Gunakan kernel RECT 3×3 sebagai dasar perbandingan iterasi
    kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    iterasi_list = [1, 3, 5, 7]  # jumlah iterasi yang diuji

    fig, sumbu = plt.subplots(2, len(iterasi_list) + 1, figsize=(16, 7))
    fig.suptitle("Demo 3 – Pengaruh Iterasi Berganda (Kernel RECT 3×3)",
                 fontsize=13, fontweight="bold")

    # Kolom pertama diisi citra asli sebagai referensi
    sumbu[0, 0].imshow(gambar_biner, cmap="gray")
    sumbu[0, 0].set_title("Asli")
    sumbu[1, 0].imshow(gambar_biner, cmap="gray")
    sumbu[1, 0].set_title("Asli")
    sumbu[0, 0].set_ylabel("EROSI",  rotation=0, labelpad=50,
                           fontsize=10, fontweight="bold")
    sumbu[1, 0].set_ylabel("DILASI", rotation=0, labelpad=50,
                           fontsize=10, fontweight="bold")

    n_asli = int(np.count_nonzero(gambar_biner))

    for idx, iterasi in enumerate(iterasi_list):
        # Terapkan operasi dengan jumlah iterasi yang berbeda
        erosi  = cv2.erode(gambar_biner,  kernel, iterations=iterasi)
        dilasi = cv2.dilate(gambar_biner, kernel, iterations=iterasi)

        n_erosi  = int(np.count_nonzero(erosi))
        n_dilasi = int(np.count_nonzero(dilasi))

        sumbu[0, idx + 1].imshow(erosi,  cmap="gray")
        sumbu[0, idx + 1].set_title(f"Iter={iterasi}\n({n_erosi - n_asli:+d} px)")

        sumbu[1, idx + 1].imshow(dilasi, cmap="gray")
        sumbu[1, idx + 1].set_title(f"Iter={iterasi}\n({n_dilasi - n_asli:+d} px)")

    for ax in sumbu.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_demo3_pengaruh_iterasi.png"), dpi=150)
    plt.show()
    print("  → Tersimpan: 15_demo3_pengaruh_iterasi.png\n")


def demo_aplikasi_nyata(gambar_biner: np.ndarray) -> None:
    """
    Demo 4 – Aplikasi Nyata: Hapus Noise dan Sambungkan Bagian Putus.

    Mendemonstrasikan dua kasus penggunaan morfologi yang umum:
      • Erosi  → menghapus noise titik-titik kecil (salt noise)
      • Dilasi → menyambungkan komponen yang hampir terputus

    Parameters
    ----------
    gambar_biner : np.ndarray
        Citra biner masukan (idealnya yang sintetis dengan noise).
    """
    print("[Demo 4] Aplikasi Nyata: Hapus Noise & Sambungkan Bagian Putus")

    # ── Tambahkan salt noise ke salinan citra ──────────────────────────────
    rng   = np.random.default_rng(7)
    noisy = gambar_biner.copy()
    # Titik putih acak sebagai noise
    koordinat = rng.integers(0, [noisy.shape[1], noisy.shape[0]], size=(300, 2))
    for x, y in koordinat:
        noisy[y, x] = 255

    # ── Buat citra dengan garis terputus-putus (target dilasi) ──────────────
    broken = np.zeros_like(gambar_biner)
    # Garis horizontal yang terputus-putus
    for x_start in range(30, 370, 20):
        cv2.line(broken, (x_start, 150), (x_start + 12, 150), 255, 2)
    # Objek padat sebagai konteks
    cv2.rectangle(broken, (40, 40),  (180, 130), 255, -1)
    cv2.circle(broken,   (300, 85), 55, 255, -1)

    # ── Erosi hapus noise: kernel ellipse kecil, 1 iterasi ─────────────────
    kernel_kecil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bersih_erosi = cv2.erode(noisy, kernel_kecil, iterations=1)

    # ── Dilasi sambungkan garis: kernel horisontal lebar ──────────────────
    kernel_sambung = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    tersambung     = cv2.dilate(broken, kernel_sambung, iterations=1)

    fig, sumbu = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Demo 4 – Aplikasi Nyata Erosi dan Dilasi",
                 fontsize=14, fontweight="bold")

    # Baris pertama: hapus noise menggunakan erosi
    sumbu[0, 0].imshow(noisy,        cmap="gray")
    sumbu[0, 0].set_title("Citra dengan Noise (Salt)")
    sumbu[0, 1].imshow(bersih_erosi, cmap="gray")
    sumbu[0, 1].set_title("Setelah Erosi\n(Noise Terhapus)")
    diff_noise = cv2.subtract(noisy, bersih_erosi)
    sumbu[0, 2].imshow(diff_noise,   cmap="hot")
    sumbu[0, 2].set_title("Piksel Terhapus (Noise)")
    sumbu[0, 0].set_ylabel("Hapus Noise", fontsize=9, fontweight="bold")

    # Baris kedua: sambungkan garis putus menggunakan dilasi
    sumbu[1, 0].imshow(broken,     cmap="gray")
    sumbu[1, 0].set_title("Garis Terputus-putus")
    sumbu[1, 1].imshow(tersambung, cmap="gray")
    sumbu[1, 1].set_title("Setelah Dilasi\n(Garis Tersambung)")
    diff_sambung = cv2.subtract(tersambung, broken)
    sumbu[1, 2].imshow(diff_sambung, cmap="hot")
    sumbu[1, 2].set_title("Piksel Ditambah (Dilasi)")
    sumbu[1, 0].set_ylabel("Sambungkan", fontsize=9, fontweight="bold")

    fig.text(0.5, 0.01,
             "Erosi: ELLIPSE 3×3 | Dilasi: RECT 11×5",
             ha="center", fontsize=10)

    for ax in sumbu.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_demo4_aplikasi_nyata.png"), dpi=150)
    plt.show()
    print("  → Tersimpan: 15_demo4_aplikasi_nyata.png\n")


# ── Fungsi utama ────────────────────────────────────────────────────────────

def main() -> None:
    """
    Fungsi utama yang menjalankan seluruh demo morfologi erosi dan dilasi.
    """
    print("=" * 60)
    print("  Praktikum 15 – Morfologi: Erosi dan Dilasi")
    print("=" * 60)
    print(f"  Output disimpan di : {OUTPUT_DIR}\n")

    # Muat atau buat citra biner sintetis
    gambar_biner = muat_atau_buat_gambar()
    print(f"  Ukuran citra  : {gambar_biner.shape[1]}×{gambar_biner.shape[0]} piksel")
    print(f"  Piksel putih  : {int(np.count_nonzero(gambar_biner))}\n")

    # Jalankan setiap demo secara berurutan
    demo_erosi_dilasi_dasar(gambar_biner)
    demo_berbagai_kernel(gambar_biner)
    demo_pengaruh_iterasi(gambar_biner)
    demo_aplikasi_nyata(gambar_biner)

    print("=" * 60)
    print("  Semua demo selesai.")
    print("=" * 60)


if __name__ == "__main__":
    main()
