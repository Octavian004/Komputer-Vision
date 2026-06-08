"""
============================================================
Praktikum Komputer Vision – Modul 03
Topik   : Operasi Morfologi Lanjut
File    : 16_morfologi_lanjut.py
Deskripsi:
    Mendemonstrasikan operasi morfologi tingkat lanjut menggunakan
    cv2.morphologyEx(). Bahasan mencakup:
      1. Opening (erosi+dilasi) hapus noise kecil vs
         Closing (dilasi+erosi) tutup lubang
      2. Morphological Gradient (dilasi − erosi = outline tipis)
      3. Top-Hat (asli − opening = objek terang) dan
         Black-Hat (closing − asli = objek gelap)
      4. Hit-or-Miss transform untuk deteksi pola khusus
Penulis : Praktikum Komputer Vision
Tanggal : 2026-03-02
============================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Konfigurasi direktori ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "..", "..", "Referensi", "images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Fungsi utilitas ──────────────────────────────────────────────────────────

def muat_atau_buat_gambar() -> np.ndarray:
    """
    Memuat citra 'kota.jpg' dari IMAGE_DIR jika tersedia.
    Jika tidak, membuat citra biner sintetis berisi objek dengan
    noise titik-titik kecil dan lubang internal.

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
        _, biner = cv2.threshold(abu, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return biner

    # ── Buat citra biner sintetis ────────────────────────────────────────────
    tinggi, lebar = 300, 400
    kanvas = np.zeros((tinggi, lebar), dtype=np.uint8)

    # Kotak besar dengan lubang internal (target closing)
    cv2.rectangle(kanvas, (40, 40), (180, 140), 255, -1)
    cv2.rectangle(kanvas, (80, 70), (140, 110), 0, -1)   # lubang di dalam

    # Lingkaran besar
    cv2.circle(kanvas, (290, 90), 65, 255, -1)
    # Buat beberapa lubang kecil di dalam lingkaran
    for cx, cy in [(270, 75), (300, 100), (280, 110)]:
        cv2.circle(kanvas, (cx, cy), 6, 0, -1)

    # Objek kecil (noise ringan – target opening)
    rng = np.random.default_rng(42)
    kx  = rng.integers(30, lebar - 30, size=80)
    ky  = rng.integers(180, tinggi - 20, size=80)
    for x, y in zip(kx, ky):
        cv2.circle(kanvas, (int(x), int(y)), 2, 255, -1)

    return kanvas


def demo_opening_closing(gambar_biner: np.ndarray) -> None:
    """
    Demo 1 – Opening vs Closing.

    Opening  = Erosi lalu Dilasi → menghapus noise kecil sambil
               mempertahankan ukuran objek besar.
    Closing  = Dilasi lalu Erosi → menutup lubang kecil di dalam
               objek sambil mempertahankan ukurannya.

    Parameters
    ----------
    gambar_biner : np.ndarray
        Citra biner masukan.
    """
    print("[Demo 1] Opening vs Closing")

    # Kernel ellipse 5×5 untuk operasi morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Opening: erosi kemudian dilasi (hapus noise kecil)
    opening = cv2.morphologyEx(gambar_biner, cv2.MORPH_OPEN,  kernel)
    # Closing: dilasi kemudian erosi (tutup lubang kecil)
    closing = cv2.morphologyEx(gambar_biner, cv2.MORPH_CLOSE, kernel)

    # Visualisasikan perbedaan antara asli dan hasil operasi
    diff_open  = cv2.subtract(gambar_biner, opening)  # piksel yang dihapus opening
    diff_close = cv2.subtract(closing, gambar_biner)  # piksel yang ditambah closing

    fig, sumbu = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Demo 1 – Opening (Hapus Noise) vs Closing (Tutup Lubang)",
                 fontsize=13, fontweight="bold")

    # Baris atas: citra asli dan hasil operasi
    sumbu[0, 0].imshow(gambar_biner, cmap="gray")
    sumbu[0, 0].set_title("Citra Asli")
    sumbu[0, 1].imshow(opening,      cmap="gray")
    sumbu[0, 1].set_title("Opening\n(Erosi→Dilasi)")
    sumbu[0, 2].imshow(closing,      cmap="gray")
    sumbu[0, 2].set_title("Closing\n(Dilasi→Erosi)")

    # Baris bawah: selisih (noise terhapus / lubang tertutup)
    sumbu[1, 0].axis("off")  # kosong
    sumbu[1, 1].imshow(diff_open,  cmap="hot")
    sumbu[1, 1].set_title("Noise Terhapus (Opening)")
    sumbu[1, 2].imshow(diff_close, cmap="hot")
    sumbu[1, 2].set_title("Lubang Tertutup (Closing)")

    # Statistik piksel sebagai catatan bawah
    n  = int(np.count_nonzero(gambar_biner))
    no = int(np.count_nonzero(opening))
    nc = int(np.count_nonzero(closing))
    fig.text(0.5, 0.01,
             f"Piksel putih – Asli: {n} | Opening: {no} ({no - n:+d}) "
             f"| Closing: {nc} ({nc - n:+d})",
             ha="center", fontsize=10)

    for ax in sumbu.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_demo1_opening_closing.png"), dpi=150)
    plt.show()
    print("  → Tersimpan: 16_demo1_opening_closing.png\n")


def demo_morphological_gradient(gambar_biner: np.ndarray) -> None:
    """
    Demo 2 – Morphological Gradient.

    Gradient morfologi = Dilasi − Erosi, menghasilkan outline tipis
    di tepi objek. Variasi kernel dan ukuran mempengaruhi ketebalan
    garis tepi yang dihasilkan.

    Parameters
    ----------
    gambar_biner : np.ndarray
        Citra biner masukan.
    """
    print("[Demo 2] Morphological Gradient")

    ukuran_list  = [3, 5, 9]
    bentuk_list  = {
        "RECT":    cv2.MORPH_RECT,
        "ELLIPSE": cv2.MORPH_ELLIPSE,
    }

    fig, sumbu = plt.subplots(len(bentuk_list), len(ukuran_list) + 1,
                              figsize=(15, 8))
    fig.suptitle("Demo 2 – Morphological Gradient (Dilasi \u2212 Erosi = Outline)",
                 fontsize=13, fontweight="bold")

    for baris, (nama, kode) in enumerate(bentuk_list.items()):
        # Kolom pertama: citra asli sebagai referensi
        sumbu[baris, 0].imshow(gambar_biner, cmap="gray")
        sumbu[baris, 0].set_title("Asli")
        sumbu[baris, 0].set_ylabel(nama, fontsize=9, fontweight="bold")

        for kolom, ukuran in enumerate(ukuran_list, start=1):
            kernel   = cv2.getStructuringElement(kode, (ukuran, ukuran))
            # Gradient = dilasi minus erosi → outline objek
            gradient = cv2.morphologyEx(gambar_biner,
                                        cv2.MORPH_GRADIENT, kernel)
            n_px = int(np.count_nonzero(gradient))
            sumbu[baris, kolom].imshow(gradient, cmap="hot")
            sumbu[baris, kolom].set_title(
                f"{ukuran}\u00d7{ukuran} ({n_px} px)", fontsize=9)

    for ax in sumbu.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_demo2_morphological_gradient.png"),
                dpi=150)
    plt.show()
    print("  → Tersimpan: 16_demo2_morphological_gradient.png\n")


def demo_tophat_blackhat(gambar_biner: np.ndarray) -> None:
    """
    Demo 3 – Top-Hat dan Black-Hat Transform.

    Top-Hat   = Asli − Opening → menonjolkan objek/highlight kecil
                yang lebih terang dari latar (foreground kecil).
    Black-Hat = Closing − Asli → menonjolkan objek gelap kecil
                di dalam region terang (background kecil).

    Parameters
    ----------
    gambar_biner : np.ndarray
        Citra biner masukan.
    """
    print("[Demo 3] Top-Hat dan Black-Hat Transform")

    # Kernel yang lebih besar agar efek Top-Hat/Black-Hat terlihat jelas
    kernel_besar = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_sed   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Top-Hat: asli minus opening (tampilkan objek terang kecil)
    tophat   = cv2.morphologyEx(gambar_biner, cv2.MORPH_TOPHAT,   kernel_besar)
    # Black-Hat: closing minus asli (tampilkan lubang gelap kecil)
    blackhat = cv2.morphologyEx(gambar_biner, cv2.MORPH_BLACKHAT, kernel_besar)

    # Eksperimen dengan ukuran kernel berbeda
    tophat_kecil   = cv2.morphologyEx(gambar_biner, cv2.MORPH_TOPHAT,   kernel_sed)
    blackhat_kecil = cv2.morphologyEx(gambar_biner, cv2.MORPH_BLACKHAT, kernel_sed)

    fig, sumbu = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Demo 3 – Top-Hat (Objek Terang Kecil) & Black-Hat (Lubang Gelap Kecil)",
                 fontsize=12, fontweight="bold")

    # Baris atas: asli, top-hat besar, black-hat besar
    sumbu[0, 0].imshow(gambar_biner, cmap="gray")
    sumbu[0, 0].set_title("Citra Asli")
    sumbu[0, 1].imshow(tophat,   cmap="hot")
    sumbu[0, 1].set_title("Top-Hat (kernel 15×15)\nAsli − Opening")
    sumbu[0, 2].imshow(blackhat, cmap="hot")
    sumbu[0, 2].set_title("Black-Hat (kernel 15×15)\nClosing − Asli")

    # Baris bawah: perbandingan kernel lebih kecil
    sumbu[1, 0].axis("off")
    sumbu[1, 1].imshow(tophat_kecil,   cmap="hot")
    sumbu[1, 1].set_title("Top-Hat (kernel 9×9)")
    sumbu[1, 2].imshow(blackhat_kecil, cmap="hot")
    sumbu[1, 2].set_title("Black-Hat (kernel 9×9)")

    fig.text(0.5, 0.01,
             "Top-Hat = cv2.MORPH_TOPHAT  |  Black-Hat = cv2.MORPH_BLACKHAT",
             ha="center", fontsize=10)

    for ax in sumbu.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_demo3_tophat_blackhat.png"), dpi=150)
    plt.show()
    print("  → Tersimpan: 16_demo3_tophat_blackhat.png\n")


def demo_hit_or_miss(gambar_biner: np.ndarray) -> None:
    """
    Demo 4 – Hit-or-Miss Transform.

    Hit-or-Miss mendeteksi pola piksel yang sangat spesifik:
    kernel foreground harus cocok (HIT) dan kernel background
    juga harus cocok (MISS). Berguna untuk mendeteksi sudut,
    titik terisolasi, atau pola geometris tertentu.

    Di sini mendeteksi piksel terisolasi (titik kecil single-pixel)
    dan ujung garis (endpoint).

    Parameters
    ----------
    gambar_biner : np.ndarray
        Citra biner masukan.
    """
    print("[Demo 4] Hit-or-Miss Transform")

    # ── Kernel 1: deteksi piksel terisolasi ─────────────────────────────────
    # Foreground = 1 (piksel tengah putih), lainnya harus background (0)
    # Gunakan representasi: 1=foreground, -1=background, 0=don't care
    kernel_isolasi = np.array([
        [-1, -1, -1],
        [-1,  1, -1],
        [-1, -1, -1],
    ], dtype=np.int32)

    # ── Kernel 2: deteksi ujung garis kanan (endpoint mengarah kanan) ────────
    kernel_ujung_kanan = np.array([
        [ 0, -1,  0],
        [-1,  1, -1],
        [ 0,  1,  0],
    ], dtype=np.int32)

    # ── Kernel 3: deteksi sudut pojok kiri-atas ─────────────────────────────
    kernel_sudut = np.array([
        [-1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
    ], dtype=np.int32)

    # Terapkan Hit-or-Miss menggunakan MORPH_HITMISS
    hit_miss_isolasi = cv2.morphologyEx(
        gambar_biner, cv2.MORPH_HITMISS,
        kernel_isolasi.astype(np.int8))
    hit_miss_ujung   = cv2.morphologyEx(
        gambar_biner, cv2.MORPH_HITMISS,
        kernel_ujung_kanan.astype(np.int8))
    hit_miss_sudut   = cv2.morphologyEx(
        gambar_biner, cv2.MORPH_HITMISS,
        kernel_sudut.astype(np.int8))

    fig, sumbu = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle("Demo 4 – Hit-or-Miss Transform (Deteksi Pola Spesifik)",
                 fontsize=13, fontweight="bold")

    # Baris atas: citra asli dan tiga kernel (visualisasi heatmap)
    for i, (judul, kern) in enumerate([
        ("Kernel: Isolasi",     kernel_isolasi),
        ("Kernel: Ujung Kanan", kernel_ujung_kanan),
        ("Kernel: Sudut ↖",     kernel_sudut),
    ]):
        sumbu[0, i + 1].imshow(kern, cmap="bwr", vmin=-1, vmax=1)
        sumbu[0, i + 1].set_title(judul, fontsize=9)
        for r in range(3):
            for c in range(3):
                label = {1: "FG", -1: "BG", 0: "?"}[kern[r, c]]
                warna = "white" if kern[r, c] != 0 else "gray"
                sumbu[0, i + 1].text(c, r, label,
                                     ha="center", va="center",
                                     fontsize=10, color=warna, fontweight="bold")

    sumbu[0, 0].imshow(gambar_biner, cmap="gray")
    sumbu[0, 0].set_title("Citra Asli")

    # Baris bawah: hasil deteksi (overlay pada citra asli)
    for idx, (judul, hasil) in enumerate([
        ("Terdeteksi: Isolasi",     hit_miss_isolasi),
        ("Terdeteksi: Ujung Kanan", hit_miss_ujung),
        ("Terdeteksi: Sudut ↖",     hit_miss_sudut),
    ]):
        # Overlay: merah = titik yang terdeteksi
        overlay = cv2.cvtColor(gambar_biner, cv2.COLOR_GRAY2RGB)
        overlay[hasil > 0] = [255, 0, 0]
        sumbu[1, idx + 1].imshow(overlay)
        n_deteksi = int(np.count_nonzero(hasil))
        sumbu[1, idx + 1].set_title(f"{judul}\n({n_deteksi} titik)", fontsize=8)

    sumbu[1, 0].imshow(gambar_biner, cmap="gray")
    sumbu[1, 0].set_title("Citra Asli (referensi)")

    for ax in sumbu.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_demo4_hit_or_miss.png"), dpi=150)
    plt.show()
    print("  → Tersimpan: 16_demo4_hit_or_miss.png\n")


# ── Fungsi utama ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Fungsi utama yang menjalankan seluruh demo morfologi lanjut.
    """
    print("=" * 60)
    print("  Praktikum 16 – Operasi Morfologi Lanjut")
    print("=" * 60)
    print(f"  Output disimpan di : {OUTPUT_DIR}\n")

    # Muat atau buat citra biner sintetis
    gambar_biner = muat_atau_buat_gambar()
    print(f"  Ukuran citra  : {gambar_biner.shape[1]}×{gambar_biner.shape[0]} piksel")
    print(f"  Piksel putih  : {int(np.count_nonzero(gambar_biner))}\n")

    # Jalankan setiap demo secara berurutan
    demo_opening_closing(gambar_biner)
    demo_morphological_gradient(gambar_biner)
    demo_tophat_blackhat(gambar_biner)
    demo_hit_or_miss(gambar_biner)

    print("=" * 60)
    print("  Semua demo selesai.")
    print("=" * 60)


if __name__ == "__main__":
    main()
