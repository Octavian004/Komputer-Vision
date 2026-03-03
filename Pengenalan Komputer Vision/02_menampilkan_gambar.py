# ============================================================
# PROGRAM: 02_menampilkan_gambar.py
# PRAKTIKUM COMPUTER VISION - BAB 1: PENDAHULUAN
# ============================================================
# Deskripsi: Program ini mendemonstrasikan berbagai cara 
#            menampilkan gambar menggunakan OpenCV dan Matplotlib
# 
# Tujuan Pembelajaran:
#   1. Memahami perbedaan tampilan OpenCV vs Matplotlib
#   2. Memahami konversi BGR ke RGB
#   3. Mengenal berbagai opsi display gambar
# ============================================================

# ====================
# IMPORT LIBRARY
# ====================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# REFERENSI: Lihat CV2_FUNCTIONS_REFERENCE.py untuk dokumentasi lengkap cv2 functions


# ============================================================
# VARIABEL YANG BISA DIUBAH-UBAH (EKSPERIMEN)
# ============================================================

# 1. Nama file gambar
NAMA_FILE_GAMBAR = "input.jpg"

# 2. Ukuran figure matplotlib (lebar, tinggi) dalam inch
UKURAN_FIGURE = (12, 8)

# 3. Tampilkan axis (koordinat)?
TAMPILKAN_AXIS = True

# 4. Jumlah millisecond untuk tampilan OpenCV (0 = tunggu tombol)
WAKTU_TUNGGU_MS = 0

# ============================================================
# FUNGSI HELPER
# ============================================================

def dapatkan_path_gambar(nama_file):
    """Mendapatkan path lengkap file gambar"""
    direktori_script = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(direktori_script, "data", "images", nama_file)
    
    if not os.path.exists(path_data):
        path_data = os.path.join(direktori_script, nama_file)
    
    return path_data


def buat_gambar_contoh():
    """Membuat gambar contoh jika file tidak ditemukan"""
    gambar = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Buat gradien warna untuk demo BGR vs RGB
    for i in range(300):
        for j in range(400):
            # BGR: Blue meningkat horizontal, Green meningkat vertikal
            gambar[i, j] = [
                int(255 * j / 400),   # Blue
                int(255 * i / 300),   # Green
                100                    # Red tetap
            ]
    
    # Tambah teks
    cv2.putText(gambar, "BGR Demo Image", (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return gambar


# ============================================================
# FUNGSI TAMPILAN
# ============================================================

def tampilkan_dengan_opencv(gambar, judul="OpenCV Window"):
    """
    Menampilkan gambar menggunakan window OpenCV
    
    Kelebihan OpenCV:
    - Real-time display
    - Cocok untuk video/streaming
    - Interaktif (bisa zoom, scroll)
    
    Kekurangan:
    - Warna dalam format BGR
    - Sulit untuk multiple plots
    """
    print(f"\n[INFO] Menampilkan dengan OpenCV: {judul}")
    print("[INFO] Tekan tombol apapun untuk menutup window...")
    
    cv2.imshow(judul, gambar)
    cv2.waitKey(WAKTU_TUNGGU_MS)
    cv2.destroyAllWindows()


def tampilkan_dengan_matplotlib(gambar_bgr, judul="Matplotlib Display"):
    """
    Menampilkan gambar menggunakan Matplotlib
    
    PENTING: Matplotlib menggunakan format RGB, sedangkan OpenCV
    menggunakan BGR. Oleh karena itu, perlu konversi!
    
    Kelebihan Matplotlib:
    - Mudah untuk subplot/multiple images
    - Cocok untuk analisis dan presentasi
    - Bisa tambahkan axis, label, colorbar
    
    Kekurangan:
    - Tidak real-time
    - Perlu konversi BGR ke RGB
    """
    print(f"\n[INFO] Menampilkan dengan Matplotlib: {judul}")
    
    # KONVERSI BGR ke RGB
    # Ini SANGAT PENTING! Jika tidak, warna akan terbalik
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=UKURAN_FIGURE)
    plt.imshow(gambar_rgb)
    plt.title(judul)
    
    if not TAMPILKAN_AXIS:
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def perbandingan_bgr_rgb(gambar_bgr):
    """
    Menampilkan perbandingan tampilan BGR vs RGB
    untuk memahami pentingnya konversi warna
    """
    print("\n[INFO] Menampilkan perbandingan BGR vs RGB")
    
    # Konversi ke RGB
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    
    # Buat figure dengan 2 subplot
    fig, axes = plt.subplots(1, 2, figsize=UKURAN_FIGURE)
    
    # Subplot 1: Tampilkan BGR langsung (warna salah)
    axes[0].imshow(gambar_bgr)  # TANPA konversi - warna terbalik!
    axes[0].set_title("SALAH: BGR tanpa konversi\n(Warna terbalik!)", color='red')
    if not TAMPILKAN_AXIS:
        axes[0].axis('off')
    
    # Subplot 2: Tampilkan RGB (warna benar)
    axes[1].imshow(gambar_rgb)  # DENGAN konversi - warna benar
    axes[1].set_title("BENAR: Setelah konversi ke RGB\n(Warna sesuai)", color='green')
    if not TAMPILKAN_AXIS:
        axes[1].axis('off')
    
    plt.suptitle("Perbandingan BGR vs RGB dalam Matplotlib", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n[PENJELASAN]")
    print("OpenCV membaca gambar dalam format BGR (Blue, Green, Red)")
    print("Matplotlib menampilkan dalam format RGB (Red, Green, Blue)")
    print("Jika tidak dikonversi, channel Red dan Blue akan tertukar!")


def tampilkan_grayscale(gambar_bgr):
    """
    Menampilkan gambar dalam mode grayscale
    """
    print("\n[INFO] Menampilkan versi grayscale")
    
    # Konversi ke grayscale
    gambar_gray = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)
    
    # Buat figure
    fig, axes = plt.subplots(1, 2, figsize=UKURAN_FIGURE)
    
    # Gambar asli
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(gambar_rgb)
    axes[0].set_title("Gambar Asli (RGB)")
    
    # Gambar grayscale
    # Gunakan cmap='gray' untuk tampilan grayscale yang benar
    axes[1].imshow(gambar_gray, cmap='gray')
    axes[1].set_title("Gambar Grayscale")
    
    if not TAMPILKAN_AXIS:
        axes[0].axis('off')
        axes[1].axis('off')
    
    plt.suptitle("Perbandingan RGB vs Grayscale", fontsize=14)
    plt.tight_layout()
    plt.show()


def tampilkan_channel_terpisah(gambar_bgr):
    """
    Menampilkan setiap channel warna secara terpisah
    """
    print("\n[INFO] Menampilkan channel warna terpisah")
    
    # Split channel
    b, g, r = cv2.split(gambar_bgr)
    
    # Buat figure dengan 4 subplot
    fig, axes = plt.subplots(2, 2, figsize=UKURAN_FIGURE)
    
    # Gambar asli
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(gambar_rgb)
    axes[0, 0].set_title("Gambar Asli (RGB)")
    
    # Channel Merah (Red)
    axes[0, 1].imshow(r, cmap='Reds')
    axes[0, 1].set_title("Channel MERAH (Red)")
    
    # Channel Hijau (Green)
    axes[1, 0].imshow(g, cmap='Greens')
    axes[1, 0].set_title("Channel HIJAU (Green)")
    
    # Channel Biru (Blue)
    axes[1, 1].imshow(b, cmap='Blues')
    axes[1, 1].set_title("Channel BIRU (Blue)")
    
    for ax in axes.flat:
        if not TAMPILKAN_AXIS:
            ax.axis('off')
    
    plt.suptitle("Visualisasi Setiap Channel Warna", fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    """Fungsi utama program"""
    
    print("\n" + "=" * 60)
    print("PRAKTIKUM 2: MENAMPILKAN GAMBAR")
    print("=" * 60)
    
    # Muat gambar
    path_gambar = dapatkan_path_gambar(NAMA_FILE_GAMBAR)
    gambar = cv2.imread(path_gambar)
    
    if gambar is None:
        print(f"[WARNING] Gambar {NAMA_FILE_GAMBAR} tidak ditemukan")
        print("[INFO] Menggunakan gambar contoh...")
        gambar = buat_gambar_contoh()
    else:
        print(f"[SUKSES] Gambar dimuat: {path_gambar}")
    
    # Menu demonstrasi
    print("\n" + "-" * 60)
    print("DEMONSTRASI CARA MENAMPILKAN GAMBAR")
    print("-" * 60)
    
    # 1. Tampilkan dengan OpenCV
    print("\n1. TAMPILAN OPENCV")
    print("   OpenCV menampilkan gambar dalam format BGR secara native")
    tampilkan_dengan_opencv(gambar, "1. Tampilan OpenCV (BGR native)")
    
    # 2. Perbandingan BGR vs RGB di Matplotlib
    print("\n2. PERBANDINGAN BGR vs RGB")
    print("   Lihat perbedaan jika tidak melakukan konversi!")
    perbandingan_bgr_rgb(gambar)
    
    # 3. Tampilan Grayscale
    print("\n3. TAMPILAN GRAYSCALE")
    print("   Konversi gambar berwarna ke grayscale")
    tampilkan_grayscale(gambar)
    
    # 4. Channel terpisah
    print("\n4. CHANNEL WARNA TERPISAH")
    print("   Visualisasi setiap channel R, G, B")
    tampilkan_channel_terpisah(gambar)
    
    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN")
    print("=" * 60)
    print("""
POIN PENTING:
1. OpenCV membaca gambar dalam format BGR, bukan RGB
2. Matplotlib menggunakan format RGB
3. Selalu konversi BGR→RGB sebelum tampilkan di Matplotlib
4. Gunakan cv2.cvtColor(img, cv2.COLOR_BGR2RGB) untuk konversi
5. Untuk grayscale, gunakan cmap='gray' di matplotlib

EKSPERIMEN YANG BISA DICOBA:
- Ubah TAMPILKAN_AXIS = False untuk hilangkan axis
- Ubah UKURAN_FIGURE untuk resize tampilan
- Coba gambar dengan warna dominan berbeda
""")


# Jalankan program utama
if __name__ == "__main__":
    main()
