# ============================================================
# PROGRAM: 03_properti_gambar.py
# PRAKTIKUM COMPUTER VISION - BAB 1: PENDAHULUAN
# ============================================================
# Deskripsi: Program ini mendemonstrasikan cara membaca dan
#            memahami properti-properti dari gambar digital
# 
# Tujuan Pembelajaran:
#   1. Memahami struktur data gambar (numpy array)
#   2. Mengenal properti: shape, dtype, size
#   3. Memahami representasi piksel dalam memori
# ============================================================

# ====================
# IMPORT LIBRARY
# ====================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# VARIABEL YANG BISA DIUBAH-UBAH (EKSPERIMEN)
# ============================================================

# 1. File gambar yang akan dianalisis
NAMA_FILE_GAMBAR = "building.jpg"

# 2. Tampilkan histogram?
TAMPILKAN_HISTOGRAM = True

# 3. Tampilkan contoh nilai piksel?
TAMPILKAN_SAMPLE_PIKSEL = True

# 4. Region piksel yang akan ditampilkan (x_start, y_start, ukuran)
REGION_SAMPLE = (100, 100, 5)  # Mulai dari (100,100), ambil 5x5 piksel

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
    """Membuat gambar contoh dengan berbagai warna"""
    gambar = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Bagi menjadi 4 region dengan warna berbeda
    gambar[0:150, 0:200] = [255, 0, 0]      # Biru (BGR)
    gambar[0:150, 200:400] = [0, 255, 0]    # Hijau
    gambar[150:300, 0:200] = [0, 0, 255]    # Merah
    gambar[150:300, 200:400] = [128, 128, 128]  # Abu-abu
    
    # Tambah gradien di tengah
    cv2.circle(gambar, (200, 150), 50, (255, 255, 255), -1)
    
    return gambar


# ============================================================
# FUNGSI ANALISIS PROPERTI
# ============================================================

def analisis_properti_dasar(gambar, nama_file=""):
    """
    Menganalisis dan menampilkan properti dasar gambar
    
    Parameter:
        gambar (numpy.ndarray): Array gambar
        nama_file (str): Nama file untuk referensi
    """
    print("\n" + "=" * 60)
    print("ANALISIS PROPERTI GAMBAR")
    print("=" * 60)
    
    if nama_file:
        print(f"\nFile: {nama_file}")
    
    # 1. TIPE DATA
    print("\n[1] TIPE DATA")
    print("-" * 40)
    print(f"    Tipe Python : {type(gambar)}")
    print(f"    Tipe NumPy  : {gambar.dtype}")
    print(f"    Penjelasan  : uint8 = unsigned integer 8-bit (0-255)")
    
    # 2. DIMENSI (SHAPE)
    print("\n[2] DIMENSI (SHAPE)")
    print("-" * 40)
    print(f"    Shape       : {gambar.shape}")
    
    if len(gambar.shape) == 3:
        tinggi, lebar, channel = gambar.shape
        print(f"    Tinggi      : {tinggi} piksel")
        print(f"    Lebar       : {lebar} piksel")
        print(f"    Channel     : {channel}")
        
        if channel == 3:
            print(f"    Format      : BGR (Blue, Green, Red)")
        elif channel == 4:
            print(f"    Format      : BGRA (dengan Alpha channel)")
    else:
        tinggi, lebar = gambar.shape
        print(f"    Tinggi      : {tinggi} piksel")
        print(f"    Lebar       : {lebar} piksel")
        print(f"    Channel     : 1 (Grayscale)")
    
    # 3. UKURAN
    print("\n[3] UKURAN")
    print("-" * 40)
    
    # Total elemen
    total_elemen = gambar.size
    print(f"    Total elemen: {total_elemen:,}")
    
    # Total piksel
    total_piksel = gambar.shape[0] * gambar.shape[1]
    print(f"    Total piksel: {total_piksel:,}")
    
    # Ukuran memori
    ukuran_bytes = gambar.nbytes
    print(f"    Ukuran RAM  : {ukuran_bytes:,} bytes")
    print(f"                : {ukuran_bytes/1024:.2f} KB")
    print(f"                : {ukuran_bytes/1024/1024:.4f} MB")
    
    # Kalkulasi teori
    print(f"\n    Kalkulasi Teori:")
    print(f"    {tinggi} x {lebar} x {len(gambar.shape) == 3 and gambar.shape[2] or 1} x 1 byte = {ukuran_bytes:,} bytes")
    
    # 4. STATISTIK NILAI PIKSEL
    print("\n[4] STATISTIK NILAI PIKSEL")
    print("-" * 40)
    print(f"    Minimum     : {gambar.min()}")
    print(f"    Maksimum    : {gambar.max()}")
    print(f"    Rata-rata   : {gambar.mean():.2f}")
    print(f"    Std Deviasi : {gambar.std():.2f}")
    
    # Statistik per channel (jika berwarna)
    if len(gambar.shape) == 3:
        print("\n    Per Channel:")
        channel_names = ['Blue', 'Green', 'Red']
        for i, name in enumerate(channel_names[:gambar.shape[2]]):
            ch = gambar[:, :, i]
            print(f"    - {name:6}: min={ch.min():3}, max={ch.max():3}, mean={ch.mean():.1f}")
    
    return {
        'tinggi': tinggi,
        'lebar': lebar,
        'channel': gambar.shape[2] if len(gambar.shape) == 3 else 1,
        'total_piksel': total_piksel,
        'ukuran_bytes': ukuran_bytes
    }


def tampilkan_sample_piksel(gambar, x_start, y_start, ukuran=5):
    """
    Menampilkan nilai piksel dari region tertentu
    
    Berguna untuk memahami bagaimana data piksel disimpan
    """
    print("\n" + "=" * 60)
    print(f"SAMPLE NILAI PIKSEL (Region {ukuran}x{ukuran} dari ({x_start},{y_start}))")
    print("=" * 60)
    
    # Pastikan region tidak keluar batas
    y_end = min(y_start + ukuran, gambar.shape[0])
    x_end = min(x_start + ukuran, gambar.shape[1])
    
    # Ambil region
    region = gambar[y_start:y_end, x_start:x_end]
    
    if len(gambar.shape) == 3:
        print("\nFormat: [B, G, R]")
        print("-" * 60)
        
        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                piksel = region[i, j]
                print(f"[{piksel[0]:3},{piksel[1]:3},{piksel[2]:3}] ", end="")
            print()
    else:
        print("\nNilai Grayscale (0-255):")
        print("-" * 40)
        print(region)
    
    print("\n[PENJELASAN]")
    print("- Setiap piksel memiliki nilai 0-255 per channel")
    print("- 0 = hitam/tidak ada warna, 255 = terang/warna penuh")
    print("- Pada gambar BGR: [Blue, Green, Red]")


def tampilkan_histogram(gambar):
    """
    Menampilkan histogram distribusi warna
    
    Histogram menunjukkan distribusi intensitas piksel
    """
    print("\n[INFO] Menampilkan histogram distribusi warna...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gambar asli
    gambar_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB) if len(gambar.shape) == 3 else gambar
    axes[0, 0].imshow(gambar_rgb if len(gambar.shape) == 3 else gambar, cmap='gray' if len(gambar.shape) == 2 else None)
    axes[0, 0].set_title("Gambar Asli")
    axes[0, 0].axis('off')
    
    # Histogram per channel
    if len(gambar.shape) == 3:
        colors = ('b', 'g', 'r')
        channel_names = ('Blue', 'Green', 'Red')
        
        for i, (col, name) in enumerate(zip(colors, channel_names)):
            hist = cv2.calcHist([gambar], [i], None, [256], [0, 256])
            axes[0, 1].plot(hist, color=col, label=name)
        
        axes[0, 1].set_title("Histogram per Channel")
        axes[0, 1].set_xlabel("Intensitas (0-255)")
        axes[0, 1].set_ylabel("Jumlah Piksel")
        axes[0, 1].legend()
        axes[0, 1].set_xlim([0, 256])
        
        # Histogram gabungan (grayscale)
        gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    else:
        gray = gambar
    
    # Histogram grayscale
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    axes[1, 0].bar(range(256), hist_gray.flatten(), color='gray', width=1)
    axes[1, 0].set_title("Histogram Grayscale")
    axes[1, 0].set_xlabel("Intensitas (0-255)")
    axes[1, 0].set_ylabel("Jumlah Piksel")
    
    # Cumulative histogram
    cumsum = np.cumsum(hist_gray)
    axes[1, 1].plot(cumsum, color='orange')
    axes[1, 1].set_title("Cumulative Histogram")
    axes[1, 1].set_xlabel("Intensitas (0-255)")
    axes[1, 1].set_ylabel("Jumlah Kumulatif")
    
    plt.suptitle("Analisis Histogram Gambar", fontsize=14)
    plt.tight_layout()
    plt.show()


def bandingkan_mode_baca(path_gambar):
    """
    Membandingkan gambar yang dibaca dengan mode berbeda
    """
    print("\n" + "=" * 60)
    print("PERBANDINGAN MODE PEMBACAAN GAMBAR")
    print("=" * 60)
    
    # Baca dengan berbagai mode
    img_color = cv2.imread(path_gambar, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(path_gambar, cv2.IMREAD_GRAYSCALE)
    img_unchanged = cv2.imread(path_gambar, cv2.IMREAD_UNCHANGED)
    
    if img_color is not None:
        print("\n[1] IMREAD_COLOR (default)")
        print(f"    Shape: {img_color.shape}")
        print(f"    Size : {img_color.nbytes:,} bytes")
        
        print("\n[2] IMREAD_GRAYSCALE")
        print(f"    Shape: {img_gray.shape}")
        print(f"    Size : {img_gray.nbytes:,} bytes")
        
        print("\n[3] IMREAD_UNCHANGED")
        print(f"    Shape: {img_unchanged.shape}")
        print(f"    Size : {img_unchanged.nbytes:,} bytes")
        
        # Tampilkan perbandingan visual
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"COLOR\nShape: {img_color.shape}")
        axes[0].axis('off')
        
        axes[1].imshow(img_gray, cmap='gray')
        axes[1].set_title(f"GRAYSCALE\nShape: {img_gray.shape}")
        axes[1].axis('off')
        
        # Untuk UNCHANGED, cek apakah ada alpha channel
        if len(img_unchanged.shape) == 3 and img_unchanged.shape[2] == 4:
            # Ada alpha channel
            axes[2].imshow(cv2.cvtColor(img_unchanged, cv2.COLOR_BGRA2RGBA))
        else:
            axes[2].imshow(cv2.cvtColor(img_unchanged, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"UNCHANGED\nShape: {img_unchanged.shape}")
        axes[2].axis('off')
        
        plt.suptitle("Perbandingan Mode Pembacaan Gambar", fontsize=14)
        plt.tight_layout()
        plt.show()


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    """Fungsi utama program"""
    
    print("\n" + "=" * 60)
    print("PRAKTIKUM 3: PROPERTI GAMBAR")
    print("=" * 60)
    
    # Muat gambar
    path_gambar = dapatkan_path_gambar(NAMA_FILE_GAMBAR)
    gambar = cv2.imread(path_gambar)
    
    if gambar is None:
        print(f"[WARNING] Gambar {NAMA_FILE_GAMBAR} tidak ditemukan")
        print("[INFO] Menggunakan gambar contoh...")
        gambar = buat_gambar_contoh()
        path_gambar = "gambar_contoh"
    else:
        print(f"[SUKSES] Gambar dimuat: {path_gambar}")
    
    # 1. Analisis properti dasar
    properti = analisis_properti_dasar(gambar, path_gambar)
    
    # 2. Tampilkan sample piksel
    if TAMPILKAN_SAMPLE_PIKSEL:
        x, y, size = REGION_SAMPLE
        tampilkan_sample_piksel(gambar, x, y, size)
    
    # 3. Tampilkan histogram
    if TAMPILKAN_HISTOGRAM:
        tampilkan_histogram(gambar)
    
    # 4. Bandingkan mode baca
    if os.path.exists(path_gambar):
        bandingkan_mode_baca(path_gambar)
    
    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)
    print(f"""
PROPERTI GAMBAR YANG DIPELAJARI:
1. Shape    : Dimensi gambar (tinggi, lebar, channel)
2. Dtype    : Tipe data (biasanya uint8 = 0-255)
3. Size     : Total elemen dalam array
4. Nbytes   : Ukuran memori yang digunakan

RUMUS PENTING:
- Total piksel = tinggi × lebar
- Ukuran memori = tinggi × lebar × channel × bytes_per_value
- Untuk uint8: 1 value = 1 byte

EKSPERIMEN YANG BISA DICOBA:
- Ubah REGION_SAMPLE untuk melihat piksel di posisi berbeda
- Bandingkan ukuran file vs ukuran memori
- Analisis histogram gambar terang vs gelap
""")


# Jalankan program utama
if __name__ == "__main__":
    main()
