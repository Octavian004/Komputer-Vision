# ============================================================
# PROGRAM: 01_loading_gambar.py
# PRAKTIKUM COMPUTER VISION - BAB 1: PENDAHULUAN
# ============================================================
# Deskripsi: Program ini mendemonstrasikan cara memuat (loading)
#            gambar dari file menggunakan OpenCV
# 
# Tujuan Pembelajaran:
#   1. Memahami cara membaca file gambar dengan cv2.imread()
#   2. Mengenal berbagai mode pembacaan gambar
#   3. Memahami pengecekan keberhasilan loading gambar
# ============================================================

# ====================
# IMPORT LIBRARY
# ====================
import cv2          # Library utama untuk computer vision
import numpy as np  # Library untuk operasi array/matriks
import os           # Library untuk operasi file dan folder
import sys          # Library untuk system operations
# REFERENSI: Lihat CV2_FUNCTIONS_REFERENCE.py untuk dokumentasi lengkap cv2 functions


# ============================================================
# VARIABEL YANG BISA DIUBAH-UBAH (EKSPERIMEN)
# ============================================================
# Coba ubah variabel-variabel di bawah ini untuk melihat efeknya!

# 1. Nama file gambar yang akan dibaca
#    - Coba ganti dengan gambar lain yang ada di folder data/images
#    - Pastikan path-nya benar
NAMA_FILE_GAMBAR = "input.jpg"

# 2. Mode pembacaan gambar
#    - cv2.IMREAD_COLOR (1)     : Baca sebagai gambar berwarna (BGR)
#    - cv2.IMREAD_GRAYSCALE (0) : Baca sebagai gambar grayscale
#    - cv2.IMREAD_UNCHANGED (-1): Baca termasuk alpha channel (jika ada)
MODE_BACA = cv2.IMREAD_COLOR

# 3. Tampilkan informasi detail?
TAMPILKAN_INFO_DETAIL = True

# ============================================================
# FUNGSI UTAMA
# ============================================================

def dapatkan_path_gambar(nama_file):
    """
    Fungsi untuk mendapatkan path lengkap file gambar
    
    Parameter:
        nama_file (str): Nama file gambar
        
    Return:
        str: Path lengkap ke file gambar
    """
    # Dapatkan direktori tempat script ini berada
    direktori_script = os.path.dirname(os.path.abspath(__file__))
    
    # Path ke folder data/images
    path_data = os.path.join(direktori_script, "data", "images", nama_file)
    
    # Jika tidak ditemukan, coba di folder yang sama dengan script
    if not os.path.exists(path_data):
        path_data = os.path.join(direktori_script, nama_file)
    
    return path_data


def muat_gambar(path_gambar, mode=cv2.IMREAD_COLOR):
    """
    Fungsi untuk memuat gambar dari file
    
    Parameter:
        path_gambar (str): Path lengkap ke file gambar
        mode (int): Mode pembacaan gambar
                    - cv2.IMREAD_COLOR (1): BGR
                    - cv2.IMREAD_GRAYSCALE (0): Grayscale
                    - cv2.IMREAD_UNCHANGED (-1): Termasuk alpha
    
    Return:
        numpy.ndarray atau None: Array gambar jika berhasil, None jika gagal
    """
    print("=" * 60)
    print("PROSES LOADING GAMBAR")
    print("=" * 60)
    
    # Cek apakah file ada
    if not os.path.exists(path_gambar):
        print(f"[ERROR] File tidak ditemukan: {path_gambar}")
        print("[INFO] Pastikan file gambar ada di folder yang benar")
        return None
    
    print(f"[INFO] Membaca file: {path_gambar}")
    print(f"[INFO] Mode baca: {mode}")
    
    # Baca gambar menggunakan cv2.imread()
    # Fungsi ini mengembalikan numpy array atau None jika gagal
    gambar = cv2.imread(path_gambar, mode)
    
    # Cek apakah gambar berhasil dibaca
    if gambar is None:
        print("[ERROR] Gagal membaca gambar!")
        print("[INFO] Kemungkinan penyebab:")
        print("       - Format file tidak didukung")
        print("       - File rusak/corrupt")
        return None
    
    print("[SUKSES] Gambar berhasil dimuat!")
    
    return gambar


def tampilkan_info_gambar(gambar, nama="Gambar"):
    """
    Fungsi untuk menampilkan informasi detail tentang gambar
    
    Parameter:
        gambar (numpy.ndarray): Array gambar
        nama (str): Nama gambar untuk label
    """
    print("\n" + "=" * 60)
    print(f"INFORMASI {nama.upper()}")
    print("=" * 60)
    
    # Tipe data numpy array
    print(f"Tipe data     : {type(gambar)}")
    print(f"Dtype array   : {gambar.dtype}")
    
    # Dimensi gambar
    # Shape format: (height, width, channels) atau (height, width) untuk grayscale
    print(f"Shape (dimensi): {gambar.shape}")
    
    if len(gambar.shape) == 3:
        tinggi, lebar, channel = gambar.shape
        print(f"  - Tinggi    : {tinggi} piksel")
        print(f"  - Lebar     : {lebar} piksel")
        print(f"  - Channel   : {channel} (BGR)" if channel == 3 else f"  - Channel   : {channel}")
    else:
        tinggi, lebar = gambar.shape
        print(f"  - Tinggi    : {tinggi} piksel")
        print(f"  - Lebar     : {lebar} piksel")
        print(f"  - Channel   : 1 (Grayscale)")
    
    # Statistik nilai piksel
    print(f"\nStatistik Piksel:")
    print(f"  - Nilai minimum : {gambar.min()}")
    print(f"  - Nilai maksimum: {gambar.max()}")
    print(f"  - Nilai rata-rata: {gambar.mean():.2f}")
    
    # Ukuran memori
    ukuran_bytes = gambar.nbytes
    ukuran_kb = ukuran_bytes / 1024
    ukuran_mb = ukuran_kb / 1024
    print(f"\nUkuran di Memori:")
    print(f"  - {ukuran_bytes:,} bytes")
    print(f"  - {ukuran_kb:.2f} KB")
    print(f"  - {ukuran_mb:.4f} MB")


def tampilkan_gambar(gambar, judul="Gambar"):
    """
    Fungsi untuk menampilkan gambar di window
    
    Parameter:
        gambar (numpy.ndarray): Array gambar
        judul (str): Judul window
    """
    print(f"\n[INFO] Menampilkan gambar '{judul}'")
    print("[INFO] Tekan tombol apapun untuk menutup window...")
    
    # Tampilkan gambar di window
    cv2.imshow(judul, gambar)
    
    # Tunggu sampai user menekan tombol
    # 0 berarti tunggu tanpa batas waktu
    cv2.waitKey(0)
    
    # Tutup semua window
    cv2.destroyAllWindows()


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    """Fungsi utama program"""
    
    print("\n" + "=" * 60)
    print("PRAKTIKUM 1: LOADING GAMBAR DENGAN OPENCV")
    print("=" * 60)
    
    # Dapatkan path gambar
    path_gambar = dapatkan_path_gambar(NAMA_FILE_GAMBAR)
    
    # Muat gambar
    gambar = muat_gambar(path_gambar, MODE_BACA)
    
    # Jika gagal, hentikan program
    if gambar is None:
        print("\n[INFO] Program dihentikan karena gambar tidak dapat dimuat")
        print("[INFO] Membuat gambar contoh...")
        
        # Buat gambar contoh jika file tidak ditemukan
        gambar = np.zeros((300, 400, 3), dtype=np.uint8)
        gambar[:] = (100, 150, 200)  # Warna BGR
        cv2.putText(gambar, "Gambar Contoh", (80, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(gambar, "(File asli tidak ditemukan)", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Tampilkan informasi jika diaktifkan
    if TAMPILKAN_INFO_DETAIL:
        tampilkan_info_gambar(gambar, "Gambar yang Dimuat")
    
    # Tampilkan gambar
    tampilkan_gambar(gambar, f"Loading Gambar - {NAMA_FILE_GAMBAR}")
    
    print("\n" + "=" * 60)
    print("PERCOBAAN SELESAI!")
    print("=" * 60)
    print("\nEKSPERIMEN YANG BISA DICOBA:")
    print("1. Ubah MODE_BACA menjadi cv2.IMREAD_GRAYSCALE")
    print("2. Coba loading gambar format berbeda (.png, .bmp)")
    print("3. Perhatikan perbedaan shape pada mode berbeda")


# Jalankan program utama
if __name__ == "__main__":
    main()
