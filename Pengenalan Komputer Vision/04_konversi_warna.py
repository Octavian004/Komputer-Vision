# ============================================================
# PROGRAM: 04_konversi_warna.py
# PRAKTIKUM COMPUTER VISION - BAB 1: PENDAHULUAN
# ============================================================
# Deskripsi: Program ini mendemonstrasikan konversi antar
#            color space (ruang warna) dalam computer vision
# 
# Tujuan Pembelajaran:
#   1. Memahami berbagai color space (BGR, RGB, HSV, Grayscale, LAB)
#   2. Menguasai fungsi cv2.cvtColor()
#   3. Memahami kapan menggunakan color space tertentu
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

# 1. File gambar yang akan diproses
NAMA_FILE_GAMBAR = "building.jpg"

# 2. Tampilkan detail setiap color space?
TAMPILKAN_DETAIL = True

# 3. Threshold untuk demo deteksi warna (HSV)
#    Coba ubah nilai ini untuk mendeteksi warna berbeda!
#    Format: [H_min, S_min, V_min], [H_max, S_max, V_max]
#    H: Hue (0-179), S: Saturation (0-255), V: Value (0-255)

# Preset warna:
WARNA_MERAH_BAWAH = np.array([0, 100, 100])
WARNA_MERAH_ATAS = np.array([10, 255, 255])

WARNA_HIJAU_BAWAH = np.array([35, 100, 100])
WARNA_HIJAU_ATAS = np.array([85, 255, 255])

WARNA_BIRU_BAWAH = np.array([100, 100, 100])
WARNA_BIRU_ATAS = np.array([130, 255, 255])

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


def buat_gambar_berwarna():
    """Membuat gambar contoh dengan berbagai warna untuk demo"""
    gambar = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Berbagai warna dalam BGR
    cv2.rectangle(gambar, (0, 0), (100, 150), (255, 0, 0), -1)      # Biru
    cv2.rectangle(gambar, (100, 0), (200, 150), (0, 255, 0), -1)    # Hijau
    cv2.rectangle(gambar, (200, 0), (300, 150), (0, 0, 255), -1)    # Merah
    cv2.rectangle(gambar, (300, 0), (400, 150), (0, 255, 255), -1)  # Kuning
    
    cv2.rectangle(gambar, (0, 150), (100, 300), (255, 0, 255), -1)   # Magenta
    cv2.rectangle(gambar, (100, 150), (200, 300), (255, 255, 0), -1) # Cyan
    cv2.rectangle(gambar, (200, 150), (300, 300), (255, 255, 255), -1) # Putih
    cv2.rectangle(gambar, (300, 150), (400, 300), (128, 128, 128), -1) # Abu-abu
    
    return gambar


# ============================================================
# FUNGSI KONVERSI COLOR SPACE
# ============================================================

def demonstrasi_konversi_dasar(gambar_bgr):
    """
    Mendemonstrasikan konversi ke berbagai color space dasar
    """
    print("\n" + "=" * 60)
    print("KONVERSI COLOR SPACE DASAR")
    print("=" * 60)
    
    # 1. BGR ke RGB
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    print("\n[1] BGR → RGB")
    print("    Digunakan untuk: Tampilan di Matplotlib, PIL")
    
    # 2. BGR ke Grayscale
    gambar_gray = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)
    print("\n[2] BGR → Grayscale")
    print("    Digunakan untuk: Edge detection, analisis bentuk")
    
    # 3. BGR ke HSV
    gambar_hsv = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2HSV)
    print("\n[3] BGR → HSV")
    print("    Digunakan untuk: Deteksi warna, segmentasi berbasis warna")
    
    # 4. BGR ke LAB
    gambar_lab = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2LAB)
    print("\n[4] BGR → LAB")
    print("    Digunakan untuk: Koreksi warna, color transfer")
    
    # Tampilkan semua
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Baris 1
    axes[0, 0].imshow(gambar_rgb)
    axes[0, 0].set_title("RGB (dari BGR)")
    
    axes[0, 1].imshow(gambar_gray, cmap='gray')
    axes[0, 1].set_title("Grayscale")
    
    axes[0, 2].imshow(gambar_hsv)
    axes[0, 2].set_title("HSV (tampilan langsung)")
    
    # Baris 2 - Channel terpisah HSV
    h, s, v = cv2.split(gambar_hsv)
    
    axes[1, 0].imshow(h, cmap='hsv')
    axes[1, 0].set_title("Hue (H) - 0-179")
    
    axes[1, 1].imshow(s, cmap='gray')
    axes[1, 1].set_title("Saturation (S) - 0-255")
    
    axes[1, 2].imshow(v, cmap='gray')
    axes[1, 2].set_title("Value (V) - 0-255")
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.suptitle("Perbandingan Berbagai Color Space", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return gambar_rgb, gambar_gray, gambar_hsv, gambar_lab


def jelaskan_hsv():
    """
    Menjelaskan color space HSV dengan visualisasi
    """
    print("\n" + "=" * 60)
    print("PENJELASAN HSV (Hue, Saturation, Value)")
    print("=" * 60)
    
    print("""
    HSV adalah color space yang lebih intuitif untuk analisis warna:
    
    H (Hue) - "Warna apa?"
    ├── 0-10    : Merah
    ├── 11-25   : Oranye
    ├── 26-34   : Kuning
    ├── 35-85   : Hijau
    ├── 86-125  : Cyan/Biru Muda
    ├── 126-155 : Biru
    ├── 156-175 : Ungu/Magenta
    └── 176-179 : Merah (kembali)
    
    S (Saturation) - "Seberapa pekat?"
    ├── 0       : Abu-abu (tidak ada warna)
    └── 255     : Warna murni (sangat pekat)
    
    V (Value) - "Seberapa terang?"
    ├── 0       : Hitam (tidak ada cahaya)
    └── 255     : Terang maksimal
    """)
    
    # Buat visualisasi HSV
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Hue wheel
    hue_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(180):
        hue_image[:, int(i*256/180):int((i+1)*256/180)] = [i, 255, 255]
    hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2RGB)
    axes[0].imshow(hue_image)
    axes[0].set_title("Variasi Hue (H)\n0 → 179")
    axes[0].set_xlabel("Hue Value")
    axes[0].set_yticks([])
    
    # Saturation gradient
    sat_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        sat_image[:, i] = [90, i, 255]  # Hue=90 (Hijau)
    sat_image = cv2.cvtColor(sat_image, cv2.COLOR_HSV2RGB)
    axes[1].imshow(sat_image)
    axes[1].set_title("Variasi Saturation (S)\n0 → 255")
    axes[1].set_xlabel("Saturation Value")
    axes[1].set_yticks([])
    
    # Value gradient
    val_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        val_image[:, i] = [90, 255, i]  # Hue=90 (Hijau)
    val_image = cv2.cvtColor(val_image, cv2.COLOR_HSV2RGB)
    axes[2].imshow(val_image)
    axes[2].set_title("Variasi Value (V)\n0 → 255")
    axes[2].set_xlabel("Value")
    axes[2].set_yticks([])
    
    plt.suptitle("Visualisasi Komponen HSV", fontsize=14)
    plt.tight_layout()
    plt.show()


def deteksi_warna_dengan_hsv(gambar_bgr, warna_bawah, warna_atas, nama_warna=""):
    """
    Mendemonstrasikan deteksi warna menggunakan HSV thresholding
    
    Ini adalah salah satu aplikasi paling umum dari konversi HSV!
    """
    print(f"\n[INFO] Deteksi warna: {nama_warna}")
    print(f"       Range HSV: {warna_bawah} - {warna_atas}")
    
    # Konversi ke HSV
    gambar_hsv = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2HSV)
    
    # Buat mask berdasarkan range warna
    mask = cv2.inRange(gambar_hsv, warna_bawah, warna_atas)
    
    # Aplikasikan mask ke gambar asli
    hasil = cv2.bitwise_and(gambar_bgr, gambar_bgr, mask=mask)
    
    return mask, hasil


def demo_deteksi_warna(gambar_bgr):
    """
    Demo deteksi berbagai warna menggunakan HSV
    """
    print("\n" + "=" * 60)
    print("DEMO DETEKSI WARNA DENGAN HSV")
    print("=" * 60)
    
    # Deteksi berbagai warna
    mask_merah, hasil_merah = deteksi_warna_dengan_hsv(
        gambar_bgr, WARNA_MERAH_BAWAH, WARNA_MERAH_ATAS, "MERAH"
    )
    
    mask_hijau, hasil_hijau = deteksi_warna_dengan_hsv(
        gambar_bgr, WARNA_HIJAU_BAWAH, WARNA_HIJAU_ATAS, "HIJAU"
    )
    
    mask_biru, hasil_biru = deteksi_warna_dengan_hsv(
        gambar_bgr, WARNA_BIRU_BAWAH, WARNA_BIRU_ATAS, "BIRU"
    )
    
    # Tampilkan hasil
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    
    # Baris 1: Deteksi Merah
    axes[0, 0].imshow(gambar_rgb)
    axes[0, 0].set_title("Gambar Asli")
    axes[0, 1].imshow(mask_merah, cmap='gray')
    axes[0, 1].set_title("Mask MERAH")
    axes[0, 2].imshow(cv2.cvtColor(hasil_merah, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Hasil Deteksi MERAH")
    
    # Baris 2: Deteksi Hijau
    axes[1, 0].imshow(gambar_rgb)
    axes[1, 0].set_title("Gambar Asli")
    axes[1, 1].imshow(mask_hijau, cmap='gray')
    axes[1, 1].set_title("Mask HIJAU")
    axes[1, 2].imshow(cv2.cvtColor(hasil_hijau, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Hasil Deteksi HIJAU")
    
    # Baris 3: Deteksi Biru
    axes[2, 0].imshow(gambar_rgb)
    axes[2, 0].set_title("Gambar Asli")
    axes[2, 1].imshow(mask_biru, cmap='gray')
    axes[2, 1].set_title("Mask BIRU")
    axes[2, 2].imshow(cv2.cvtColor(hasil_biru, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title("Hasil Deteksi BIRU")
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.suptitle("Deteksi Warna dengan HSV Thresholding", fontsize=14)
    plt.tight_layout()
    plt.show()


def konversi_grayscale_berbagai_metode(gambar_bgr):
    """
    Membandingkan berbagai metode konversi ke grayscale
    """
    print("\n" + "=" * 60)
    print("PERBANDINGAN METODE KONVERSI GRAYSCALE")
    print("=" * 60)
    
    # Metode 1: cv2.cvtColor (default)
    gray_cv2 = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)
    print("\n[1] cv2.COLOR_BGR2GRAY")
    print("    Formula: 0.299*R + 0.587*G + 0.114*B")
    
    # Metode 2: Rata-rata channel
    gray_mean = np.mean(gambar_bgr, axis=2).astype(np.uint8)
    print("\n[2] Rata-rata Channel")
    print("    Formula: (R + G + B) / 3")
    
    # Metode 3: Mengambil channel tertentu
    b, g, r = cv2.split(gambar_bgr)
    print("\n[3] Channel Individu")
    print("    Mengambil hanya 1 channel")
    
    # Tampilkan perbandingan
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    
    axes[0, 0].imshow(gambar_rgb)
    axes[0, 0].set_title("Gambar Asli (RGB)")
    
    axes[0, 1].imshow(gray_cv2, cmap='gray')
    axes[0, 1].set_title("Grayscale (cv2.cvtColor)\nLuminosity method")
    
    axes[0, 2].imshow(gray_mean, cmap='gray')
    axes[0, 2].set_title("Grayscale (Rata-rata)\nAverage method")
    
    axes[1, 0].imshow(r, cmap='gray')
    axes[1, 0].set_title("Channel Red saja")
    
    axes[1, 1].imshow(g, cmap='gray')
    axes[1, 1].set_title("Channel Green saja")
    
    axes[1, 2].imshow(b, cmap='gray')
    axes[1, 2].set_title("Channel Blue saja")
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.suptitle("Perbandingan Metode Konversi Grayscale", fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    """Fungsi utama program"""
    
    print("\n" + "=" * 60)
    print("PRAKTIKUM 4: KONVERSI COLOR SPACE")
    print("=" * 60)
    
    # Muat gambar
    path_gambar = dapatkan_path_gambar(NAMA_FILE_GAMBAR)
    gambar = cv2.imread(path_gambar)
    
    if gambar is None:
        print(f"[WARNING] Gambar {NAMA_FILE_GAMBAR} tidak ditemukan")
        print("[INFO] Menggunakan gambar contoh berwarna...")
        gambar = buat_gambar_berwarna()
    else:
        print(f"[SUKSES] Gambar dimuat: {path_gambar}")
    
    # 1. Demonstrasi konversi dasar
    demonstrasi_konversi_dasar(gambar)
    
    # 2. Penjelasan HSV
    jelaskan_hsv()
    
    # 3. Demo deteksi warna
    demo_deteksi_warna(gambar)
    
    # 4. Perbandingan metode grayscale
    konversi_grayscale_berbagai_metode(gambar)
    
    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN KONVERSI COLOR SPACE")
    print("=" * 60)
    print("""
DAFTAR KONVERSI YANG UMUM DIGUNAKAN:
├── cv2.COLOR_BGR2RGB      : Untuk tampilan di Matplotlib
├── cv2.COLOR_BGR2GRAY     : Untuk analisis bentuk/edge
├── cv2.COLOR_BGR2HSV      : Untuk deteksi warna
├── cv2.COLOR_BGR2LAB      : Untuk koreksi warna
├── cv2.COLOR_RGB2BGR      : Dari PIL ke OpenCV
└── cv2.COLOR_HSV2BGR      : Kembali ke BGR dari HSV

KAPAN MENGGUNAKAN COLOR SPACE TERTENTU:
├── RGB/BGR : Tampilan, penyimpanan
├── Grayscale: Edge detection, morfologi, template matching
├── HSV     : Deteksi warna, segmentasi berbasis warna
└── LAB     : Color transfer, koreksi white balance

EKSPERIMEN YANG BISA DICOBA:
- Ubah range warna HSV untuk deteksi warna berbeda
- Bandingkan hasil grayscale dari metode berbeda
- Coba deteksi warna kulit (skin detection) dengan HSV
""")


# Jalankan program utama
if __name__ == "__main__":
    main()
