# ============================================================
# PROGRAM: 05_manipulasi_piksel.py
# PRAKTIKUM COMPUTER VISION - BAB 1: PENDAHULUAN
# ============================================================
# Deskripsi: Program ini mendemonstrasikan cara mengakses dan
#            memanipulasi piksel gambar secara langsung
# 
# Tujuan Pembelajaran:
#   1. Mengakses nilai piksel individual
#   2. Memodifikasi region of interest (ROI)
#   3. Operasi aritmatika pada piksel
#   4. Memahami koordinat dan indexing gambar
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
NAMA_FILE_GAMBAR = "portrait.jpg"

# 2. Koordinat piksel yang akan diakses (y, x)
KOORDINAT_PIKSEL = (100, 150)

# 3. Ukuran ROI (Region of Interest)
ROI_START = (50, 50)      # (y, x) titik awal
ROI_SIZE = (100, 100)     # (tinggi, lebar)

# 4. Nilai untuk mengubah brightness (-100 sampai 100)
NILAI_BRIGHTNESS = 70

# 5. Nilai untuk mengubah contrast (0.5 sampai 2.0)
NILAI_CONTRAST = 1.0

# 6. Warna untuk menggambar ROI (BGR)
WARNA_ROI = (0, 255, 0)  # Hijau

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


def buat_gambar_gradient():
    """Membuat gambar gradient untuk demonstrasi"""
    gambar = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Gradient horizontal (Blue)
    for i in range(400):
        gambar[:, i, 0] = int(i * 255 / 400)
    
    # Gradient vertikal (Green)
    for i in range(300):
        gambar[i, :, 1] = int(i * 255 / 300)
    
    # Red tetap di tengah
    cv2.rectangle(gambar, (150, 100), (250, 200), (0, 0, 255), -1)
    
    return gambar


# ============================================================
# FUNGSI AKSES DAN MANIPULASI PIKSEL
# ============================================================

def akses_piksel_individual(gambar):
    """
    Mendemonstrasikan cara mengakses nilai piksel individual
    """
    print("\n" + "=" * 60)
    print("AKSES PIKSEL INDIVIDUAL")
    print("=" * 60)
    
    y, x = KOORDINAT_PIKSEL
    
    # Cek apakah koordinat valid
    if y >= gambar.shape[0] or x >= gambar.shape[1]:
        print(f"[ERROR] Koordinat ({y}, {x}) di luar batas gambar!")
        y, x = 0, 0
    
    # Akses piksel - PENTING: OpenCV menggunakan (y, x) bukan (x, y)!
    print("\n[PENTING] OpenCV menggunakan indexing (y, x) atau (row, col)")
    print(f"          Bukan (x, y) seperti koordinat matematika!")
    
    if len(gambar.shape) == 3:
        # Gambar berwarna (3 channel: BGR)
        piksel_bgr = gambar[y, x]
        blue, green, red = piksel_bgr
        
        print(f"\nNilai piksel di koordinat (y={y}, x={x}):")
        print(f"├── Format BGR : {piksel_bgr}")
        print(f"├── Blue  (B)  : {blue}")
        print(f"├── Green (G)  : {green}")
        print(f"└── Red   (R)  : {red}")
        
        # Akses channel terpisah
        print(f"\nAkses channel terpisah:")
        print(f"├── gambar[{y}, {x}, 0] = {gambar[y, x, 0]} (Blue)")
        print(f"├── gambar[{y}, {x}, 1] = {gambar[y, x, 1]} (Green)")
        print(f"└── gambar[{y}, {x}, 2] = {gambar[y, x, 2]} (Red)")
        
    else:
        # Gambar grayscale (1 channel)
        nilai = gambar[y, x]
        print(f"\nNilai piksel grayscale di ({y}, {x}): {nilai}")
    
    return y, x


def modifikasi_piksel_individual(gambar):
    """
    Mendemonstrasikan cara memodifikasi piksel individual
    """
    print("\n" + "=" * 60)
    print("MODIFIKASI PIKSEL INDIVIDUAL")
    print("=" * 60)
    
    gambar_copy = gambar.copy()
    
    # Modifikasi 1: Ubah satu piksel
    y, x = 60, 40
    
    print(f"\n[SEBELUM] Nilai di ({y}, {x}): {gambar_copy[y, x]}")
    gambar_copy[y, x] = [255, 255, 255]  # Ubah jadi putih
    print(f"[SESUDAH] Nilai di ({y}, {x}): {gambar_copy[y, x]}")
    
    # Modifikasi 2: Ubah sekelompok piksel (membuat kotak merah)
    print("\n[INFO] Membuat kotak merah 10x10 di sudut kiri atas...")
    gambar_copy[0:10, 0:10] = [0, 0, 255]  # BGR: Merah
    
    # Modifikasi 3: Ubah hanya satu channel
    print("[INFO] Mengubah channel Blue menjadi 0 di area tertentu...")
    gambar_copy[20:40, 20:40, 0] = 0  # Hilangkan Blue
    
    return gambar_copy


def demo_roi(gambar):
    """
    Mendemonstrasikan Region of Interest (ROI)
    """
    print("\n" + "=" * 60)
    print("REGION OF INTEREST (ROI)")
    print("=" * 60)
    
    y_start, x_start = ROI_START
    tinggi, lebar = ROI_SIZE
    y_end = y_start + tinggi
    x_end = x_start + lebar
    
    # Validasi koordinat
    y_end = min(y_end, gambar.shape[0])
    x_end = min(x_end, gambar.shape[1])
    
    print(f"\nROI dari ({y_start}, {x_start}) sampai ({y_end}, {x_end})")
    print(f"Ukuran ROI: {tinggi} x {lebar} piksel")
    
    # Ekstrak ROI menggunakan slicing
    # Sintaks: gambar[y_start:y_end, x_start:x_end]
    roi = gambar[y_start:y_end, x_start:x_end].copy()
    
    print("\n[INFO] Ekstrak ROI dengan slicing:")
    print(f"       roi = gambar[{y_start}:{y_end}, {x_start}:{x_end}]")
    
    # Tampilkan gambar dengan ROI ditandai
    gambar_marked = gambar.copy()
    cv2.rectangle(gambar_marked, (x_start, y_start), (x_end, y_end), WARNA_ROI, 2)
    
    # Tampilkan
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Gambar Asli")
    
    axes[1].imshow(cv2.cvtColor(gambar_marked, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"ROI ditandai\n({x_start},{y_start}) - ({x_end},{y_end})")
    
    axes[2].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"ROI diekstrak\nUkuran: {roi.shape}")
    
    for ax in axes:
        ax.axis('off')
    
    plt.suptitle("Demonstrasi Region of Interest (ROI)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return roi


def operasi_copy_paste_roi(gambar):
    """
    Mendemonstrasikan copy-paste ROI
    """
    print("\n" + "=" * 60)
    print("COPY-PASTE ROI")
    print("=" * 60)
    
    gambar_copy = gambar.copy()
    
    # Definisikan area sumber
    y1_src, x1_src = 0, 0
    y2_src, x2_src = 80, 80
    
    # Ekstrak ROI sumber
    roi_source = gambar[y1_src:y2_src, x1_src:x2_src].copy()
    
    # Paste ke lokasi berbeda
    y_dst = gambar.shape[0] - 80
    x_dst = gambar.shape[1] - 80
    
    print(f"\n[INFO] Copy dari ({y1_src},{x1_src}) - ({y2_src},{x2_src})")
    print(f"[INFO] Paste ke ({y_dst},{x_dst})")
    
    # Pastikan area tujuan valid
    if y_dst + roi_source.shape[0] <= gambar_copy.shape[0] and \
       x_dst + roi_source.shape[1] <= gambar_copy.shape[1]:
        gambar_copy[y_dst:y_dst+80, x_dst:x_dst+80] = roi_source
    
    # Tampilkan
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Gambar Asli")
    
    axes[1].imshow(cv2.cvtColor(gambar_copy, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Setelah Copy-Paste ROI")
    
    for ax in axes:
        ax.axis('off')
    
    plt.suptitle("Operasi Copy-Paste Region of Interest", fontsize=14)
    plt.tight_layout()
    plt.show()


def operasi_aritmatika_piksel(gambar):
    """
    Mendemonstrasikan operasi aritmatika pada piksel
    """
    print("\n" + "=" * 60)
    print("OPERASI ARITMATIKA PADA PIKSEL")
    print("=" * 60)
    
    # 1. Brightness (menambah/mengurangi nilai)
    print(f"\n[1] Brightness: Menambah {NILAI_BRIGHTNESS} ke semua piksel")
    
    # Cara SALAH (bisa overflow/underflow!)
    # gambar_bright = gambar + NILAI_BRIGHTNESS  # JANGAN!
    
    # Cara BENAR menggunakan cv2.add dengan saturasi
    if NILAI_BRIGHTNESS >= 0:
        bright_matrix = np.ones(gambar.shape, dtype=np.uint8) * abs(NILAI_BRIGHTNESS)
        gambar_bright = cv2.add(gambar, bright_matrix)
    else:
        bright_matrix = np.ones(gambar.shape, dtype=np.uint8) * abs(NILAI_BRIGHTNESS)
        gambar_bright = cv2.subtract(gambar, bright_matrix)
    
    # 2. Contrast (mengalikan nilai)
    print(f"[2] Contrast: Mengalikan dengan {NILAI_CONTRAST}")
    gambar_contrast = np.clip(gambar.astype(np.float32) * NILAI_CONTRAST, 0, 255).astype(np.uint8)
    
    # 3. Invert (membalik nilai)
    print("[3] Invert: 255 - nilai_piksel")
    gambar_invert = cv2.bitwise_not(gambar)
    
    # 4. Kombinasi brightness + contrast
    print(f"[4] Kombinasi: contrast={NILAI_CONTRAST}, brightness={NILAI_BRIGHTNESS}")
    gambar_combined = np.clip(
        gambar.astype(np.float32) * NILAI_CONTRAST + NILAI_BRIGHTNESS, 
        0, 255
    ).astype(np.uint8)
    
    # Tampilkan
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Gambar Asli")
    
    axes[0, 1].imshow(cv2.cvtColor(gambar_bright, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Brightness +{NILAI_BRIGHTNESS}")
    
    axes[0, 2].imshow(cv2.cvtColor(gambar_contrast, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"Contrast x{NILAI_CONTRAST}")
    
    axes[1, 0].imshow(cv2.cvtColor(gambar_invert, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Inverted (Negatif)")
    
    axes[1, 1].imshow(cv2.cvtColor(gambar_combined, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Combined\nC={NILAI_CONTRAST}, B={NILAI_BRIGHTNESS}")
    
    # Histogram perbandingan
    axes[1, 2].hist(gambar.ravel(), 256, [0, 256], alpha=0.5, label='Asli')
    axes[1, 2].hist(gambar_bright.ravel(), 256, [0, 256], alpha=0.5, label='Bright')
    axes[1, 2].set_title("Histogram Perbandingan")
    axes[1, 2].legend()
    
    for ax in axes.flat[:5]:
        ax.axis('off')
    
    plt.suptitle("Operasi Aritmatika pada Piksel", fontsize=14)
    plt.tight_layout()
    plt.show()


def blending_dua_gambar(gambar1, gambar2=None):
    """
    Mendemonstrasikan blending dua gambar
    """
    print("\n" + "=" * 60)
    print("BLENDING DUA GAMBAR")
    print("=" * 60)
    
    # Jika gambar2 tidak ada, buat gambar solid
    if gambar2 is None:
        gambar2 = np.zeros_like(gambar1)
        gambar2[:, :] = [255, 165, 0]  # Orange
    
    # Pastikan ukuran sama
    if gambar1.shape != gambar2.shape:
        gambar2 = cv2.resize(gambar2, (gambar1.shape[1], gambar1.shape[0]))
    
    # Blending dengan berbagai alpha
    alphas = [0.25, 0.5, 0.75]
    hasil = []
    
    print("\nFormula blending: output = α*gambar1 + (1-α)*gambar2")
    
    for alpha in alphas:
        beta = 1.0 - alpha
        blended = cv2.addWeighted(gambar1, alpha, gambar2, beta, 0)
        hasil.append(blended)
        print(f"α={alpha}: {alpha*100:.0f}% gambar1 + {beta*100:.0f}% gambar2")
    
    # Tampilkan
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(gambar1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Gambar 1 (α=1.0)")
    
    axes[0, 1].imshow(cv2.cvtColor(gambar2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Gambar 2 (α=0.0)")
    
    axes[0, 2].imshow(cv2.cvtColor(hasil[1], cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Blended (α=0.5)")
    
    for i, (alpha, img) in enumerate(zip(alphas, hasil)):
        axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"α = {alpha}")
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.suptitle("Blending Dua Gambar dengan cv2.addWeighted()", fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    """Fungsi utama program"""
    
    print("\n" + "=" * 60)
    print("PRAKTIKUM 5: MANIPULASI PIKSEL")
    print("=" * 60)
    
    # Muat gambar
    path_gambar = dapatkan_path_gambar(NAMA_FILE_GAMBAR)
    gambar = cv2.imread(path_gambar)
    
    if gambar is None:
        print(f"[WARNING] Gambar {NAMA_FILE_GAMBAR} tidak ditemukan")
        print("[INFO] Menggunakan gambar gradient contoh...")
        gambar = buat_gambar_gradient()
    else:
        print(f"[SUKSES] Gambar dimuat: {path_gambar}")
    
    # 1. Akses piksel individual
    akses_piksel_individual(gambar)
    
    # 2. Demo ROI
    demo_roi(gambar)
    
    # 3. Copy-paste ROI
    operasi_copy_paste_roi(gambar)
    
    # 4. Operasi aritmatika
    operasi_aritmatika_piksel(gambar)
    
    # 5. Blending
    blending_dua_gambar(gambar)
    
    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN MANIPULASI PIKSEL")
    print("=" * 60)
    print("""
AKSES PIKSEL:
├── gambar[y, x]           : Akses 1 piksel (semua channel)
├── gambar[y, x, 0]        : Akses channel Blue
├── gambar[y, x, 1]        : Akses channel Green
└── gambar[y, x, 2]        : Akses channel Red

ROI (REGION OF INTEREST):
├── roi = gambar[y1:y2, x1:x2]  : Ekstrak ROI
├── gambar[y1:y2, x1:x2] = roi  : Replace dengan ROI
└── roi.copy()                   : Copy ROI (hindari reference!)

OPERASI ARITMATIKA:
├── cv2.add(img, nilai)     : Tambah dengan saturasi
├── cv2.subtract(img, nilai): Kurang dengan saturasi
├── cv2.multiply(img, nilai): Kali dengan saturasi
├── cv2.bitwise_not(img)    : Invert/negatif
└── cv2.addWeighted()       : Blending dua gambar

PENTING:
- OpenCV menggunakan (y, x) bukan (x, y)!
- Selalu gunakan .copy() saat ekstrak ROI
- Gunakan cv2.add/subtract untuk hindari overflow
- Nilai piksel 0-255 (uint8)
""")


# Jalankan program utama
if __name__ == "__main__":
    main()
