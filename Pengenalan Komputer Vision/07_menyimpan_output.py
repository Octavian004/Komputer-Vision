# ============================================================
# PROGRAM: 07_menyimpan_output.py
# PRAKTIKUM COMPUTER VISION - BAB 1: PENDAHULUAN
# ============================================================
# Deskripsi: Program ini mendemonstrasikan cara menyimpan
#            hasil pemrosesan gambar dalam berbagai format
# 
# Tujuan Pembelajaran:
#   1. Menyimpan gambar dengan cv2.imwrite()
#   2. Memahami berbagai format file dan kompresi
#   3. Mengatur kualitas output
#   4. Menyimpan sequence gambar (video)
# ============================================================

# ====================
# IMPORT LIBRARY
# ====================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============================================================
# VARIABEL YANG BISA DIUBAH-UBAH (EKSPERIMEN)
# ============================================================

# 1. File gambar sumber
NAMA_FILE_SUMBER = "input.jpg"

# 2. Direktori output
DIREKTORI_OUTPUT = os.path.join("output", "output7")

# 3. Kualitas JPEG (0-100, semakin tinggi semakin bagus tapi ukuran besar)
KUALITAS_JPEG = 95

# 4. Kompresi PNG (0-9, semakin tinggi semakin kecil tapi proses lebih lama)
KOMPRESI_PNG = 3

# 5. Nama file output (tanpa ekstensi)
NAMA_OUTPUT_BASE = "hasil_pemrosesan"

# 6. Format yang akan didemonstrasikan
FORMAT_OUTPUT = ['jpg', 'png', 'bmp', 'tiff', 'webp']

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


def buat_gambar_sample():
    """Membuat gambar sample untuk demonstrasi"""
    gambar = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(640):
        gambar[:, i] = [int(i/2.5), int(255 - i/2.5), 128]
    
    # Tambah beberapa shapes
    cv2.circle(gambar, (320, 240), 100, (255, 255, 255), -1)
    cv2.rectangle(gambar, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.putText(gambar, "Sample Image", (220, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return gambar


def dapatkan_ukuran_file(path):
    """Mendapatkan ukuran file dalam KB"""
    if os.path.exists(path):
        return os.path.getsize(path) / 1024  # KB
    return 0


def buat_direktori_output():
    """Membuat direktori output jika belum ada"""
    direktori_script = os.path.dirname(os.path.abspath(__file__))
    path_output = os.path.join(direktori_script, DIREKTORI_OUTPUT)
    
    os.makedirs(path_output, exist_ok=True)
    print(f"[INFO] Direktori siap: {path_output}")
    
    return path_output


# ============================================================
# FUNGSI MENYIMPAN GAMBAR
# ============================================================

def demo_simpan_dasar(gambar, dir_output):
    """
    Demonstrasi dasar menyimpan gambar
    """
    print("\n" + "=" * 60)
    print("MENYIMPAN GAMBAR DASAR (cv2.imwrite)")
    print("=" * 60)
    
    print("""
Sintaks:
    cv2.imwrite(filename, img)
    cv2.imwrite(filename, img, [params])

Parameter:
    - filename: path lengkap termasuk ekstensi
    - img: array numpy (gambar)
    - params: parameter opsional untuk kualitas/kompresi
    """)
    
    # Simpan dalam berbagai format
    hasil = {}
    
    for fmt in FORMAT_OUTPUT:
        nama_file = f"{NAMA_OUTPUT_BASE}.{fmt}"
        path_file = os.path.join(dir_output, nama_file)
        
        waktu_mulai = time.time()
        sukses = cv2.imwrite(path_file, gambar)
        waktu_simpan = (time.time() - waktu_mulai) * 1000  # ms
        
        ukuran = dapatkan_ukuran_file(path_file)
        
        hasil[fmt] = {
            'path': path_file,
            'sukses': sukses,
            'ukuran_kb': ukuran,
            'waktu_ms': waktu_simpan
        }
        
        status = "✓ SUKSES" if sukses else "✗ GAGAL"
        print(f"\n{status} : {nama_file}")
        print(f"        Ukuran  : {ukuran:.2f} KB")
        print(f"        Waktu   : {waktu_simpan:.2f} ms")
    
    return hasil


def demo_simpan_jpeg_kualitas(gambar, dir_output):
    """
    Demonstrasi menyimpan JPEG dengan berbagai kualitas
    """
    print("\n" + "=" * 60)
    print("JPEG DENGAN BERBAGAI KUALITAS")
    print("=" * 60)
    
    print("""
Parameter JPEG:
    cv2.IMWRITE_JPEG_QUALITY: 0-100 (default: 95)
    - 0  : Kompresi maksimal, kualitas rendah
    - 100: Tanpa kompresi lossy, kualitas maksimal
    """)
    
    kualitas_list = [10, 30, 50, 70, 90, 100]
    hasil = {}
    
    print("\nKualitas | Ukuran (KB) | Rasio Kompresi")
    print("-" * 45)
    
    ukuran_referensi = None
    
    for kualitas in kualitas_list:
        nama_file = f"jpeg_q{kualitas}.jpg"
        path_file = os.path.join(dir_output, nama_file)
        
        # Simpan dengan parameter kualitas
        cv2.imwrite(path_file, gambar, [cv2.IMWRITE_JPEG_QUALITY, kualitas])
        
        ukuran = dapatkan_ukuran_file(path_file)
        hasil[kualitas] = {'path': path_file, 'ukuran_kb': ukuran}
        
        if ukuran_referensi is None:
            ukuran_referensi = ukuran
        
        rasio = (ukuran / ukuran_referensi) * 100
        print(f"   {kualitas:3d}   |   {ukuran:8.2f}  |    {rasio:6.1f}%")
    
    return hasil


def demo_simpan_png_kompresi(gambar, dir_output):
    """
    Demonstrasi menyimpan PNG dengan berbagai level kompresi
    """
    print("\n" + "=" * 60)
    print("PNG DENGAN BERBAGAI LEVEL KOMPRESI")
    print("=" * 60)
    
    print("""
Parameter PNG:
    cv2.IMWRITE_PNG_COMPRESSION: 0-9 (default: 3)
    - 0: Tanpa kompresi (tercepat, ukuran besar)
    - 9: Kompresi maksimal (terlama, ukuran kecil)
    
CATATAN: PNG adalah lossless - kualitas tetap sama!
         Hanya waktu encoding dan ukuran yang berbeda.
    """)
    
    kompresi_list = [0, 1, 3, 5, 7, 9]
    hasil = {}
    
    print("\nKompresi | Ukuran (KB) | Waktu (ms)")
    print("-" * 40)
    
    for kompresi in kompresi_list:
        nama_file = f"png_c{kompresi}.png"
        path_file = os.path.join(dir_output, nama_file)
        
        waktu_mulai = time.time()
        cv2.imwrite(path_file, gambar, [cv2.IMWRITE_PNG_COMPRESSION, kompresi])
        waktu_simpan = (time.time() - waktu_mulai) * 1000
        
        ukuran = dapatkan_ukuran_file(path_file)
        hasil[kompresi] = {'path': path_file, 'ukuran_kb': ukuran, 'waktu_ms': waktu_simpan}
        
        print(f"   {kompresi}     |   {ukuran:8.2f}  |   {waktu_simpan:6.2f}")
    
    return hasil


def demo_simpan_dengan_transparansi(dir_output):
    """
    Demonstrasi menyimpan gambar dengan transparansi (alpha channel)
    """
    print("\n" + "=" * 60)
    print("MENYIMPAN GAMBAR DENGAN TRANSPARANSI (RGBA)")
    print("=" * 60)
    
    print("""
Untuk menyimpan transparansi, gunakan format:
    - PNG (mendukung alpha channel)
    - TIFF (mendukung alpha channel)
    - WebP (mendukung alpha channel)
    
JPEG TIDAK mendukung transparansi!
    """)
    
    # Buat gambar RGBA
    gambar_rgba = np.zeros((300, 400, 4), dtype=np.uint8)
    
    # Isi dengan warna dan gradient alpha
    gambar_rgba[:, :, :3] = [255, 100, 100]  # Warna merah-pink (BGR)
    
    # Buat gradient alpha (dari transparan ke opaque)
    for i in range(400):
        gambar_rgba[:, i, 3] = int(i * 255 / 400)
    
    # Tambah lingkaran putih solid di tengah
    cv2.circle(gambar_rgba, (200, 150), 80, (255, 255, 255, 255), -1)
    
    # Simpan dalam berbagai format
    path_png = os.path.join(dir_output, "dengan_alpha.png")
    path_tiff = os.path.join(dir_output, "dengan_alpha.tiff")
    path_webp = os.path.join(dir_output, "dengan_alpha.webp")
    
    cv2.imwrite(path_png, gambar_rgba)
    cv2.imwrite(path_tiff, gambar_rgba)
    cv2.imwrite(path_webp, gambar_rgba)
    
    print(f"\n[SUKSES] PNG dengan alpha  : {path_png}")
    print(f"[SUKSES] TIFF dengan alpha : {path_tiff}")
    print(f"[SUKSES] WebP dengan alpha : {path_webp}")
    
    # Tampilkan
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Checkerboard background untuk visualisasi alpha
    checker = np.zeros((300, 400, 3), dtype=np.uint8)
    checker_size = 20
    for i in range(0, 300, checker_size):
        for j in range(0, 400, checker_size):
            if (i // checker_size + j // checker_size) % 2:
                checker[i:i+checker_size, j:j+checker_size] = [200, 200, 200]
            else:
                checker[i:i+checker_size, j:j+checker_size] = [255, 255, 255]
    
    # Composite dengan background
    alpha = gambar_rgba[:, :, 3:4] / 255.0
    bgr = gambar_rgba[:, :, :3]
    composite = (bgr * alpha + checker * (1 - alpha)).astype(np.uint8)
    
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("RGB saja (tanpa alpha)")
    
    axes[1].imshow(gambar_rgba[:, :, 3], cmap='gray')
    axes[1].set_title("Alpha Channel\n(hitam=transparan, putih=opaque)")
    
    axes[2].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Hasil Composite\n(dengan checkerboard background)")
    
    for ax in axes:
        ax.axis('off')
    
    plt.suptitle("Demonstrasi Gambar dengan Transparansi (RGBA)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return gambar_rgba


def demo_simpan_video(dir_output):
    """
    Demonstrasi menyimpan sequence gambar sebagai video
    """
    print("\n" + "=" * 60)
    print("MENYIMPAN VIDEO (cv2.VideoWriter)")
    print("=" * 60)
    
    print("""
Sintaks:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # atau 'mp4v', 'MJPG'
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    out.write(frame)  # untuk setiap frame
    out.release()
    
Codec umum:
    - 'XVID': AVI format (cross-platform)
    - 'mp4v': MP4 format
    - 'MJPG': Motion JPEG
    """)
    
    # Parameter video
    width, height = 640, 480
    fps = 30
    durasi = 3  # detik
    total_frames = fps * durasi
    
    path_video = os.path.join(dir_output, "animasi_demo.avi")
    
    # Buat VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path_video, fourcc, fps, (width, height))
    
    print(f"\n[INFO] Membuat video: {path_video}")
    print(f"       Resolusi: {width}x{height}")
    print(f"       FPS: {fps}")
    print(f"       Durasi: {durasi} detik ({total_frames} frames)")
    print("\n       Generating frames", end="")
    
    # Generate frames dengan animasi
    for frame_idx in range(total_frames):
        # Buat frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient yang berubah
        hue = int((frame_idx / total_frames) * 180)
        frame[:] = cv2.cvtColor(
            np.full((height, width, 3), [hue, 255, 128], dtype=np.uint8),
            cv2.COLOR_HSV2BGR
        )
        
        # Lingkaran yang bergerak
        x = int(100 + (frame_idx / total_frames) * (width - 200))
        y = int(height/2 + 100 * np.sin(frame_idx * 0.1))
        cv2.circle(frame, (x, y), 50, (255, 255, 255), -1)
        
        # Tambah frame counter
        cv2.putText(frame, f"Frame: {frame_idx+1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tulis frame
        out.write(frame)
        
        if frame_idx % 30 == 0:
            print(".", end="", flush=True)
    
    out.release()
    print(" Done!")
    
    ukuran = dapatkan_ukuran_file(path_video)
    print(f"\n[SUKSES] Video disimpan: {path_video}")
    print(f"         Ukuran file: {ukuran:.2f} KB")
    
    return path_video


def visualisasi_perbandingan_format(gambar, dir_output):
    """
    Visualisasi perbandingan format file
    """
    print("\n" + "=" * 60)
    print("VISUALISASI PERBANDINGAN FORMAT")
    print("=" * 60)
    
    # Simpan dalam berbagai format
    formats = {
        'BMP': ('bmp', []),
        'PNG (0)': ('png', [cv2.IMWRITE_PNG_COMPRESSION, 0]),
        'PNG (9)': ('png', [cv2.IMWRITE_PNG_COMPRESSION, 9]),
        'JPEG (100)': ('jpg', [cv2.IMWRITE_JPEG_QUALITY, 100]),
        'JPEG (50)': ('jpg', [cv2.IMWRITE_JPEG_QUALITY, 50]),
        'JPEG (10)': ('jpg', [cv2.IMWRITE_JPEG_QUALITY, 10]),
    }
    
    data = []
    for nama, (ext, params) in formats.items():
        nama_file = f"compare_{nama.replace(' ', '_').replace('(', '').replace(')', '')}.{ext}"
        path = os.path.join(dir_output, nama_file)
        
        if params:
            cv2.imwrite(path, gambar, params)
        else:
            cv2.imwrite(path, gambar)
        
        ukuran = dapatkan_ukuran_file(path)
        data.append((nama, ukuran))
    
    # Buat bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    nama_format = [d[0] for d in data]
    ukuran_format = [d[1] for d in data]
    
    colors = ['#3498db', '#2ecc71', '#27ae60', '#e74c3c', '#c0392b', '#922b21']
    
    ax1.barh(nama_format, ukuran_format, color=colors)
    ax1.set_xlabel('Ukuran File (KB)')
    ax1.set_title('Perbandingan Ukuran File per Format')
    
    for i, (nama, ukuran) in enumerate(data):
        ax1.text(ukuran + 5, i, f'{ukuran:.1f} KB', va='center')
    
    # Tabel info
    ax2.axis('off')
    
    tabel_info = """
    PERBANDINGAN FORMAT GAMBAR
    ==========================
    
    BMP (Bitmap)
    ├── Tidak terkompresi
    ├── Ukuran besar
    └── Cocok: Data mentah, debugging
    
    PNG (Portable Network Graphics)
    ├── Lossless compression
    ├── Mendukung transparansi
    └── Cocok: Screenshots, grafik, logo
    
    JPEG (Joint Photographic Experts Group)
    ├── Lossy compression
    ├── Ukuran kecil
    └── Cocok: Foto, web, storage
    
    TIFF (Tagged Image File Format)
    ├── Mendukung berbagai kompresi
    ├── Metadata lengkap
    └── Cocok: Profesional, printing
    
    WebP
    ├── Modern format (Google)
    ├── Lossy dan lossless
    └── Cocok: Web, mobile
    """
    
    ax2.text(0.1, 0.9, tabel_info, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle("Perbandingan Format File Gambar", fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    """Fungsi utama program"""
    
    print("\n" + "=" * 60)
    print("PRAKTIKUM 7: MENYIMPAN OUTPUT")
    print("=" * 60)
    
    # Buat direktori output
    dir_output = buat_direktori_output()
    
    # Muat atau buat gambar
    path_gambar = dapatkan_path_gambar(NAMA_FILE_SUMBER)
    gambar = cv2.imread(path_gambar)
    
    if gambar is None:
        print(f"[WARNING] Gambar {NAMA_FILE_SUMBER} tidak ditemukan")
        print("[INFO] Menggunakan gambar sample...")
        gambar = buat_gambar_sample()
    else:
        print(f"[SUKSES] Gambar dimuat: {path_gambar}")
    
    print(f"[INFO] Ukuran gambar: {gambar.shape}")
    print(f"[INFO] Output directory: {dir_output}")
    
    # 1. Demo simpan dasar
    demo_simpan_dasar(gambar, dir_output)
    
    # 2. Demo JPEG kualitas
    demo_simpan_jpeg_kualitas(gambar, dir_output)
    
    # 3. Demo PNG kompresi
    demo_simpan_png_kompresi(gambar, dir_output)
    
    # 4. Demo transparansi
    demo_simpan_dengan_transparansi(dir_output)
    
    # 5. Demo video
    demo_simpan_video(dir_output)
    
    # 6. Visualisasi perbandingan
    visualisasi_perbandingan_format(gambar, dir_output)
    
    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN MENYIMPAN OUTPUT")
    print("=" * 60)
    print(f"""
FUNGSI UTAMA:
├── cv2.imwrite(filename, img)         : Simpan gambar
├── cv2.imwrite(filename, img, params) : Simpan dengan parameter
└── cv2.VideoWriter()                   : Simpan video

PARAMETER PENTING:
├── JPEG:
│   └── cv2.IMWRITE_JPEG_QUALITY: 0-100 (default 95)
├── PNG:
│   └── cv2.IMWRITE_PNG_COMPRESSION: 0-9 (default 3)
└── WebP:
    └── cv2.IMWRITE_WEBP_QUALITY: 0-100

PILIHAN FORMAT:
├── BMP  : Tanpa kompresi, debugging
├── PNG  : Lossless, transparansi
├── JPEG : Foto, ukuran kecil
├── TIFF : Profesional, metadata
└── WebP : Modern, web/mobile

OUTPUT TERSIMPAN DI: {dir_output}

EKSPERIMEN YANG BISA DICOBA:
- Ubah KUALITAS_JPEG dan bandingkan hasil
- Bandingkan ukuran file dengan gambar berbeda
- Coba simpan screenshot dengan PNG vs JPEG
""")


# Jalankan program utama
if __name__ == "__main__":
    main()
