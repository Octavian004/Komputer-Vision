"""
=============================================================================
Modul 03 - Pemrosesan Citra (Image Processing)
Praktikum 18: Transformasi Fourier dan Analisis Domain Frekuensi
=============================================================================
Deskripsi:
    Mempelajari Discrete Fourier Transform (DFT) untuk menganalisis gambar
    dalam domain frekuensi. Komponen frekuensi rendah merepresentasikan
    area halus (smooth), sedangkan frekuensi tinggi merepresentasikan
    detail dan tepi (edges).

    Transformasi Fourier 2D:
        F(u,v) = sum_x sum_y f(x,y) * exp(-j2pi(ux/M + vy/N))

    Magnitude spectrum dalam skala logaritmik membantu memvisualisasikan
    distribusi energi frekuensi secara intuitif.

Topik yang dibahas:
    1. DFT dengan NumPy: fft2 -> fftshift -> magnitude spectrum (log)
    2. Filter frekuensi: Low-Pass, High-Pass, Band-Pass (circular mask)
    3. DFT dengan OpenCV: cv2.dft + cv2.idft vs NumPy
    4. Sinyal sinusoidal sintetis dan spike di magnitude spectrum

Referensi:
    - Gonzalez & Woods, "Digital Image Processing", 4th Edition, Bab 4
    - OpenCV Documentation: cv2.dft(), cv2.idft()
    - NumPy Documentation: np.fft.fft2(), np.fft.fftshift()

Penulis  : Praktikum Komputer Vision
Tanggal  : 2026-03-02
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    Memuat gambar kota.jpg sebagai grayscale (512x512).
    Jika tidak ditemukan, membuat gambar sintetis berupa superposisi
    gelombang cosinus 2D yang berguna untuk mendemonstrasikan
    analisis frekuensi Fourier (spike di magnitude spectrum).

    Returns:
        tuple: (gambar_gray uint8 512x512, label_sumber str)
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
    x = np.arange(lebar, dtype=np.float32)
    y = np.arange(tinggi, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Superposisi beberapa gelombang cosinus 2D
    # Setiap komponen menghasilkan spike di posisi frekuensi yang sesuai
    gambar = (
        128
        + 50 * np.cos(2 * np.pi * 5  * xx / lebar)     # frekuensi rendah-x
        + 30 * np.cos(2 * np.pi * 20 * yy / tinggi)    # frekuensi sedang-y
        + 20 * np.cos(2 * np.pi * 3  * xx / lebar
                      + 2 * np.pi * 3 * yy / tinggi)   # frekuensi diagonal
        + 10 * np.random.randn(tinggi, lebar)           # noise ringan
    )
    gambar = np.clip(gambar, 0, 255).astype(np.uint8)

    return gambar, "sintetis"


# ---------------------------------------------------------------------------
# Demo 1: Magnitude Spectrum dengan NumPy FFT
# ---------------------------------------------------------------------------
def demo_magnitude_spectrum(gambar_gray, label_sumber):
    """
    Menghitung dan memvisualisasikan magnitude spectrum menggunakan NumPy:
        1. np.fft.fft2()     -> DFT 2D (domain spasial ke frekuensi)
        2. np.fft.fftshift() -> geser komponen DC ke tengah gambar
        3. magnitude = |F(u,v)|
        4. log_magnitude = 20 * log(magnitude + 1) -> skala log

    Komponen DC (titik tengah setelah fftshift) merepresentasikan
    rata-rata intensitas gambar. Frekuensi tinggi berada di tepi.

    Args:
        gambar_gray  : gambar grayscale uint8
        label_sumber : string label sumber gambar
    """
    print("\n[Demo 1] Magnitude Spectrum via NumPy FFT")

    img_f32 = gambar_gray.astype(np.float32)

    # Langkah 1: DFT 2D menggunakan NumPy
    f_transform = np.fft.fft2(img_f32)

    # Langkah 2: Geser zero-frequency (DC) ke tengah untuk visualisasi
    f_shift = np.fft.fftshift(f_transform)

    # Langkah 3: Hitung magnitude (modulus dari bilangan kompleks)
    magnitude = np.abs(f_shift)

    # Langkah 4: Skala logaritmik agar komponen DC tidak mendominasi tampilan
    mag_log = 20 * np.log(magnitude + 1)

    # Phase spectrum (sudut fasa dari bilangan kompleks)
    phase = np.angle(f_shift)

    print(f"  Ukuran DFT   : {f_transform.shape}")
    print(f"  Nilai DC     : {np.abs(f_transform[0, 0]):.0f}")
    print(f"  Magnitude max: {magnitude.max():.0f}")
    print(f"  Phase range  : [{phase.min():.2f}, {phase.max():.2f}] rad")

    # ---- Visualisasi 6 panel ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Demo 1: Magnitude Spectrum dengan NumPy FFT\n"
        "fft2 -> fftshift -> skala log  |  Komponen DC di tengah",
        fontsize=13, fontweight="bold"
    )

    axes[0, 0].imshow(gambar_gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Gambar Asli ({label_sumber})")
    axes[0, 0].axis("off")

    # Magnitude SEBELUM fftshift: DC di sudut kiri-atas
    mag_before = 20 * np.log(np.abs(f_transform) + 1)
    axes[0, 1].imshow(mag_before, cmap="inferno")
    axes[0, 1].set_title("Magnitude SEBELUM fftshift\n(DC di sudut kiri-atas)")
    axes[0, 1].axis("off")

    # Magnitude SETELAH fftshift: DC di tengah
    axes[0, 2].imshow(mag_log, cmap="inferno")
    axes[0, 2].set_title("Magnitude SETELAH fftshift\n(DC di tengah, skala log)")
    axes[0, 2].axis("off")

    # Phase spectrum (menggunakan colormap HSV agar simetri terlihat)
    axes[1, 0].imshow(phase, cmap="hsv")
    axes[1, 0].set_title("Phase Spectrum (radian)")
    axes[1, 0].axis("off")

    # Profil horizontal spectrum (irisan baris tengah)
    tengah = mag_log.shape[0] // 2
    irisan = mag_log[tengah, :]
    axes[1, 1].plot(irisan, color="darkorange")
    axes[1, 1].set_title("Profil Horizontal Spectrum\n(irisan baris tengah)")
    axes[1, 1].set_xlabel("Frekuensi (piksel)")
    axes[1, 1].set_ylabel("Magnitude (log)")
    axes[1, 1].grid(True, alpha=0.3)

    # Rekonstruksi dari DFT untuk memverifikasi roundtrip
    f_ishift = np.fft.ifftshift(f_shift)
    rekonstruksi = np.abs(np.fft.ifft2(f_ishift)).astype(np.uint8)
    error = float(np.mean(cv2.absdiff(gambar_gray, rekonstruksi)))
    axes[1, 2].imshow(rekonstruksi, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title(f"Rekonstruksi IDFT\n(error rata-rata={error:.4f})")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_demo1_magnitude_spectrum.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Tersimpan: 18_demo1_magnitude_spectrum.png")


# ---------------------------------------------------------------------------
# Demo 2: Filter Frekuensi (Low-Pass, High-Pass, Band-Pass)
# ---------------------------------------------------------------------------
def demo_filter_frekuensi(gambar_gray):
    """
    Membuat filter dalam domain Fourier menggunakan circular mask:

    Low-Pass Filter (LPF):
        Mask = 1 di dalam lingkaran radius R, 0 di luar
        Efek: gambar lebih halus (blur), detail dan tepi hilang

    High-Pass Filter (HPF):
        Mask = 0 di dalam lingkaran radius R, 1 di luar
        Efek: hanya detail dan tepi yang tersisa (background gelap)

    Band-Pass Filter (BPF):
        Mask = 1 di antara radius R_dalam dan R_luar
        Efek: hanya frekuensi dalam pita tertentu yang lolos

    Langkah umum untuk setiap filter:
        1. DFT + fftshift -> spectrum terpusat
        2. Buat circular mask dan kalikan dengan spectrum
        3. ifftshift + ifft2 -> gambar terfilter di domain spasial

    Args:
        gambar_gray : gambar grayscale uint8
    """
    print("\n[Demo 2] Filter Frekuensi: Low-Pass, High-Pass, Band-Pass")

    tinggi, lebar = gambar_gray.shape
    cy, cx = tinggi // 2, lebar // 2      # pusat spectrum setelah fftshift

    # Hitung DFT + fftshift sekali untuk semua filter
    f_shift = np.fft.fftshift(np.fft.fft2(gambar_gray.astype(np.float32)))

    # Buat grid jarak dari pusat (untuk pembuatan mask lingkaran)
    y_idx, x_idx = np.ogrid[:tinggi, :lebar]
    jarak = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    def terapkan_mask(f_shift_lokal, mask):
        """Terapkan mask, lakukan IDFT, kembalikan gambar uint8."""
        f_filter = f_shift_lokal * mask
        f_ishift = np.fft.ifftshift(f_filter)
        img_filter = np.abs(np.fft.ifft2(f_ishift))
        return np.clip(img_filter, 0, 255).astype(np.uint8)

    # ---- Low-Pass Filter: loloskan frekuensi rendah (radius <= 30) ----
    radius_lpf = 30
    mask_lpf = (jarak <= radius_lpf).astype(np.float32)
    hasil_lpf = terapkan_mask(f_shift, mask_lpf)

    # ---- High-Pass Filter: loloskan frekuensi tinggi (radius > 30) ----
    radius_hpf = 30
    mask_hpf = (jarak > radius_hpf).astype(np.float32)
    hasil_hpf = terapkan_mask(f_shift, mask_hpf)

    # ---- Band-Pass Filter: loloskan frekuensi antara radius 20 dan 60 ----
    radius_bpf_dalam = 20
    radius_bpf_luar  = 60
    mask_bpf = (
        (jarak >= radius_bpf_dalam) & (jarak <= radius_bpf_luar)
    ).astype(np.float32)
    hasil_bpf = terapkan_mask(f_shift, mask_bpf)

    # Magnitude spectrum asli untuk referensi
    mag_asli = 20 * np.log(np.abs(f_shift) + 1)

    # ---- Visualisasi 8 panel ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(
        "Demo 2: Filter Frekuensi dalam Domain Fourier\n"
        "Circular Mask  ->  IDFT  ->  Gambar Terfilter",
        fontsize=13, fontweight="bold"
    )

    axes[0, 0].imshow(gambar_gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Gambar Asli")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mag_asli, cmap="inferno")
    axes[0, 1].set_title("Magnitude Spectrum\n(referensi)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask_lpf, cmap="gray")
    axes[0, 2].set_title(f"Mask Low-Pass\n(radius={radius_lpf})")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(mask_hpf, cmap="gray")
    axes[0, 3].set_title(f"Mask High-Pass\n(radius={radius_hpf})")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(mask_bpf, cmap="gray")
    axes[1, 0].set_title(f"Mask Band-Pass\n(R={radius_bpf_dalam}..{radius_bpf_luar})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hasil_lpf, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Hasil Low-Pass Filter\n(blur, detail hilang)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(hasil_hpf, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("Hasil High-Pass Filter\n(edge/detail saja)")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(hasil_bpf, cmap="gray", vmin=0, vmax=255)
    axes[1, 3].set_title("Hasil Band-Pass Filter\n(pita frekuensi tertentu)")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_demo2_filter_frekuensi.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Tersimpan: 18_demo2_filter_frekuensi.png")


# ---------------------------------------------------------------------------
# Demo 3: cv2.dft() dan cv2.idft() vs NumPy
# ---------------------------------------------------------------------------
def demo_opencv_dft(gambar_gray):
    """
    Membandingkan implementasi DFT OpenCV dengan NumPy:

    OpenCV cv2.dft:
        cv2.dft(src, flags=cv2.DFT_COMPLEX_OUTPUT)
          -> array shape (H, W, 2): channel 0=Re, channel 1=Im
        cv2.magnitude(Re, Im) -> magnitude array
        cv2.idft(dft_result, flags=cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)
          -> rekonstruksi gambar

    NumPy:
        np.fft.fft2()  -> complex array (H, W)
        np.abs()       -> magnitude
        np.fft.ifft2() -> rekonstruksi

    Keduanya harus menghasilkan spectrum dan rekonstruksi yang identik
    (selisih sangat kecil akibat presisi floating-point).

    Args:
        gambar_gray : gambar grayscale uint8
    """
    print("\n[Demo 3] OpenCV cv2.dft() vs NumPy: Perbandingan")

    img_f32 = gambar_gray.astype(np.float32)

    # ---- NumPy FFT ----
    f_np        = np.fft.fft2(img_f32)
    f_shift_np  = np.fft.fftshift(f_np)
    mag_np      = 20 * np.log(np.abs(f_shift_np) + 1)
    rekon_np    = np.abs(np.fft.ifft2(
        np.fft.ifftshift(f_shift_np)
    )).astype(np.uint8)

    # ---- OpenCV DFT ----
    # DFT_COMPLEX_OUTPUT: output berupa array dua channel (Re, Im)
    dft_cv       = cv2.dft(img_f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    # Geser hanya sumbu spasial (bukan sumbu channel)
    dft_shift_cv = np.fft.fftshift(dft_cv, axes=(0, 1))

    # Hitung magnitude dari dua channel menggunakan cv2.magnitude
    mag_cv_raw   = cv2.magnitude(dft_shift_cv[:, :, 0], dft_shift_cv[:, :, 1])
    mag_cv       = 20 * np.log(mag_cv_raw + 1)

    # Rekonstruksi: unshift kemudian cv2.idft
    dft_unshift_cv = np.fft.ifftshift(dft_shift_cv, axes=(0, 1))
    rekon_cv_raw   = cv2.idft(dft_unshift_cv,
                               flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    rekon_cv = np.clip(rekon_cv_raw, 0, 255).astype(np.uint8)

    # ---- Hitung selisih antara dua implementasi ----
    selisih_mag   = np.abs(mag_np - mag_cv)
    selisih_rekon = cv2.absdiff(rekon_np, rekon_cv).astype(np.float32)

    print(f"  Shape DFT NumPy  : {f_np.shape} (complex)")
    print(f"  Shape DFT OpenCV : {dft_cv.shape} (2-channel float32)")
    print(f"  Selisih magnitude max   : {selisih_mag.max():.4f}")
    print(f"  Selisih rekonstruksi max: {selisih_rekon.max():.4f}")

    # ---- Visualisasi 8 panel ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(
        "Demo 3: OpenCV cv2.dft() vs NumPy fft2()\n"
        "Perbandingan Magnitude Spectrum dan Rekonstruksi",
        fontsize=13, fontweight="bold"
    )

    axes[0, 0].imshow(gambar_gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Gambar Asli")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mag_np, cmap="magma")
    axes[0, 1].set_title("Magnitude Spectrum\n(NumPy fft2)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mag_cv, cmap="magma")
    axes[0, 2].set_title("Magnitude Spectrum\n(OpenCV cv2.dft)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(selisih_mag, cmap="hot")
    axes[0, 3].set_title(f"Selisih Magnitude\n(max={selisih_mag.max():.4f})")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(rekon_np, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Rekonstruksi NumPy\n(ifft2)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rekon_cv, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Rekonstruksi OpenCV\n(cv2.idft)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(selisih_rekon, cmap="gray")
    axes[1, 2].set_title(f"Selisih Rekonstruksi\n(max={selisih_rekon.max():.4f})")
    axes[1, 2].axis("off")

    # Histogram distribusi selisih (skala log agar detil terlihat)
    axes[1, 3].hist(selisih_rekon.ravel(), bins=50,
                    color="coral", edgecolor="darkred")
    axes[1, 3].set_title("Distribusi Selisih Rekonstruksi")
    axes[1, 3].set_xlabel("Nilai Selisih")
    axes[1, 3].set_ylabel("Jumlah Piksel")
    axes[1, 3].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_demo3_opencv_vs_numpy.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Tersimpan: 18_demo3_opencv_vs_numpy.png")


# ---------------------------------------------------------------------------
# Demo 4: Sinyal Sinusoidal Sintetis dan Spike di Spectrum
# ---------------------------------------------------------------------------
def demo_sinusoidal_sintetis():
    """
    Membuat gambar sintetis dari kombinasi gelombang sinusoidal 2D dan
    mengamati spike (titik diskrit) yang muncul di magnitude spectrum.

    Teori Fourier untuk sinyal cosinus:
        f(x) = A * cos(2*pi*f0*x)
        -> spike di +f0 dan -f0 (simetri Hermitian)

        Untuk gambar 2D:
        f(x,y) = A * cos(2*pi*(fx*x/W + fy*y/H))
        -> spike di (fx, fy) dan (-fx, -fy) pada magnitude spectrum

    Percobaan ini menampilkan 4 gambar sinusoidal dengan frekuensi berbeda
    beserta magnitude spectrum masing-masing untuk memverifikasi teori.
    """
    print("\n[Demo 4] Sinyal Sinusoidal Sintetis dan Spike di Spectrum")

    tinggi, lebar = 256, 256
    x = np.arange(lebar, dtype=np.float32)
    y = np.arange(tinggi, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    def hitung_spectrum(gambar):
        """Hitung magnitude spectrum (skala log) setelah fftshift."""
        f_shift = np.fft.fftshift(np.fft.fft2(gambar.astype(np.float32)))
        return 20 * np.log(np.abs(f_shift) + 1)

    # Sinyal 1: gelombang horizontal frekuensi rendah (5 siklus/lebar)
    # Spike muncul di kiri-kanan pusat pada sumbu horizontal
    sin1 = 128 + 100 * np.cos(2 * np.pi * 5 * xx / lebar)

    # Sinyal 2: gelombang vertikal frekuensi lebih tinggi (15 siklus/tinggi)
    # Spike muncul di atas-bawah pusat pada sumbu vertikal
    sin2 = 128 + 100 * np.cos(2 * np.pi * 15 * yy / tinggi)

    # Sinyal 3: gelombang diagonal (5 siklus di x dan y sekaligus)
    # Spike muncul di arah diagonal
    sin3 = 128 + 80 * np.cos(
        2 * np.pi * 5 * xx / lebar + 2 * np.pi * 5 * yy / tinggi
    )

    # Sinyal 4: superposisi tiga frekuensi berbeda
    # Spectrum menampilkan beberapa pasang spike sesuai frekuensi komponennya
    sin4 = (
        128
        + 60 * np.cos(2 * np.pi * 3  * xx / lebar)
        + 40 * np.cos(2 * np.pi * 10 * yy / tinggi)
        + 30 * np.cos(2 * np.pi * 7  * xx / lebar
                      + 2 * np.pi * 7 * yy / tinggi)
    )

    gambar_list = [sin1, sin2, sin3, sin4]
    judul_list  = [
        "Gelombang Horizontal\nfx=5, fy=0",
        "Gelombang Vertikal\nfx=0, fy=15",
        "Gelombang Diagonal\nfx=5, fy=5",
        "Superposisi 3 Frekuensi\n{fx,fy}={(3,0),(0,10),(7,7)}"
    ]

    # Normalisasi ke uint8 untuk tampilan yang konsisten
    gambar_uint8 = [np.clip(g, 0, 255).astype(np.uint8) for g in gambar_list]

    # ---- Visualisasi 8 panel (2 baris x 4 kolom) ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(
        "Demo 4: Sinyal Sinusoidal Sintetis => Spike di Magnitude Spectrum\n"
        "Setiap frekuensi cosinus menghasilkan dua spike simetris di spectrum",
        fontsize=13, fontweight="bold"
    )

    for idx, (gambar, judul) in enumerate(zip(gambar_uint8, judul_list)):
        spectrum = hitung_spectrum(gambar)

        # Baris atas: gambar sinusoidal dalam domain spasial
        axes[0, idx].imshow(gambar, cmap="gray", vmin=0, vmax=255)
        axes[0, idx].set_title(judul, fontsize=10)
        axes[0, idx].axis("off")

        # Baris bawah: magnitude spectrum -> spike terlihat jelas
        im = axes[1, idx].imshow(spectrum, cmap="inferno")
        axes[1, idx].set_title("Magnitude Spectrum\n(skala log)")
        axes[1, idx].axis("off")

        # Tandai posisi pusat dengan tanda + putih (DC component)
        cx_s = spectrum.shape[1] // 2
        cy_s = spectrum.shape[0] // 2
        axes[1, idx].plot(cx_s, cy_s, "w+", markersize=14, markeredgewidth=2)

    # Tambahkan colorbar pada spectrum terakhir sebagai referensi skala
    plt.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04,
                 label="Magnitude (log)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_demo4_sinusoidal_spectrum.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Tersimpan: 18_demo4_sinusoidal_spectrum.png")


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------
def main():
    """
    Fungsi utama yang menjalankan seluruh demonstrasi Transformasi Fourier.
    """
    print("=" * 70)
    print("  Praktikum 18: Transformasi Fourier dan Analisis Domain Frekuensi")
    print("=" * 70)
    print(f"  Output directory : {OUTPUT_DIR}")

    # Muat gambar dari disk atau buat gambar sintetis
    gambar_gray, label_sumber = muat_atau_buat_gambar()
    print(f"  Ukuran gambar    : {gambar_gray.shape[1]}x{gambar_gray.shape[0]} piksel")

    # Jalankan semua demo secara berurutan
    demo_magnitude_spectrum(gambar_gray, label_sumber)  # Demo 1
    demo_filter_frekuensi(gambar_gray)                  # Demo 2
    demo_opencv_dft(gambar_gray)                        # Demo 3
    demo_sinusoidal_sintetis()                          # Demo 4

    print("\n" + "=" * 70)
    print("  Semua demo selesai. Hasil tersimpan di folder output/")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
