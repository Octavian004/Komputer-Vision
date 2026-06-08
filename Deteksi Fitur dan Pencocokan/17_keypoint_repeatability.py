"""
==========================================================================
PERCOBAAN 17: ANALISIS REPEATABILITY KEYPOINT
==========================================================================
Program ini mengukur repeatability keypoint: seberapa banyak keypoint
yang terdeteksi kembali setelah gambar mengalami transformasi seperti
rotasi, skala, blur, noise, dan perubahan brightness.
Repeatability = keypoint yang cocok / total keypoint di gambar transformasi.

Konsep yang dipelajari:
- Repeatability: metrik fundamental untuk mengevaluasi detektor fitur
- Pengaruh berbagai transformasi terhadap stabilitas keypoint
- Perbandingan robustness antar detektor: SIFT, ORB, AKAZE, FAST
- Heatmap repeatability: visualisasi performa per detektor per transformasi
- Trade-off antara jumlah keypoint dan repeatability
- Identifikasi detektor terbaik untuk setiap jenis transformasi

Fungsi utama yang dipelajari:
- cv2.SIFT_create()               : Detektor SIFT (scale-invariant)
- cv2.ORB_create()                 : Detektor ORB (binary, cepat)
- cv2.AKAZE_create()               : Detektor AKAZE (nonlinear scale)
- cv2.FastFeatureDetector_create() : Detektor FAST (corner, sangat cepat)
- cv2.BFMatcher()                  : Matcher untuk menghitung korespondensi
- detectAndCompute()               : Deteksi + deskripsi fitur

Hasil: Tabel repeatability, chart perbandingan, dan heatmap komprehensif
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan fitur
import cv2

# Mengimpor NumPy untuk operasi array dan matriks numerik
import numpy as np

# Mengimpor os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk menyimpan visualisasi hasil
import matplotlib.pyplot as plt

# Mengimpor time untuk mengukur waktu pemrosesan
import time

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan judul percobaan
print("=" * 60)
print("PERCOBAAN 17: ANALISIS REPEATABILITY KEYPOINT")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar Referensi
# ============================================================

# Menampilkan header bagian
print("\n--- 1. Memuat Gambar Referensi ---")

# Membaca gambar bangunan sebagai gambar referensi
img_ref = cv2.imread(os.path.join(IMAGE_DIR, "bangunan.jpg"))

# Memeriksa apakah gambar berhasil dimuat
if img_ref is None:
    print("[ERROR] bangunan.jpg tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan ukuran gambar referensi
print(f"[INFO] Ukuran gambar referensi: {img_ref.shape}")

# Mengkonversi gambar referensi ke grayscale
gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. Mendefinisikan Detektor Fitur
# ============================================================

# Menampilkan header bagian
print("\n--- 2. Menyiapkan Detektor Fitur ---")

# Membuat detektor SIFT
sift = cv2.SIFT_create()

# Membuat detektor ORB
orb = cv2.ORB_create(nfeatures=1000)

# Membuat detektor AKAZE
akaze = cv2.AKAZE_create()

# Membuat detektor FAST (hanya keypoint, tanpa deskriptor)
fast = cv2.FastFeatureDetector_create()

# Mendefinisikan dictionary detektor dengan matcher yang sesuai
detektor_dict = {
    'SIFT': {'detektor': sift, 'norm': cv2.NORM_L2},
    'ORB': {'detektor': orb, 'norm': cv2.NORM_HAMMING},
    'AKAZE': {'detektor': akaze, 'norm': cv2.NORM_HAMMING},
}

# Menampilkan detektor yang disiapkan
print(f"[INFO] Detektor yang akan diuji: {list(detektor_dict.keys())} + FAST")


# Mendefinisikan fungsi untuk menghitung keypoint FAST dengan deskriptor ORB
def fast_detect_compute(gray_img):
    """Mendeteksi keypoint FAST dan menghitung deskriptor ORB."""
    # Mendeteksi keypoint menggunakan FAST
    kp = fast.detect(gray_img, None)

    # Membatasi jumlah keypoint menjadi 1000 terkuat
    kp = sorted(kp, key=lambda x: x.response, reverse=True)[:1000]

    # Menghitung deskriptor ORB dari keypoint FAST
    kp, des = orb.compute(gray_img, kp)

    # Mengembalikan keypoint dan deskriptor
    return kp, des


# ============================================================
# 3. Mendefinisikan Transformasi
# ============================================================

# Menampilkan header bagian
print("\n--- 3. Mendefinisikan Transformasi ---")


# Mendefinisikan fungsi untuk merotasi gambar
def transformasi_rotasi(img, sudut):
    """Merotasi gambar dengan sudut tertentu."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, sudut, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=200)
    return rotated


# Mendefinisikan fungsi untuk mengubah skala gambar
def transformasi_skala(img, faktor):
    """Mengubah skala gambar lalu resize kembali ke ukuran asli."""
    h, w = img.shape[:2]
    new_w = int(w * faktor)
    new_h = int(h * faktor)
    scaled = cv2.resize(img, (new_w, new_h))
    result = cv2.resize(scaled, (w, h))
    return result


# Mendefinisikan fungsi untuk menambahkan blur
def transformasi_blur(img, kernel_size):
    """Menambahkan Gaussian blur pada gambar."""
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred


# Mendefinisikan fungsi untuk menambahkan noise
def transformasi_noise(img, sigma):
    """Menambahkan Gaussian noise pada gambar."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


# Mendefinisikan fungsi untuk mengubah brightness
def transformasi_brightness(img, delta):
    """Mengubah brightness gambar."""
    adjusted = cv2.convertScaleAbs(img, alpha=1.0, beta=delta)
    return adjusted


# Mendefinisikan semua transformasi yang akan diuji
transformasi_semua = {
    'Rotasi 10°': lambda img: transformasi_rotasi(img, 10),
    'Rotasi 30°': lambda img: transformasi_rotasi(img, 30),
    'Rotasi 45°': lambda img: transformasi_rotasi(img, 45),
    'Skala 0.5x': lambda img: transformasi_skala(img, 0.5),
    'Skala 1.5x': lambda img: transformasi_skala(img, 1.5),
    'Blur k=5': lambda img: transformasi_blur(img, 5),
    'Blur k=11': lambda img: transformasi_blur(img, 11),
    'Noise σ=20': lambda img: transformasi_noise(img, 20),
    'Noise σ=50': lambda img: transformasi_noise(img, 50),
    'Bright +60': lambda img: transformasi_brightness(img, 60),
    'Bright -60': lambda img: transformasi_brightness(img, -60),
}

# Menampilkan jumlah transformasi
print(f"[INFO] Jumlah transformasi: {len(transformasi_semua)}")

# ============================================================
# 4. Menghitung Repeatability untuk Setiap Kombinasi
# ============================================================

# Menampilkan header bagian
print("\n--- 4. Menghitung Repeatability ---")

# Menyiapkan dictionary untuk menyimpan hasil repeatability
hasil_repeatability = {}

# Mendefinisikan nama-nama detektor (termasuk FAST)
nama_detektor = ['SIFT', 'ORB', 'AKAZE', 'FAST+ORB']

# Melakukan iterasi untuk setiap detektor
for nama_det in nama_detektor:
    # Menginisialisasi dictionary untuk detektor ini
    hasil_repeatability[nama_det] = {}

    # Menampilkan proses detektor saat ini
    print(f"\n  === Detektor: {nama_det} ===")

    # Mendeteksi keypoint dan deskriptor pada gambar referensi
    if nama_det == 'FAST+ORB':
        kp_ref, des_ref = fast_detect_compute(gray_ref)
        norm_type = cv2.NORM_HAMMING
    else:
        det_info = detektor_dict[nama_det]
        kp_ref, des_ref = det_info['detektor'].detectAndCompute(gray_ref, None)
        norm_type = det_info['norm']

    # Menampilkan jumlah keypoint referensi
    jumlah_kp_ref = len(kp_ref) if kp_ref else 0
    print(f"  Keypoint referensi: {jumlah_kp_ref}")

    # Memeriksa apakah deskriptor referensi valid
    if des_ref is None or jumlah_kp_ref == 0:
        print(f"  [SKIP] Tidak ada deskriptor untuk {nama_det}")
        for nama_trans in transformasi_semua:
            hasil_repeatability[nama_det][nama_trans] = 0.0
        continue

    # Membuat matcher yang sesuai
    bf_matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    # Melakukan iterasi untuk setiap transformasi
    for nama_trans, fungsi_trans in transformasi_semua.items():
        # Menerapkan transformasi pada gambar referensi
        img_trans = fungsi_trans(gray_ref)

        # Mendeteksi keypoint dan deskriptor pada gambar transformasi
        if nama_det == 'FAST+ORB':
            kp_trans, des_trans = fast_detect_compute(img_trans)
        else:
            kp_trans, des_trans = det_info['detektor'].detectAndCompute(img_trans, None)

        # Menghitung jumlah keypoint pada gambar transformasi
        jumlah_kp_trans = len(kp_trans) if kp_trans else 0

        # Memeriksa apakah ada deskriptor untuk dicocokkan
        if des_trans is None or jumlah_kp_trans == 0:
            hasil_repeatability[nama_det][nama_trans] = 0.0
            continue

        # Melakukan KNN matching
        try:
            matches = bf_matcher.knnMatch(des_ref, des_trans, k=2)
        except cv2.error:
            hasil_repeatability[nama_det][nama_trans] = 0.0
            continue

        # Menerapkan ratio test untuk mendapatkan good matches
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Menghitung repeatability rate
        repeatability = len(good_matches) / max(jumlah_kp_trans, 1)

        # Membatasi repeatability ke range [0, 1]
        repeatability = min(repeatability, 1.0)

        # Menyimpan hasil repeatability
        hasil_repeatability[nama_det][nama_trans] = repeatability

        # Menampilkan hasil untuk kombinasi ini
        print(f"    {nama_trans}: {repeatability:.3f} "
              f"({len(good_matches)}/{jumlah_kp_trans})")

# ============================================================
# 5. Visualisasi Tabel Repeatability
# ============================================================

# Menampilkan header bagian
print("\n--- 5. Visualisasi Tabel Repeatability ---")

# Mengekstrak nama-nama transformasi
nama_trans_list = list(transformasi_semua.keys())

# Menyiapkan data tabel
data_tabel = []
for det in nama_detektor:
    baris = [det]
    for trans in nama_trans_list:
        val = hasil_repeatability[det].get(trans, 0.0)
        baris.append(f"{val:.3f}")
    data_tabel.append(baris)

# Membuat figure untuk tabel
fig, ax = plt.subplots(figsize=(20, 6))

# Menyembunyikan axes
ax.axis('off')

# Menyiapkan header kolom
kolom_header = ['Detektor'] + [t.replace(' ', '\n') for t in nama_trans_list]

# Membuat tabel matplotlib
tabel = ax.table(cellText=data_tabel, colLabels=kolom_header, loc='center', cellLoc='center')

# Mengatur ukuran font tabel
tabel.auto_set_font_size(False)
tabel.set_fontsize(8)

# Mengatur tinggi baris
tabel.scale(1, 1.8)

# Mewarnai header
for j in range(len(kolom_header)):
    tabel[0, j].set_facecolor('#4472C4')
    tabel[0, j].set_text_props(color='white', fontweight='bold')

# Mewarnai sel berdasarkan nilai repeatability
for i in range(len(nama_detektor)):
    for j in range(len(nama_trans_list)):
        val = hasil_repeatability[nama_detektor[i]].get(nama_trans_list[j], 0.0)
        # Menentukan warna berdasarkan tingkat repeatability
        if val >= 0.5:
            warna = '#C6EFCE'  # hijau
        elif val >= 0.3:
            warna = '#FFEB9C'  # kuning
        else:
            warna = '#FFC7CE'  # merah
        tabel[i + 1, j + 1].set_facecolor(warna)

# Menambahkan judul
ax.set_title('Tabel Repeatability Keypoint per Detektor per Transformasi',
             fontsize=14, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Menyimpan tabel repeatability
output_path_tabel = os.path.join(OUTPUT_DIR, "17_repeatability_table.png")
plt.savefig(output_path_tabel, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_tabel}")

# ============================================================
# 6. Chart Repeatability per Transformasi
# ============================================================

# Menampilkan header bagian
print("\n--- 6. Chart Repeatability ---")

# Mengelompokkan transformasi berdasarkan jenis
kelompok_trans = {
    'Rotasi': ['Rotasi 10°', 'Rotasi 30°', 'Rotasi 45°'],
    'Skala': ['Skala 0.5x', 'Skala 1.5x'],
    'Blur': ['Blur k=5', 'Blur k=11'],
    'Noise': ['Noise σ=20', 'Noise σ=50'],
    'Brightness': ['Bright +60', 'Bright -60'],
}

# Membuat figure dengan subplot untuk setiap kelompok transformasi
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Mendefinisikan warna untuk setiap detektor
warna_detektor = {'SIFT': '#4472C4', 'ORB': '#ED7D31', 'AKAZE': '#70AD47', 'FAST+ORB': '#FFC000'}

# Melakukan iterasi untuk setiap kelompok transformasi
for idx, (kelompok_nama, trans_list) in enumerate(kelompok_trans.items()):
    # Menentukan posisi subplot
    row = idx // 3
    col = idx % 3

    # Mendefinisikan posisi bar pada sumbu x
    x_pos = np.arange(len(trans_list))

    # Mendefinisikan lebar bar
    lebar = 0.2

    # Menggambar bar untuk setiap detektor
    for i, det_nama in enumerate(nama_detektor):
        # Mengekstrak nilai repeatability untuk transformasi dalam kelompok ini
        nilai = [hasil_repeatability[det_nama].get(t, 0.0) for t in trans_list]

        # Menggambar bar chart
        axes[row, col].bar(x_pos + i * lebar, nilai, lebar,
                           label=det_nama, color=warna_detektor[det_nama])

    # Mengatur label sumbu x
    axes[row, col].set_xticks(x_pos + lebar * 1.5)
    axes[row, col].set_xticklabels([t.replace(' ', '\n') for t in trans_list], fontsize=9)

    # Mengatur label sumbu y
    axes[row, col].set_ylabel('Repeatability', fontsize=10)

    # Mengatur judul subplot
    axes[row, col].set_title(f'Repeatability - {kelompok_nama}', fontsize=12, fontweight='bold')

    # Menambahkan legenda
    axes[row, col].legend(fontsize=8)

    # Mengatur range sumbu y
    axes[row, col].set_ylim(0, 1.0)

    # Menambahkan grid horizontal
    axes[row, col].grid(axis='y', alpha=0.3)

# Menghitung rata-rata repeatability per detektor di subplot terakhir
ax_rata = axes[1, 2]

# Menghitung rata-rata per detektor
rata_per_det = {}
for det_nama in nama_detektor:
    semua_val = [hasil_repeatability[det_nama].get(t, 0.0) for t in nama_trans_list]
    rata_per_det[det_nama] = np.mean(semua_val)

# Menggambar bar chart rata-rata
bar_colors = [warna_detektor[d] for d in nama_detektor]
ax_rata.bar(nama_detektor, [rata_per_det[d] for d in nama_detektor], color=bar_colors)

# Menambahkan label nilai di atas bar
for i, det in enumerate(nama_detektor):
    ax_rata.text(i, rata_per_det[det] + 0.02, f'{rata_per_det[det]:.3f}',
                 ha='center', fontsize=10, fontweight='bold')

# Mengatur judul subplot rata-rata
ax_rata.set_title('Rata-rata Repeatability\n(Semua Transformasi)', fontsize=12, fontweight='bold')
ax_rata.set_ylabel('Repeatability', fontsize=10)
ax_rata.set_ylim(0, 1.0)
ax_rata.grid(axis='y', alpha=0.3)

# Menambahkan judul utama
fig.suptitle('Percobaan 17: Repeatability Keypoint per Transformasi', fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan chart repeatability
output_path_chart = os.path.join(OUTPUT_DIR, "17_repeatability_chart.png")
plt.savefig(output_path_chart, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_chart}")

# ============================================================
# 7. Heatmap Repeatability
# ============================================================

# Menampilkan header bagian
print("\n--- 7. Heatmap Repeatability ---")

# Menyiapkan matriks data untuk heatmap
data_heatmap = np.zeros((len(nama_detektor), len(nama_trans_list)))

# Mengisi matriks dengan nilai repeatability
for i, det in enumerate(nama_detektor):
    for j, trans in enumerate(nama_trans_list):
        data_heatmap[i, j] = hasil_repeatability[det].get(trans, 0.0)

# Membuat figure untuk heatmap
fig, ax = plt.subplots(figsize=(16, 6))

# Menampilkan heatmap menggunakan imshow
im = ax.imshow(data_heatmap, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Mengatur label sumbu x (transformasi)
ax.set_xticks(range(len(nama_trans_list)))
ax.set_xticklabels([t.replace(' ', '\n') for t in nama_trans_list], fontsize=9, rotation=0)

# Mengatur label sumbu y (detektor)
ax.set_yticks(range(len(nama_detektor)))
ax.set_yticklabels(nama_detektor, fontsize=11)

# Menambahkan nilai repeatability di setiap sel heatmap
for i in range(len(nama_detektor)):
    for j in range(len(nama_trans_list)):
        nilai = data_heatmap[i, j]
        # Menentukan warna teks berdasarkan latar belakang
        warna_teks = 'white' if nilai < 0.3 else 'black'
        ax.text(j, i, f'{nilai:.2f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color=warna_teks)

# Menambahkan colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Repeatability Rate', fontsize=11)

# Mengatur judul
ax.set_title('Heatmap Repeatability Keypoint\n(Hijau=Tinggi, Merah=Rendah)',
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan heatmap repeatability
output_path_heatmap = os.path.join(OUTPUT_DIR, "17_repeatability_heatmap.png")
plt.savefig(output_path_heatmap, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_heatmap}")

# ============================================================
# 8. Identifikasi Detektor Terbaik per Transformasi
# ============================================================

# Menampilkan header bagian
print("\n--- 8. Detektor Terbaik per Jenis Transformasi ---")

# Melakukan iterasi untuk setiap kelompok transformasi
for kelompok_nama, trans_list in kelompok_trans.items():
    # Menghitung rata-rata repeatability per detektor untuk kelompok ini
    skor_kelompok = {}
    for det in nama_detektor:
        vals = [hasil_repeatability[det].get(t, 0.0) for t in trans_list]
        skor_kelompok[det] = np.mean(vals)

    # Mencari detektor dengan skor tertinggi
    det_terbaik = max(skor_kelompok, key=skor_kelompok.get)

    # Menampilkan detektor terbaik untuk kelompok ini
    print(f"  {kelompok_nama}: {det_terbaik} (rata-rata={skor_kelompok[det_terbaik]:.3f})")

# ============================================================
# 9. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 17: ANALISIS REPEATABILITY KEYPOINT")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan repeatability
print("1. Repeatability mengukur seberapa stabil detektor dalam")
print("   menemukan keypoint yang sama setelah transformasi.")

# Menampilkan temuan tentang SIFT
print("2. SIFT umumnya memiliki repeatability tinggi karena")
print("   dirancang untuk scale dan rotation invariance.")

# Menampilkan temuan tentang ORB
print("3. ORB cepat tapi repeatability lebih rendah pada")
print("   transformasi besar (rotasi/skala ekstrem).")

# Menampilkan temuan blur dan noise
print("4. Blur dan noise menurunkan repeatability semua detektor,")
print("   namun tingkat penurunan berbeda per detektor.")

# Menampilkan rekomendasi
print("5. Pilihan detektor tergantung pada jenis transformasi")
print("   yang diharapkan dalam aplikasi target.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 17_repeatability_table.png")
print("  - 17_repeatability_chart.png")
print("  - 17_repeatability_heatmap.png")

# Menampilkan garis penutup
print("=" * 60)
