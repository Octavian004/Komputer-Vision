"""
==========================================================================
PERCOBAAN 2: OPENCV STITCHER API
==========================================================================
Program ini menggunakan cv2.Stitcher untuk membuat panorama otomatis
dengan pipeline lengkap bawaan OpenCV. Stitcher API menangani seluruh
proses dari deteksi fitur hingga blending secara otomatis.

Konsep yang dipelajari:
- Penggunaan cv2.Stitcher sebagai high-level API stitching
- Perbedaan mode PANORAMA vs SCANS
- Penanganan error dari Stitcher (status codes)
- Cropping border hitam pada hasil panorama
- Perbandingan kecepatan dan kualitas dengan pipeline manual

Fungsi utama yang dipelajari:
- cv2.Stitcher_create()             : Membuat objek Stitcher
- stitcher.stitch()                 : Menjalankan pipeline stitching lengkap
- cv2.Stitcher_PANORAMA             : Mode panorama (kamera berputar)
- cv2.Stitcher_SCANS                : Mode scan (translasi/dokumen flat)
- cv2.findNonZero()                 : Mencari piksel non-zero untuk crop
- cv2.boundingRect()                : Mendapatkan bounding rectangle
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan stitching
import cv2

# Mengimpor library NumPy untuk operasi array dan matriks
import numpy as np

# Mengimpor library os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi dan menyimpan perbandingan
import matplotlib.pyplot as plt

# Mengimpor modul time untuk mengukur waktu eksekusi stitching
import time

# ============================================================
# SETUP PATH DIREKTORI
# ============================================================

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HEADER PROGRAM
# ============================================================
print("=" * 65)
print("PERCOBAAN 2: OPENCV STITCHER API")
print("=" * 65)


# ============================================================
# FUNGSI HELPER: Interpretasi Status Stitcher
# ============================================================
def interpretasi_status(status):
    """
    Menginterpretasikan kode status dari cv2.Stitcher.stitch().
    Status menunjukkan apakah stitching berhasil atau gagal beserta alasannya.
    """
    # Dictionary mapping kode status ke deskripsi
    status_map = {
        cv2.Stitcher_OK: "OK - Stitching berhasil",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERROR - Perlu lebih banyak gambar",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERROR - Estimasi homography gagal",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERROR - Penyesuaian parameter kamera gagal"
    }
    # Mengembalikan deskripsi sesuai kode status
    return status_map.get(status, f"ERROR - Kode tidak dikenal: {status}")


# ============================================================
# FUNGSI HELPER: Crop Border Hitam dari Panorama
# ============================================================
def crop_border_hitam(img, threshold=1):
    """
    Menghilangkan border hitam dari hasil stitching.
    Border hitam muncul karena warping yang membuat area kosong.

    Parameter:
    - img       : Gambar hasil stitching
    - threshold : Nilai minimum piksel dianggap non-hitam (default=1)
    """
    # Mengkonversi gambar ke grayscale untuk deteksi area hitam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Membuat mask: piksel yang nilainya di atas threshold dianggap non-hitam
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Menerapkan morphological closing untuk mengisi lubang kecil dalam mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Mencari kontur terbesar sebagai area yang valid
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Jika tidak ada kontur ditemukan, kembalikan gambar asli
    if not contours:
        return img

    # Mengambil kontur terbesar (area panorama yang valid)
    largest_contour = max(contours, key=cv2.contourArea)

    # Mendapatkan bounding rectangle dari kontur terbesar
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Memotong gambar sesuai bounding rectangle
    cropped = img[y:y + h, x:x + w]

    return cropped


# ============================================================
# LANGKAH 1: Memuat Gambar Panorama Outdoor (3 Gambar)
# ============================================================
print("\n[LANGKAH 1] Memuat gambar panorama outdoor...")

# Memuat 3 gambar panorama outdoor yang overlapping
outdoor_images = []
for i in range(1, 4):
    # Membaca setiap gambar panorama outdoor
    path = os.path.join(IMAGE_DIR, f"panorama_outdoor_{i}.jpg")
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] panorama_outdoor_{i}.jpg tidak ditemukan!")
        exit()
    # Menambahkan gambar ke list
    outdoor_images.append(img)
    print(f"  panorama_outdoor_{i}.jpg: {img.shape[1]}x{img.shape[0]} piksel")

print(f"  Total gambar outdoor: {len(outdoor_images)}")

# ============================================================
# LANGKAH 2: Stitching dengan Mode PANORAMA
# ============================================================
print("\n[LANGKAH 2] Stitching mode PANORAMA...")

# Membuat objek Stitcher dengan mode PANORAMA
# Mode PANORAMA: diasumsikan kamera berputar di tempat (rotasi)
stitcher_panorama = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# Mengukur waktu eksekusi stitching
waktu_mulai = time.time()

# Menjalankan proses stitching pada kumpulan gambar
# stitch() mengembalikan tuple: (status, hasil_panorama)
status_pano, result_pano = stitcher_panorama.stitch(outdoor_images)

# Menghitung durasi proses stitching
waktu_pano = time.time() - waktu_mulai

# Menampilkan status hasil stitching
print(f"  Status: {interpretasi_status(status_pano)}")
print(f"  Waktu eksekusi: {waktu_pano:.3f} detik")

# Memeriksa apakah stitching berhasil
if status_pano == cv2.Stitcher_OK:
    # Menampilkan dimensi hasil panorama
    print(f"  Ukuran panorama: {result_pano.shape[1]}x{result_pano.shape[0]} piksel")

    # Menyimpan hasil panorama mode PANORAMA
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_panorama_mode_raw.jpg"), result_pano)
    print("  [OK] Hasil panorama mode PANORAMA disimpan.")

    # Melakukan crop border hitam pada hasil
    result_pano_cropped = crop_border_hitam(result_pano)
    print(f"  Ukuran setelah crop: {result_pano_cropped.shape[1]}x{result_pano_cropped.shape[0]}")

    # Menyimpan hasil yang sudah di-crop
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_panorama_mode_cropped.jpg"), result_pano_cropped)
    print("  [OK] Hasil panorama cropped disimpan.")
else:
    print("  [GAGAL] Stitching mode PANORAMA gagal!")
    result_pano_cropped = None

# ============================================================
# LANGKAH 3: Stitching dengan Mode SCANS
# ============================================================
print("\n[LANGKAH 3] Stitching mode SCANS...")

# Membuat objek Stitcher dengan mode SCANS
# Mode SCANS: diasumsikan kamera bertranslasi (cocok untuk dokumen/scan)
stitcher_scans = cv2.Stitcher_create(cv2.Stitcher_SCANS)

# Mengukur waktu eksekusi
waktu_mulai = time.time()

# Menjalankan stitching mode SCANS
status_scan, result_scan = stitcher_scans.stitch(outdoor_images)

# Menghitung durasi
waktu_scan = time.time() - waktu_mulai

# Menampilkan status
print(f"  Status: {interpretasi_status(status_scan)}")
print(f"  Waktu eksekusi: {waktu_scan:.3f} detik")

# Memeriksa hasil stitching mode SCANS
if status_scan == cv2.Stitcher_OK:
    print(f"  Ukuran panorama: {result_scan.shape[1]}x{result_scan.shape[0]} piksel")

    # Menyimpan hasil mode SCANS
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_scans_mode_raw.jpg"), result_scan)

    # Melakukan crop border hitam
    result_scan_cropped = crop_border_hitam(result_scan)
    print(f"  Ukuran setelah crop: {result_scan_cropped.shape[1]}x{result_scan_cropped.shape[0]}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_scans_mode_cropped.jpg"), result_scan_cropped)
    print("  [OK] Hasil mode SCANS disimpan.")
else:
    print(f"  [INFO] Mode SCANS gagal/tidak sesuai untuk gambar outdoor ini.")
    result_scan_cropped = None

# ============================================================
# LANGKAH 4: Stitching dengan Urutan Gambar Acak
# ============================================================
print("\n[LANGKAH 4] Stitching dengan urutan gambar diacak...")

# Membuat salinan list gambar dan membalik urutannya (3, 2, 1)
scrambled_images = outdoor_images[::-1]

# Stitcher seharusnya bisa menangani urutan acak secara otomatis
stitcher_scrambled = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# Menjalankan stitching pada gambar yang urutannya diacak
status_scrambled, result_scrambled = stitcher_scrambled.stitch(scrambled_images)

# Menampilkan status
print(f"  Urutan input: terbalik (3, 2, 1)")
print(f"  Status: {interpretasi_status(status_scrambled)}")

# Menyimpan hasil jika berhasil
if status_scrambled == cv2.Stitcher_OK:
    result_scrambled_cropped = crop_border_hitam(result_scrambled)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_scrambled_order_result.jpg"), result_scrambled_cropped)
    print(f"  Ukuran: {result_scrambled_cropped.shape[1]}x{result_scrambled_cropped.shape[0]}")
    print("  [OK] Stitcher berhasil menangani urutan acak!")
else:
    print("  [INFO] Stitcher gagal dengan urutan acak.")
    result_scrambled_cropped = None

# ============================================================
# LANGKAH 5: Stitching Gambar Indoor (4 Gambar)
# ============================================================
print("\n[LANGKAH 5] Stitching gambar panorama indoor (4 gambar)...")

# Memuat 4 gambar panorama indoor
indoor_images = []
for i in range(1, 5):
    path = os.path.join(IMAGE_DIR, f"panorama_indoor_{i}.jpg")
    img = cv2.imread(path)
    if img is not None:
        indoor_images.append(img)
        print(f"  panorama_indoor_{i}.jpg: {img.shape[1]}x{img.shape[0]} piksel")
    else:
        print(f"  [WARNING] panorama_indoor_{i}.jpg tidak ditemukan, dilewati.")

# Melakukan stitching pada gambar indoor
if len(indoor_images) >= 2:
    stitcher_indoor = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    waktu_mulai = time.time()
    status_indoor, result_indoor = stitcher_indoor.stitch(indoor_images)
    waktu_indoor = time.time() - waktu_mulai

    print(f"  Status: {interpretasi_status(status_indoor)}")
    print(f"  Waktu eksekusi: {waktu_indoor:.3f} detik")

    if status_indoor == cv2.Stitcher_OK:
        result_indoor_cropped = crop_border_hitam(result_indoor)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "02_indoor_panorama.jpg"), result_indoor_cropped)
        print(f"  Ukuran: {result_indoor_cropped.shape[1]}x{result_indoor_cropped.shape[0]}")
        print("  [OK] Panorama indoor disimpan.")
    else:
        result_indoor_cropped = None
        print("  [INFO] Stitching indoor gagal.")
else:
    print("  [WARNING] Tidak cukup gambar indoor untuk stitching.")
    result_indoor_cropped = None

# ============================================================
# LANGKAH 6: Error Handling - Gambar Tanpa Overlap
# ============================================================
print("\n[LANGKAH 6] Tes error handling: gambar tanpa overlap...")

# Mencoba stitching gambar yang tidak saling overlap
# Menggunakan gambar dari set berbeda yang tidak memiliki konten sama
non_overlap_imgs = []

# Gambar 1: dari alignment test
img_test1 = cv2.imread(os.path.join(IMAGE_DIR, "alignment_original.jpg"))
if img_test1 is not None:
    non_overlap_imgs.append(img_test1)

# Gambar 2: dari grid test
img_test2 = cv2.imread(os.path.join(IMAGE_DIR, "grid_test.jpg"))
if img_test2 is not None:
    non_overlap_imgs.append(img_test2)

# Mencoba stitching gambar tanpa overlap
if len(non_overlap_imgs) >= 2:
    stitcher_fail = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status_fail, result_fail = stitcher_fail.stitch(non_overlap_imgs)

    # Menampilkan status error
    print(f"  Status: {interpretasi_status(status_fail)}")
    print(f"  Penjelasan: Gambar tanpa overlap menyebabkan Stitcher gagal.")
    print(f"  Ini adalah perilaku yang diharapkan!")

    # Demonstrasi cara menangani error
    if status_fail != cv2.Stitcher_OK:
        print("  [INFO] Best practice: selalu periksa status sebelum menggunakan hasil.")
else:
    print("  [WARNING] Tidak cukup gambar untuk tes error handling.")

# ============================================================
# LANGKAH 7: Pengukuran Waktu - 3, 4, 5 Gambar
# ============================================================
print("\n[LANGKAH 7] Mengukur waktu stitching untuk jumlah gambar berbeda...")

# Memuat gambar panorama wide (5 gambar) untuk pengukuran waktu
wide_images = []
for i in range(1, 6):
    path = os.path.join(IMAGE_DIR, f"panorama_wide_{i}.jpg")
    img = cv2.imread(path)
    if img is not None:
        wide_images.append(img)

# Dictionary untuk menyimpan hasil pengukuran waktu
timing_results = {}

# Mengukur waktu stitching untuk jumlah gambar yang berbeda
for jumlah in [3, 4, 5]:
    if jumlah <= len(wide_images):
        # Mengambil subset gambar sesuai jumlah
        subset = wide_images[:jumlah]

        # Membuat Stitcher baru untuk setiap percobaan
        stitcher_time = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

        # Mengukur waktu eksekusi
        waktu_mulai = time.time()
        status_time, result_time = stitcher_time.stitch(subset)
        waktu_time = time.time() - waktu_mulai

        # Menyimpan hasil pengukuran
        timing_results[jumlah] = {
            'waktu': waktu_time,
            'status': status_time,
            'ukuran': result_time.shape if status_time == cv2.Stitcher_OK else None
        }

        # Menampilkan hasil
        status_str = "OK" if status_time == cv2.Stitcher_OK else "GAGAL"
        print(f"  {jumlah} gambar: {waktu_time:.3f} detik ({status_str})")

        # Menyimpan hasil jika berhasil
        if status_time == cv2.Stitcher_OK:
            cropped = crop_border_hitam(result_time)
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"02_wide_{jumlah}_gambar.jpg"),
                cropped
            )
    else:
        print(f"  [WARNING] Hanya {len(wide_images)} gambar tersedia, skip {jumlah}")

# ============================================================
# LANGKAH 8: Perbandingan Stitcher API vs Pipeline Manual
# ============================================================
print("\n[LANGKAH 8] Membandingkan Stitcher API vs pipeline manual...")

# Membaca hasil pipeline manual dari percobaan 01 (jika tersedia)
manual_result_path = os.path.join(OUTPUT_DIR, "01_stitching_raw.jpg")
manual_result = cv2.imread(manual_result_path)

# Melakukan stitching API pada gambar pasangan
pair_images = []
img_pair_left = cv2.imread(os.path.join(IMAGE_DIR, "pair_left.jpg"))
img_pair_right = cv2.imread(os.path.join(IMAGE_DIR, "pair_right.jpg"))

api_pair_result = None
if img_pair_left is not None and img_pair_right is not None:
    pair_images = [img_pair_left, img_pair_right]

    # Stitching API pada pasangan gambar
    stitcher_pair = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    waktu_mulai = time.time()
    status_pair, result_pair = stitcher_pair.stitch(pair_images)
    waktu_api = time.time() - waktu_mulai

    print(f"  Stitcher API (pair): {interpretasi_status(status_pair)}")
    print(f"  Waktu API: {waktu_api:.3f} detik")

    if status_pair == cv2.Stitcher_OK:
        api_pair_result = crop_border_hitam(result_pair)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "02_api_pair_result.jpg"), api_pair_result)
        print(f"  Ukuran hasil API: {api_pair_result.shape[1]}x{api_pair_result.shape[0]}")

# Mencetak perbandingan
print("\n  Perbandingan:")
if manual_result is not None:
    print(f"    Pipeline Manual : {manual_result.shape[1]}x{manual_result.shape[0]}")
else:
    print(f"    Pipeline Manual : (belum tersedia, jalankan 01_* terlebih dahulu)")
if api_pair_result is not None:
    print(f"    Stitcher API    : {api_pair_result.shape[1]}x{api_pair_result.shape[0]}")
else:
    print(f"    Stitcher API    : (stitching gagal pada pasangan ini)")

# ============================================================
# LANGKAH 9: Membuat Grid Perbandingan dengan Matplotlib
# ============================================================
print("\n[LANGKAH 9] Membuat grid perbandingan matplotlib...")

# --- Grid 1: Gambar input dan hasil panorama outdoor ---
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))

# Baris 1: 3 gambar input outdoor
for i in range(3):
    axes1[0, i].imshow(cv2.cvtColor(outdoor_images[i], cv2.COLOR_BGR2RGB))
    axes1[0, i].set_title(f"Input outdoor_{i+1}", fontsize=11)
    axes1[0, i].axis("off")

# Baris 2: Hasil stitching
# Subplot (1,0): Hasil panorama mode PANORAMA
if result_pano_cropped is not None:
    axes1[1, 0].imshow(cv2.cvtColor(result_pano_cropped, cv2.COLOR_BGR2RGB))
    axes1[1, 0].set_title(f"Mode PANORAMA ({waktu_pano:.2f}s)", fontsize=11)
else:
    axes1[1, 0].text(0.5, 0.5, "GAGAL", ha='center', va='center', fontsize=16)
    axes1[1, 0].set_title("Mode PANORAMA (gagal)", fontsize=11)
axes1[1, 0].axis("off")

# Subplot (1,1): Hasil mode SCANS
if result_scan_cropped is not None:
    axes1[1, 1].imshow(cv2.cvtColor(result_scan_cropped, cv2.COLOR_BGR2RGB))
    axes1[1, 1].set_title(f"Mode SCANS ({waktu_scan:.2f}s)", fontsize=11)
else:
    axes1[1, 1].text(0.5, 0.5, "GAGAL", ha='center', va='center', fontsize=16)
    axes1[1, 1].set_title("Mode SCANS (gagal)", fontsize=11)
axes1[1, 1].axis("off")

# Subplot (1,2): Hasil urutan acak
if result_scrambled_cropped is not None:
    axes1[1, 2].imshow(cv2.cvtColor(result_scrambled_cropped, cv2.COLOR_BGR2RGB))
    axes1[1, 2].set_title("Urutan Diacak", fontsize=11)
else:
    axes1[1, 2].text(0.5, 0.5, "GAGAL", ha='center', va='center', fontsize=16)
    axes1[1, 2].set_title("Urutan Diacak (gagal)", fontsize=11)
axes1[1, 2].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 2: OpenCV Stitcher API - Outdoor Panorama",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid
plt.savefig(os.path.join(OUTPUT_DIR, "02_grid_outdoor_panorama.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid outdoor panorama disimpan.")
plt.close()

# --- Grid 2: Indoor panorama dan perbandingan API vs Manual ---
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

# Subplot (0,0): Gambar indoor input (montage)
if indoor_images:
    # Menggabungkan gambar indoor secara horizontal untuk ditampilkan
    indoor_display = indoor_images[0]
    for im in indoor_images[1:]:
        # Meresize agar tinggi sama sebelum menggabungkan
        h_target = indoor_display.shape[0]
        w_new = int(im.shape[1] * h_target / im.shape[0])
        im_resized = cv2.resize(im, (w_new, h_target))
        indoor_display = np.hstack([indoor_display, im_resized])
    axes2[0, 0].imshow(cv2.cvtColor(indoor_display, cv2.COLOR_BGR2RGB))
    axes2[0, 0].set_title(f"Input Indoor ({len(indoor_images)} gambar)", fontsize=11)
else:
    axes2[0, 0].text(0.5, 0.5, "Tidak tersedia", ha='center', va='center')
axes2[0, 0].axis("off")

# Subplot (0,1): Hasil indoor panorama
if result_indoor_cropped is not None:
    axes2[0, 1].imshow(cv2.cvtColor(result_indoor_cropped, cv2.COLOR_BGR2RGB))
    axes2[0, 1].set_title("Indoor Panorama", fontsize=11)
else:
    axes2[0, 1].text(0.5, 0.5, "GAGAL", ha='center', va='center', fontsize=16)
    axes2[0, 1].set_title("Indoor Panorama (gagal)", fontsize=11)
axes2[0, 1].axis("off")

# Subplot (1,0): Hasil pipeline manual
if manual_result is not None:
    axes2[1, 0].imshow(cv2.cvtColor(manual_result, cv2.COLOR_BGR2RGB))
    axes2[1, 0].set_title("Pipeline Manual (Percobaan 1)", fontsize=11)
else:
    axes2[1, 0].text(0.5, 0.5, "Jalankan 01_* dulu", ha='center', va='center')
    axes2[1, 0].set_title("Pipeline Manual (belum ada)", fontsize=11)
axes2[1, 0].axis("off")

# Subplot (1,1): Hasil Stitcher API pada pair
if api_pair_result is not None:
    axes2[1, 1].imshow(cv2.cvtColor(api_pair_result, cv2.COLOR_BGR2RGB))
    axes2[1, 1].set_title("Stitcher API (pair)", fontsize=11)
else:
    axes2[1, 1].text(0.5, 0.5, "GAGAL", ha='center', va='center', fontsize=16)
    axes2[1, 1].set_title("Stitcher API pair (gagal)", fontsize=11)
axes2[1, 1].axis("off")

# Menambahkan judul utama
plt.suptitle("Percobaan 2: Perbandingan Indoor, Manual vs API",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan grid kedua
plt.savefig(os.path.join(OUTPUT_DIR, "02_grid_indoor_dan_perbandingan.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid indoor dan perbandingan disimpan.")
plt.close()

# --- Grid 3: Perbandingan waktu stitching ---
if timing_results:
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot kiri: Bar chart waktu stitching
    jumlah_list = sorted(timing_results.keys())
    waktu_list = [timing_results[j]['waktu'] for j in jumlah_list]
    warna_bar = ['#4CAF50' if timing_results[j]['status'] == cv2.Stitcher_OK else '#F44336'
                 for j in jumlah_list]

    # Membuat bar chart waktu eksekusi
    bars = axes3[0].bar([str(j) for j in jumlah_list], waktu_list, color=warna_bar)

    # Menambahkan label waktu di atas setiap bar
    for bar, waktu in zip(bars, waktu_list):
        axes3[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f'{waktu:.3f}s', ha='center', va='bottom', fontsize=10)

    axes3[0].set_xlabel("Jumlah Gambar", fontsize=11)
    axes3[0].set_ylabel("Waktu (detik)", fontsize=11)
    axes3[0].set_title("Waktu Stitching vs Jumlah Gambar", fontsize=12)
    axes3[0].grid(axis='y', alpha=0.3)

    # Subplot kanan: Hasil stitching 5 gambar (jika tersedia)
    result_5_path = os.path.join(OUTPUT_DIR, "02_wide_5_gambar.jpg")
    result_5 = cv2.imread(result_5_path)
    if result_5 is not None:
        axes3[1].imshow(cv2.cvtColor(result_5, cv2.COLOR_BGR2RGB))
        axes3[1].set_title("Panorama 5 Gambar Wide", fontsize=12)
    else:
        result_3_path = os.path.join(OUTPUT_DIR, "02_wide_3_gambar.jpg")
        result_3 = cv2.imread(result_3_path)
        if result_3 is not None:
            axes3[1].imshow(cv2.cvtColor(result_3, cv2.COLOR_BGR2RGB))
            axes3[1].set_title("Panorama 3 Gambar Wide", fontsize=12)
        else:
            axes3[1].text(0.5, 0.5, "Tidak tersedia", ha='center', va='center')
    axes3[1].axis("off")

    # Menambahkan judul utama
    plt.suptitle("Percobaan 2: Analisis Waktu Stitching",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Menyimpan grid waktu
    plt.savefig(os.path.join(OUTPUT_DIR, "02_grid_analisis_waktu.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Grid analisis waktu disimpan.")
    plt.close()

# ============================================================
# LANGKAH 10: Ringkasan dan Statistik
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 2: OPENCV STITCHER API")
print("=" * 65)

# Menampilkan tabel ringkasan semua percobaan stitching
print("\n  Tabel Ringkasan Stitching:")
print(f"  {'Percobaan':<30} | {'Status':<12} | {'Waktu':>8}")
print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*8}")
print(f"  {'Outdoor PANORAMA':<30} | {'OK' if status_pano == cv2.Stitcher_OK else 'GAGAL':<12} | {waktu_pano:>7.3f}s")
print(f"  {'Outdoor SCANS':<30} | {'OK' if status_scan == cv2.Stitcher_OK else 'GAGAL':<12} | {waktu_scan:>7.3f}s")
print(f"  {'Urutan Diacak':<30} | {'OK' if status_scrambled == cv2.Stitcher_OK else 'GAGAL':<12} |      - ")
if len(indoor_images) >= 2:
    print(f"  {'Indoor (4 gambar)':<30} | {'OK' if status_indoor == cv2.Stitcher_OK else 'GAGAL':<12} | {waktu_indoor:>7.3f}s")
print(f"  {'Non-overlap (error test)':<30} | {'OK' if status_fail == cv2.Stitcher_OK else 'GAGAL':<12} |      - ")

# Menampilkan tabel waktu berdasarkan jumlah gambar
if timing_results:
    print(f"\n  Waktu Stitching berdasarkan Jumlah Gambar:")
    print(f"  {'Jumlah':>8} | {'Waktu':>10} | {'Status':<12}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}")
    for j in sorted(timing_results.keys()):
        r = timing_results[j]
        status_str = "OK" if r['status'] == cv2.Stitcher_OK else "GAGAL"
        print(f"  {j:>8} | {r['waktu']:>9.3f}s | {status_str:<12}")

# Menampilkan daftar output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("02_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.Stitcher_create()      → Membuat objek stitcher")
print("    stitcher.stitch()          → Menjalankan pipeline stitching otomatis")
print("    cv2.Stitcher_PANORAMA      → Mode untuk kamera berputar")
print("    cv2.Stitcher_SCANS         → Mode untuk translasi kamera/dokumen")
print("    crop_border_hitam()        → Menghilangkan border hitam hasil warping")
print("    interpretasi_status()      → Memahami kode error Stitcher")
print("=" * 65)
