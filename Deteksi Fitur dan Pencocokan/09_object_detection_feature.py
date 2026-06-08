"""
==========================================================================
PERCOBAAN 9: DETEKSI OBJEK MENGGUNAKAN FEATURE MATCHING
==========================================================================
Program ini mempelajari cara mendeteksi dan melokalisasi objek tertentu
di dalam sebuah scene menggunakan feature matching dan verifikasi
geometri (homography). Jika jumlah inlier cukup, objek dianggap
terdeteksi dan bounding box digambar pada scene.

Konsep yang dipelajari:
- Pipeline deteksi objek berbasis fitur: detect -> match -> verify
- SIFT + FLANN + Ratio Test + RANSAC homography
- Kriteria deteksi berdasarkan jumlah inlier minimum
- Confidence scoring berdasarkan inlier count dan persentase
- Cross-testing: deteksi objek pada scene yang salah (negatif)
- Pembuatan tabel laporan deteksi

Fungsi utama yang dipelajari:
- cv2.findHomography()        : Estimasi homography untuk verifikasi geometri
- cv2.perspectiveTransform()  : Mentransformasi titik corner template
- cv2.polylines()             : Menggambar bounding box poligon
- cv2.drawMatches()           : Menggambar hasil matching

Hasil: Visualisasi deteksi objek pada setiap scene dan tabel laporan
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
print("PERCOBAAN 9: DETEKSI OBJEK MENGGUNAKAN FEATURE MATCHING")
print("=" * 60)

# ============================================================
# 1. Mendefinisikan Fungsi Deteksi Objek
# ============================================================

def deteksi_objek(img_template, img_scene, nama_objek="objek", min_inliers=10):
    """
    Mendeteksi objek template di dalam scene menggunakan
    SIFT + FLANN + ratio test + RANSAC homography.
    Mengembalikan dictionary hasil deteksi.
    """

    # Mengkonversi template ke grayscale untuk deteksi fitur
    gray_tmpl = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    # Mengkonversi scene ke grayscale untuk deteksi fitur
    gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

    # Membuat detektor SIFT
    sift = cv2.SIFT_create()

    # Mendeteksi keypoints dan descriptor pada template
    kp_tmpl, desc_tmpl = sift.detectAndCompute(gray_tmpl, None)

    # Mendeteksi keypoints dan descriptor pada scene
    kp_scene, desc_scene = sift.detectAndCompute(gray_scene, None)

    # Menyiapkan dictionary hasil
    hasil = {
        'nama': nama_objek,
        'kp_template': len(kp_tmpl),
        'kp_scene': len(kp_scene),
        'good_matches': 0,
        'inliers': 0,
        'rasio_inlier': 0.0,
        'terdeteksi': False,
        'confidence': 0.0,
        'corners': None,
        'img_result': None,
        'waktu': 0.0
    }

    # Mencatat waktu mulai
    t_start = time.time()

    # Memeriksa apakah deskriptor cukup untuk matching
    if desc_tmpl is None or desc_scene is None or len(kp_tmpl) < 4 or len(kp_scene) < 4:
        # Menghitung waktu proses
        hasil['waktu'] = time.time() - t_start
        return hasil

    # Mendefinisikan parameter FLANN untuk deskriptor float
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Membuat matcher FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Melakukan KNN matching k=2
    matches_knn = flann.knnMatch(desc_tmpl, desc_scene, k=2)

    # Menerapkan Lowe's Ratio Test
    good_matches = []
    for m, n in matches_knn:
        # Memeriksa rasio jarak
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Menyimpan jumlah good matches
    hasil['good_matches'] = len(good_matches)

    # Memeriksa apakah cukup match untuk homography (minimal 4)
    if len(good_matches) >= 4:
        # Mengekstrak lokasi keypoint template
        src_pts = np.float32([kp_tmpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Mengekstrak lokasi keypoint scene
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Menghitung homography dengan RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Memeriksa apakah homography berhasil dihitung
        if H is not None and mask is not None:
            # Menghitung jumlah inlier
            n_inlier = int(mask.sum())

            # Menghitung rasio inlier
            rasio = n_inlier / len(mask) * 100

            # Menyimpan hasil
            hasil['inliers'] = n_inlier
            hasil['rasio_inlier'] = rasio

            # Menghitung confidence score (kombinasi jumlah dan rasio inlier)
            conf_count = min(n_inlier / 30.0, 1.0)
            conf_ratio = rasio / 100.0
            hasil['confidence'] = (conf_count * 0.6 + conf_ratio * 0.4) * 100

            # Memeriksa apakah deteksi memenuhi threshold minimum inlier
            if n_inlier >= min_inliers:
                # Menandai objek terdeteksi
                hasil['terdeteksi'] = True

                # Mendapatkan ukuran template
                h_t, w_t = img_template.shape[:2]

                # Mendefinisikan corner template
                corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)

                # Mentransformasikan corner ke scene
                corners_scene = cv2.perspectiveTransform(corners, H)

                # Menyimpan corner hasil transformasi
                hasil['corners'] = corners_scene

    # Membuat gambar hasil visualisasi
    img_result = img_scene.copy()

    # Memeriksa apakah objek terdeteksi
    if hasil['terdeteksi'] and hasil['corners'] is not None:
        # Menggambar bounding box hijau pada scene
        cv2.polylines(img_result, [np.int32(hasil['corners'])], True, (0, 255, 0), 3)

        # Menambahkan label nama objek
        corner_top = tuple(np.int32(hasil['corners'][0][0]))
        cv2.putText(img_result, f"{nama_objek} ({hasil['confidence']:.0f}%)",
                    (corner_top[0], corner_top[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Menyimpan gambar hasil
    hasil['img_result'] = img_result

    # Menghitung waktu proses
    hasil['waktu'] = time.time() - t_start

    return hasil

# ============================================================
# 2. Memuat Semua Gambar Template dan Scene
# ============================================================

# Mendefinisikan daftar nama objek yang akan dideteksi
nama_objek_list = ["buku", "poster", "kartu"]

# Menyiapkan dictionary untuk menyimpan gambar template
templates = {}

# Menyiapkan dictionary untuk menyimpan gambar scene
scenes = {}

# Memuat semua gambar template dan scene
for nama in nama_objek_list:
    # Membaca gambar template objek
    templates[nama] = cv2.imread(os.path.join(IMAGE_DIR, f"objek_{nama}.jpg"))

    # Membaca gambar scene yang mengandung objek
    scenes[nama] = cv2.imread(os.path.join(IMAGE_DIR, f"scene_{nama}.jpg"))

    # Memeriksa apakah gambar berhasil dimuat
    if templates[nama] is None or scenes[nama] is None:
        print(f"[ERROR] Gambar objek_{nama}.jpg atau scene_{nama}.jpg tidak ditemukan!")
        exit()

    # Menampilkan informasi ukuran gambar
    print(f"[INFO] Template {nama}: {templates[nama].shape}, Scene {nama}: {scenes[nama].shape}")

# ============================================================
# 3. Deteksi Setiap Objek pada Scene yang Benar
# ============================================================

# Menampilkan header bagian deteksi
print(f"\n--- Deteksi Objek pada Scene yang Sesuai ---")

# Menyiapkan list untuk menyimpan semua hasil deteksi
semua_hasil = []

# Melakukan deteksi setiap objek pada scene yang benar
for nama in nama_objek_list:
    # Menampilkan proses deteksi
    print(f"\n[PROSES] Mendeteksi {nama} pada scene_{nama}...")

    # Menjalankan fungsi deteksi
    hasil = deteksi_objek(templates[nama], scenes[nama], nama)

    # Menambahkan informasi scene ke hasil
    hasil['scene'] = f"scene_{nama}"

    # Menyimpan hasil ke list
    semua_hasil.append(hasil)

    # Menampilkan hasil deteksi
    status = "TERDETEKSI" if hasil['terdeteksi'] else "TIDAK TERDETEKSI"
    print(f"  Status: {status}")
    print(f"  Good matches: {hasil['good_matches']}, Inliers: {hasil['inliers']}")
    print(f"  Confidence: {hasil['confidence']:.1f}%")
    print(f"  Waktu: {hasil['waktu'] * 1000:.1f} ms")

# ============================================================
# 4. Cross-Test: Deteksi pada Scene yang Salah
# ============================================================

# Menampilkan header cross-test
print(f"\n--- Cross-Test: Deteksi pada Scene yang Salah ---")

# Menyiapkan list untuk hasil cross-test
hasil_cross = []

# Melakukan cross-test (mendeteksi objek pada scene yang bukan miliknya)
for nama_tmpl in nama_objek_list:
    for nama_scene in nama_objek_list:
        # Melewati jika template dan scene sama (bukan cross-test)
        if nama_tmpl == nama_scene:
            continue

        # Menampilkan proses cross-test
        print(f"\n[CROSS] Mendeteksi {nama_tmpl} pada scene_{nama_scene}...")

        # Menjalankan deteksi
        hasil = deteksi_objek(templates[nama_tmpl], scenes[nama_scene],
                              f"{nama_tmpl} di scene_{nama_scene}")

        # Menambahkan info scene
        hasil['scene'] = f"scene_{nama_scene}"
        hasil['template'] = nama_tmpl

        # Menyimpan hasil cross-test
        hasil_cross.append(hasil)

        # Menampilkan hasil
        status = "TERDETEKSI (FALSE POSITIVE!)" if hasil['terdeteksi'] else "TIDAK TERDETEKSI (BENAR)"
        print(f"  Status: {status}")
        print(f"  Good matches: {hasil['good_matches']}, Inliers: {hasil['inliers']}")

# ============================================================
# 5. Visualisasi Hasil Deteksi per Objek
# ============================================================

# Melakukan iterasi untuk setiap objek dan menyimpan visualisasi
for i, nama in enumerate(nama_objek_list):
    # Membuat figure 1x3 (template, scene, hasil deteksi)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Menampilkan gambar template
    axes[0].imshow(cv2.cvtColor(templates[nama], cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Template: objek_{nama}", fontsize=12)
    axes[0].axis('off')

    # Menampilkan gambar scene asli
    axes[1].imshow(cv2.cvtColor(scenes[nama], cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Scene: scene_{nama}", fontsize=12)
    axes[1].axis('off')

    # Menampilkan gambar hasil deteksi dengan bounding box
    axes[2].imshow(cv2.cvtColor(semua_hasil[i]['img_result'], cv2.COLOR_BGR2RGB))

    # Menentukan status terdeteksi
    status = "TERDETEKSI" if semua_hasil[i]['terdeteksi'] else "TIDAK"
    conf = semua_hasil[i]['confidence']

    # Memberikan judul hasil
    axes[2].set_title(f"Hasil: {status} (Conf: {conf:.0f}%)", fontsize=12)
    axes[2].axis('off')

    # Memberikan judul utama
    fig.suptitle(f"Deteksi Objek: {nama.upper()}", fontsize=15, fontweight='bold')

    # Mengatur layout
    plt.tight_layout()

    # Menyimpan visualisasi ke file
    plt.savefig(os.path.join(OUTPUT_DIR, f"09_deteksi_{nama}.png"), dpi=150, bbox_inches='tight')
    plt.show()

    # Menampilkan pesan file tersimpan
    print(f"\n[SAVED] 09_deteksi_{nama}.png")

    # Menutup figure
    plt.close()

# ============================================================
# 6. Membuat Tabel Laporan Deteksi
# ============================================================

# Menampilkan header laporan
print(f"\n--- Laporan Lengkap Deteksi Objek ---")

# Membuat figure untuk tabel laporan
fig, ax = plt.subplots(figsize=(14, 8))

# Menonaktifkan sumbu
ax.axis('off')

# Menyusun data tabel untuk deteksi yang benar
tabel_data = []
tabel_header = ["Template", "Scene", "Good Match", "Inliers", "Rasio(%)", "Confidence(%)", "Status"]

# Menambahkan data deteksi positif (scene yang benar)
for h in semua_hasil:
    # Menentukan status
    status = "OK" if h['terdeteksi'] else "GAGAL"

    # Menambahkan baris data
    tabel_data.append([
        h['nama'],
        h['scene'],
        str(h['good_matches']),
        str(h['inliers']),
        f"{h['rasio_inlier']:.1f}",
        f"{h['confidence']:.1f}",
        status
    ])

# Menambahkan separator
tabel_data.append(["---" for _ in tabel_header])

# Menambahkan data cross-test (scene yang salah)
for h in hasil_cross:
    # Menentukan status cross-test
    status = "FALSE POS" if h['terdeteksi'] else "OK (Neg)"

    # Menambahkan baris data
    tabel_data.append([
        h.get('template', h['nama']),
        h['scene'],
        str(h['good_matches']),
        str(h['inliers']),
        f"{h['rasio_inlier']:.1f}",
        f"{h['confidence']:.1f}",
        status
    ])

# Membuat tabel pada figure
table = ax.table(cellText=tabel_data, colLabels=tabel_header,
                 cellLoc='center', loc='center')

# Mengatur ukuran font tabel
table.auto_set_font_size(False)
table.set_fontsize(9)

# Mengatur skala tabel
table.scale(1.0, 1.4)

# Memberi warna header tabel
for j in range(len(tabel_header)):
    # Memberikan warna biru muda pada header
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Memberi warna baris berdasarkan status
for i in range(len(tabel_data)):
    for j in range(len(tabel_header)):
        # Memeriksa apakah baris adalah separator
        if tabel_data[i][0] == "---":
            table[i + 1, j].set_facecolor('#D9D9D9')
        # Memeriksa apakah kolom status
        elif j == len(tabel_header) - 1:
            if tabel_data[i][j] == "OK":
                table[i + 1, j].set_facecolor('#C6EFCE')
            elif tabel_data[i][j] == "OK (Neg)":
                table[i + 1, j].set_facecolor('#C6EFCE')
            elif tabel_data[i][j] == "FALSE POS":
                table[i + 1, j].set_facecolor('#FFC7CE')
            elif tabel_data[i][j] == "GAGAL":
                table[i + 1, j].set_facecolor('#FFC7CE')

# Memberikan judul tabel
ax.set_title("Laporan Deteksi Objek: Feature Matching + Homography",
             fontsize=14, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Menyimpan tabel laporan ke file
plt.savefig(os.path.join(OUTPUT_DIR, "09_detection_report.png"), dpi=150, bbox_inches='tight')
plt.show()

# Menampilkan pesan file tersimpan
print(f"[SAVED] 09_detection_report.png")

# Menutup figure
plt.close()

# ============================================================
# 7. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 9: DETEKSI OBJEK MENGGUNAKAN FEATURE MATCHING")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan pipeline deteksi
print("1. Pipeline deteksi objek berbasis fitur:")
print("   Detect(SIFT) -> Match(FLANN) -> Filter(Ratio) -> Verify(RANSAC)")

# Menampilkan penjelasan tentang verifikasi geometri
print("2. Verifikasi geometri menggunakan homography memastikan")
print("   konsistensi spasial antar match, bukan hanya kemiripan fitur.")

# Menampilkan kriteria deteksi
print("3. Objek dianggap terdeteksi jika:")
print("   - Jumlah inlier >= threshold minimum (10)")
print("   - Homography berhasil dihitung")

# Menampilkan tentang confidence scoring
print("4. Confidence scoring menggabungkan jumlah inlier (60%)")
print("   dan rasio inlier (40%) untuk mengukur keyakinan deteksi.")

# Menampilkan tentang cross-test
print("5. Cross-test pada scene yang salah memvalidasi bahwa metode")
print("   tidak menghasilkan false positive.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 09_deteksi_buku.png")
print("  - 09_deteksi_poster.png")
print("  - 09_deteksi_kartu.png")
print("  - 09_detection_report.png")

# Menampilkan garis penutup
print("=" * 60)
