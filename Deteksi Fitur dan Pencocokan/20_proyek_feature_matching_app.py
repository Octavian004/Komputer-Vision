"""
==========================================================================
PERCOBAAN 20: PROYEK AKHIR - APLIKASI FEATURE MATCHING
Sistem Deteksi dan Lokalisasi Multi-Objek
==========================================================================
Program ini membangun aplikasi lengkap berbasis feature matching yang
menggabungkan semua konsep dari Modul 07. Aplikasi ini mendeteksi dan
melokalisasi beberapa objek (buku, poster, kartu) dalam sebuah scene
menggunakan feature matching dan homography estimation.

Konsep yang dipelajari:
- Integrasi lengkap: database objek → deteksi → lokalisasi → evaluasi
- Multi-object detection: mendeteksi beberapa objek sekaligus
- Bounding box labeled: menggambar kotak pembatas berlabel
- Confidence score: menilai tingkat keyakinan deteksi
- False positive filtering: menggunakan minimum inlier threshold
- Benchmarking: mengukur waktu per objek dan total pipeline
- Rekomendasi detektor berdasarkan benchmark

Fungsi utama yang dipelajari:
- cv2.SIFT_create()          : Detektor/deskriptor SIFT
- cv2.ORB_create()            : Detektor/deskriptor ORB
- cv2.AKAZE_create()          : Detektor/deskriptor AKAZE
- cv2.BFMatcher()             : Brute-Force matcher
- cv2.FlannBasedMatcher()     : FLANN matcher
- cv2.findHomography()        : Estimasi homography
- cv2.perspectiveTransform()  : Transformasi titik
- cv2.polylines()             : Menggambar bounding box
- cv2.putText()               : Menambahkan label teks

Hasil: Deteksi multi-objek, laporan evaluasi, dan benchmark lengkap
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
print("PERCOBAAN 20: PROYEK AKHIR - APLIKASI FEATURE MATCHING")
print("Sistem Deteksi dan Lokalisasi Multi-Objek")
print("=" * 60)

# ============================================================
# 1. Membangun Database Objek
# ============================================================

# Menampilkan header bagian
print("\n--- 1. Membangun Database Objek ---")

# Mendefinisikan daftar objek yang akan dideteksi
daftar_objek = [
    {'nama': 'Buku', 'file': 'objek_buku.jpg', 'warna': (0, 255, 0)},
    {'nama': 'Poster', 'file': 'objek_poster.jpg', 'warna': (255, 0, 0)},
    {'nama': 'Kartu', 'file': 'objek_kartu.jpg', 'warna': (0, 0, 255)},
]

# Membuat detektor SIFT sebagai detektor default
sift = cv2.SIFT_create()

# Menyiapkan database objek
database_objek = []

# Melakukan iterasi untuk memuat dan memproses setiap objek
for objek_info in daftar_objek:
    # Membaca gambar objek
    img_objek = cv2.imread(os.path.join(IMAGE_DIR, objek_info['file']))

    # Memeriksa apakah gambar berhasil dimuat
    if img_objek is None:
        print(f"  [SKIP] {objek_info['nama']}: file tidak ditemukan")
        continue

    # Mengkonversi ke grayscale
    gray_objek = cv2.cvtColor(img_objek, cv2.COLOR_BGR2GRAY)

    # Mendeteksi keypoint dan deskriptor SIFT
    kp_objek, des_objek = sift.detectAndCompute(gray_objek, None)

    # Menyimpan ke database
    database_objek.append({
        'nama': objek_info['nama'],
        'file': objek_info['file'],
        'gambar': img_objek,
        'gray': gray_objek,
        'keypoints': kp_objek,
        'deskriptor': des_objek,
        'warna': objek_info['warna'],
        'jumlah_kp': len(kp_objek)
    })

    # Menampilkan informasi objek
    print(f"  [OK] {objek_info['nama']}: {img_objek.shape}, {len(kp_objek)} keypoints")

# Menampilkan total objek di database
print(f"\n[INFO] Total objek di database: {len(database_objek)}")

# ============================================================
# 2. Memuat Scene untuk Deteksi
# ============================================================

# Menampilkan header bagian
print("\n--- 2. Memuat Scene ---")

# Mendefinisikan daftar scene yang akan diproses
daftar_scene = ['scene_buku.jpg', 'scene_poster.jpg', 'scene_kartu.jpg']

# Menyiapkan list untuk menyimpan data scene
scene_data = []

# Melakukan iterasi untuk memuat setiap scene
for scene_file in daftar_scene:
    # Membaca gambar scene
    img_scene = cv2.imread(os.path.join(IMAGE_DIR, scene_file))

    # Memeriksa apakah gambar berhasil dimuat
    if img_scene is None:
        print(f"  [SKIP] {scene_file}: tidak ditemukan")
        continue

    # Mengkonversi ke grayscale
    gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

    # Mendeteksi keypoint dan deskriptor pada scene
    kp_scene, des_scene = sift.detectAndCompute(gray_scene, None)

    # Menyimpan data scene
    scene_data.append({
        'file': scene_file,
        'gambar': img_scene,
        'gray': gray_scene,
        'keypoints': kp_scene,
        'deskriptor': des_scene,
        'jumlah_kp': len(kp_scene)
    })

    # Menampilkan informasi scene
    print(f"  [OK] {scene_file}: {img_scene.shape}, {len(kp_scene)} keypoints")

# Menampilkan total scene
print(f"\n[INFO] Total scene dimuat: {len(scene_data)}")

# ============================================================
# 3. Fungsi Deteksi Multi-Objek
# ============================================================

# Menampilkan header bagian
print("\n--- 3. Mendefinisikan Fungsi Deteksi ---")

# Mendefinisikan threshold minimum inlier untuk deteksi valid
MIN_INLIER_THRESHOLD = 8

# Mendefinisikan parameter FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Membuat FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)


# Mendefinisikan fungsi deteksi objek tunggal
def deteksi_objek(objek_db, scene_kp, scene_des, scene_shape):
    """
    Mendeteksi satu objek dalam scene menggunakan feature matching.
    Return: dictionary berisi hasil deteksi.
    """
    # Menyiapkan hasil default
    hasil = {
        'nama': objek_db['nama'],
        'terdeteksi': False,
        'matches': 0,
        'inliers': 0,
        'confidence': 0.0,
        'corners': None,
        'waktu': 0.0,
    }

    # Mengukur waktu mulai
    t_start = time.time()

    # Memeriksa apakah deskriptor objek valid
    if objek_db['deskriptor'] is None or scene_des is None:
        hasil['waktu'] = time.time() - t_start
        return hasil

    # Melakukan KNN matching
    try:
        matches = flann.knnMatch(objek_db['deskriptor'], scene_des, k=2)
    except cv2.error:
        hasil['waktu'] = time.time() - t_start
        return hasil

    # Menerapkan ratio test Lowe
    good_matches = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Menyimpan jumlah good matches
    hasil['matches'] = len(good_matches)

    # Memeriksa apakah cukup matches untuk homography
    if len(good_matches) < 4:
        hasil['waktu'] = time.time() - t_start
        return hasil

    # Mengekstrak titik korespondensi
    src_pts = np.float32([objek_db['keypoints'][m.queryIdx].pt
                          for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([scene_kp[m.trainIdx].pt
                          for m in good_matches]).reshape(-1, 1, 2)

    # Mengestimasi homography dengan RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Memeriksa apakah homography berhasil
    if H is None or mask is None:
        hasil['waktu'] = time.time() - t_start
        return hasil

    # Menghitung jumlah inlier
    inliers = int(mask.sum())
    hasil['inliers'] = inliers

    # Menghitung confidence
    hasil['confidence'] = inliers / max(len(good_matches), 1)

    # Memeriksa apakah memenuhi threshold minimum inlier
    if inliers >= MIN_INLIER_THRESHOLD:
        # Mendapatkan sudut-sudut objek
        h_obj, w_obj = objek_db['gambar'].shape[:2]
        corners = np.float32([[0, 0], [w_obj, 0], [w_obj, h_obj], [0, h_obj]]).reshape(-1, 1, 2)

        # Mentransformasi sudut ke koordinat scene
        corners_scene = cv2.perspectiveTransform(corners, H)

        # Memeriksa apakah bounding box masuk akal (tidak terlalu kecil/besar)
        area = cv2.contourArea(corners_scene.reshape(-1, 2).astype(np.float32))
        scene_area = scene_shape[0] * scene_shape[1]

        # Memvalidasi area bounding box
        if 100 < area < scene_area * 0.9:
            hasil['terdeteksi'] = True
            hasil['corners'] = corners_scene
        else:
            hasil['terdeteksi'] = False
    else:
        hasil['terdeteksi'] = False

    # Menghitung waktu total
    hasil['waktu'] = time.time() - t_start

    # Mengembalikan hasil deteksi
    return hasil


# Mendefinisikan fungsi deteksi multi-objek
def deteksi_multi_objek(database, scene_kp, scene_des, scene_shape):
    """Mendeteksi semua objek dalam database di scene."""
    # Menyiapkan list hasil
    semua_hasil = []

    # Melakukan deteksi untuk setiap objek
    for objek in database:
        hasil = deteksi_objek(objek, scene_kp, scene_des, scene_shape)
        hasil['warna'] = objek['warna']
        semua_hasil.append(hasil)

    # Mengembalikan semua hasil
    return semua_hasil


# Menampilkan informasi fungsi
print(f"[INFO] Fungsi deteksi siap (threshold inlier: {MIN_INLIER_THRESHOLD})")

# ============================================================
# 4. Menjalankan Deteksi Multi-Objek pada Semua Scene
# ============================================================

# Menampilkan header bagian
print("\n--- 4. Menjalankan Deteksi Multi-Objek ---")

# Menyiapkan list untuk menyimpan semua hasil deteksi
semua_hasil_deteksi = []

# Melakukan iterasi untuk setiap scene
for scene in scene_data:
    # Menampilkan header scene
    print(f"\n  === Scene: {scene['file']} ===")

    # Mengukur waktu total pipeline
    t_pipeline_start = time.time()

    # Menjalankan deteksi multi-objek
    hasil_deteksi = deteksi_multi_objek(
        database_objek, scene['keypoints'], scene['deskriptor'], scene['gambar'].shape
    )

    # Menghitung waktu total pipeline
    t_pipeline = time.time() - t_pipeline_start

    # Menampilkan hasil per objek
    for hasil in hasil_deteksi:
        status = "TERDETEKSI" if hasil['terdeteksi'] else "TIDAK DITEMUKAN"
        print(f"    {hasil['nama']}: [{status}] matches={hasil['matches']}, "
              f"inliers={hasil['inliers']}, confidence={hasil['confidence']:.3f}, "
              f"waktu={hasil['waktu']:.3f}s")

    # Menyimpan hasil keseluruhan untuk scene ini
    semua_hasil_deteksi.append({
        'scene': scene,
        'deteksi': hasil_deteksi,
        'waktu_pipeline': t_pipeline
    })

    # Menampilkan waktu total pipeline
    print(f"    [TOTAL] Waktu pipeline: {t_pipeline:.3f}s")

# ============================================================
# 5. Visualisasi Deteksi Multi-Objek
# ============================================================

# Menampilkan header bagian
print("\n--- 5. Visualisasi Deteksi ---")

# Membuat figure untuk deteksi multi-objek
jumlah_scene = len(semua_hasil_deteksi)
fig, axes = plt.subplots(1, max(jumlah_scene, 1), figsize=(8 * max(jumlah_scene, 1), 8))

# Menghandle kasus scene tunggal
if jumlah_scene == 1:
    axes = [axes]
elif jumlah_scene == 0:
    axes = []

# Melakukan iterasi untuk setiap scene
for idx, data in enumerate(semua_hasil_deteksi):
    # Membuat salinan scene untuk visualisasi
    scene_vis = data['scene']['gambar'].copy()

    # Menghitung jumlah objek terdeteksi
    jumlah_terdeteksi = 0

    # Menggambar bounding box untuk setiap objek yang terdeteksi
    for hasil in data['deteksi']:
        if hasil['terdeteksi'] and hasil['corners'] is not None:
            # Mendapatkan sudut-sudut bounding box
            corners = np.int32(hasil['corners'])

            # Menggambar polyline bounding box dengan warna objek
            cv2.polylines(scene_vis, [corners], True, hasil['warna'], 3)

            # Menghitung posisi label (di atas bounding box)
            label_x = int(np.mean(corners[:, 0, 0]))
            label_y = int(np.min(corners[:, 0, 1])) - 10

            # Memastikan posisi label tidak keluar dari gambar
            label_y = max(label_y, 25)

            # Menambahkan background untuk label
            label_text = f"{hasil['nama']} ({hasil['confidence']:.2f})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(scene_vis,
                          (label_x - tw // 2 - 5, label_y - th - 5),
                          (label_x + tw // 2 + 5, label_y + 5),
                          hasil['warna'], -1)

            # Menambahkan teks label
            cv2.putText(scene_vis, label_text,
                        (label_x - tw // 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Menambahkan jumlah terdeteksi
            jumlah_terdeteksi += 1

    # Menampilkan scene dengan bounding box
    axes[idx].imshow(cv2.cvtColor(scene_vis, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f'{data["scene"]["file"]}\n'
                        f'Terdeteksi: {jumlah_terdeteksi}/{len(database_objek)} objek\n'
                        f'Waktu: {data["waktu_pipeline"]:.3f}s',
                        fontsize=11, fontweight='bold')
    axes[idx].axis('off')

# Menambahkan judul utama
fig.suptitle('Percobaan 20: Deteksi dan Lokalisasi Multi-Objek',
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan visualisasi deteksi multi-objek
output_deteksi = os.path.join(OUTPUT_DIR, "20_deteksi_multi_objek.png")
plt.savefig(output_deteksi, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_deteksi}")

# ============================================================
# 6. Laporan Tabel Evaluasi
# ============================================================

# Menampilkan header bagian
print("\n--- 6. Laporan Tabel Evaluasi ---")

# Menyiapkan data untuk tabel laporan
header_report = ['Scene', 'Objek', 'Status', 'Matches', 'Inliers',
                 'Confidence', 'Waktu(s)']

# Menyiapkan baris tabel
baris_report = []

# Mengumpulkan semua data hasil deteksi
for data in semua_hasil_deteksi:
    scene_nama = data['scene']['file'].replace('.jpg', '')
    for hasil in data['deteksi']:
        status = 'OK' if hasil['terdeteksi'] else 'MISS'
        baris_report.append([
            scene_nama,
            hasil['nama'],
            status,
            str(hasil['matches']),
            str(hasil['inliers']),
            f"{hasil['confidence']:.3f}",
            f"{hasil['waktu']:.3f}",
        ])

# Membuat figure untuk tabel laporan
fig, ax = plt.subplots(figsize=(16, max(6, len(baris_report) * 0.5 + 2)))

# Menyembunyikan axes
ax.axis('off')

# Membuat tabel
tabel = ax.table(cellText=baris_report, colLabels=header_report,
                 loc='center', cellLoc='center')

# Mengatur ukuran font
tabel.auto_set_font_size(False)
tabel.set_fontsize(10)

# Mengatur tinggi baris
tabel.scale(1, 1.8)

# Mewarnai header
for j in range(len(header_report)):
    tabel[0, j].set_facecolor('#2E4057')
    tabel[0, j].set_text_props(color='white', fontweight='bold')

# Mewarnai baris berdasarkan status deteksi
for i, baris in enumerate(baris_report):
    if baris[2] == 'OK':
        warna_baris = '#C6EFCE'
    else:
        warna_baris = '#FFC7CE'
    for j in range(len(header_report)):
        tabel[i + 1, j].set_facecolor(warna_baris)

# Menghitung statistik deteksi
total_deteksi = sum(1 for b in baris_report if b[2] == 'OK')
total_uji = len(baris_report)
akurasi_total = total_deteksi / max(total_uji, 1) * 100

# Menambahkan judul
ax.set_title(f'Laporan Deteksi Multi-Objek\n'
             f'Akurasi Total: {total_deteksi}/{total_uji} ({akurasi_total:.1f}%)',
             fontsize=14, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Menyimpan tabel laporan
output_report = os.path.join(OUTPUT_DIR, "20_report_tabel.png")
plt.savefig(output_report, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_report}")

# ============================================================
# 7. Benchmark Waktu dan Chart
# ============================================================

# Menampilkan header bagian
print("\n--- 7. Benchmark Waktu ---")

# Mendefinisikan detektor yang akan di-benchmark
detektor_benchmark = {
    'SIFT': cv2.SIFT_create(),
    'ORB': cv2.ORB_create(nfeatures=1000),
    'AKAZE': cv2.AKAZE_create(),
}

# Menyiapkan dictionary untuk menyimpan waktu benchmark
waktu_benchmark = {}

# Mengambil scene pertama untuk benchmark (jika ada)
if len(scene_data) > 0:
    scene_bench = scene_data[0]
else:
    print("[ERROR] Tidak ada scene untuk benchmark!")
    scene_bench = None

# Melakukan benchmark untuk setiap detektor
if scene_bench is not None:
    for det_nama, detector in detektor_benchmark.items():
        # Menampilkan nama detektor
        print(f"\n  === Benchmark: {det_nama} ===")

        # Menyiapkan list waktu per objek
        waktu_per_objek = []

        # Mendeteksi fitur pada scene
        t_scene_start = time.time()
        kp_s, des_s = detector.detectAndCompute(scene_bench['gray'], None)
        t_scene_det = time.time() - t_scene_start

        # Membuat matcher yang sesuai
        if det_nama == 'SIFT':
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5), dict(checks=50)
            )
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Melakukan deteksi untuk setiap objek
        for objek in database_objek:
            # Mendeteksi fitur pada objek
            t_obj_start = time.time()
            kp_obj, des_obj = detector.detectAndCompute(objek['gray'], None)

            # Memeriksa apakah deskriptor valid
            if des_obj is None or des_s is None:
                waktu_per_objek.append({'nama': objek['nama'], 'waktu': 0.0})
                continue

            # Melakukan matching
            try:
                matches = matcher.knnMatch(des_obj, des_s, k=2)
            except cv2.error:
                waktu_per_objek.append({'nama': objek['nama'], 'waktu': 0.0})
                continue

            # Menerapkan ratio test
            good = [m for m_pair in matches if len(m_pair) == 2
                    for m in [m_pair[0]] if m.distance < 0.75 * m_pair[1].distance]

            # Menghitung waktu per objek
            t_obj = time.time() - t_obj_start

            # Menyimpan waktu
            waktu_per_objek.append({
                'nama': objek['nama'],
                'waktu': t_obj,
                'matches': len(good)
            })

            # Menampilkan informasi benchmark
            print(f"    {objek['nama']}: waktu={t_obj:.3f}s, matches={len(good)}")

        # Menyimpan waktu benchmark
        waktu_benchmark[det_nama] = {
            'waktu_scene_det': t_scene_det,
            'per_objek': waktu_per_objek,
            'total': t_scene_det + sum(w['waktu'] for w in waktu_per_objek)
        }

        # Menampilkan total waktu
        print(f"    [TOTAL] {waktu_benchmark[det_nama]['total']:.3f}s")

# Membuat figure untuk timing chart
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Bar chart waktu per detektor ---
if waktu_benchmark:
    # Mendapatkan nama detektor dan waktu total
    det_names = list(waktu_benchmark.keys())
    waktu_total = [waktu_benchmark[d]['total'] for d in det_names]

    # Mendefinisikan warna
    warna_det = ['#4472C4', '#ED7D31', '#70AD47']

    # Menggambar bar chart
    bars = axes[0].bar(det_names, waktu_total, color=warna_det[:len(det_names)])

    # Menambahkan label waktu di atas bar
    for bar, wt in zip(bars, waktu_total):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{wt:.3f}s', ha='center', fontsize=11, fontweight='bold')

    # Mengatur label sumbu
    axes[0].set_ylabel('Waktu Total (detik)', fontsize=12)
    axes[0].set_title('Total Waktu Pipeline per Detektor', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # --- Stacked bar chart waktu per objek ---
    # Mendefinisikan posisi x
    x_pos = np.arange(len(det_names))
    lebar = 0.6

    # Menggambar stacked bar
    bottom = np.zeros(len(det_names))

    # Menambahkan waktu deteksi scene
    waktu_scene_list = [waktu_benchmark[d]['waktu_scene_det'] for d in det_names]
    axes[1].bar(x_pos, waktu_scene_list, lebar, label='Deteksi Scene',
                bottom=bottom, color='#4472C4')
    bottom += np.array(waktu_scene_list)

    # Menambahkan waktu per objek
    warna_objek = ['#ED7D31', '#70AD47', '#FFC000']
    for obj_idx in range(len(database_objek)):
        waktu_obj_list = []
        for d in det_names:
            if obj_idx < len(waktu_benchmark[d]['per_objek']):
                waktu_obj_list.append(waktu_benchmark[d]['per_objek'][obj_idx]['waktu'])
            else:
                waktu_obj_list.append(0)
        nama_obj = database_objek[obj_idx]['nama'] if obj_idx < len(database_objek) else f'Objek {obj_idx}'
        axes[1].bar(x_pos, waktu_obj_list, lebar, label=f'Match: {nama_obj}',
                    bottom=bottom, color=warna_objek[obj_idx % len(warna_objek)])
        bottom += np.array(waktu_obj_list)

    # Mengatur label sumbu
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(det_names, fontsize=11)
    axes[1].set_ylabel('Waktu (detik)', fontsize=12)
    axes[1].set_title('Breakdown Waktu per Komponen', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

# Menambahkan judul utama
fig.suptitle('Percobaan 20: Benchmark Waktu Pipeline Deteksi', fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan timing chart
output_timing = os.path.join(OUTPUT_DIR, "20_timing_chart.png")
plt.savefig(output_timing, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"\n[SAVED] {output_timing}")

# ============================================================
# 8. Laporan Final Komprehensif
# ============================================================

# Menampilkan header bagian
print("\n--- 8. Laporan Final ---")

# Membuat figure untuk laporan final
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# --- Panel 1: Ringkasan Deteksi ---
ax1 = axes[0, 0]
ax1.axis('off')

# Menyiapkan teks ringkasan deteksi
teks_deteksi = "RINGKASAN DETEKSI MULTI-OBJEK\n"
teks_deteksi += "=" * 40 + "\n\n"

# Menghitung statistik per objek
for objek in database_objek:
    # Menghitung berapa kali objek terdeteksi
    deteksi_count = 0
    total_count = 0
    for data in semua_hasil_deteksi:
        for h in data['deteksi']:
            if h['nama'] == objek['nama']:
                total_count += 1
                if h['terdeteksi']:
                    deteksi_count += 1

    # Menambahkan ke teks
    teks_deteksi += f"{objek['nama']}:\n"
    teks_deteksi += f"  Terdeteksi: {deteksi_count}/{total_count}\n"
    teks_deteksi += f"  Keypoints: {objek['jumlah_kp']}\n\n"

# Menambahkan akurasi total
teks_deteksi += f"\nAkurasi Total: {akurasi_total:.1f}%"
teks_deteksi += f"\nMin Inlier Threshold: {MIN_INLIER_THRESHOLD}"

# Menampilkan teks ringkasan
ax1.text(0.1, 0.95, teks_deteksi, transform=ax1.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax1.set_title('Hasil Deteksi', fontsize=12, fontweight='bold')

# --- Panel 2: Benchmark Comparison ---
ax2 = axes[0, 1]

# Menampilkan perbandingan detektor dalam radar-like chart sebagai bar chart
if waktu_benchmark:
    # Mengumpulkan metrik per detektor
    metrik_names = ['Waktu (s)', 'Kecepatan\n(1/waktu)']
    metrik_data = {}
    for det in det_names:
        total_t = waktu_benchmark[det]['total']
        metrik_data[det] = [total_t, 1.0 / max(total_t, 0.001)]

    # Membuat grouped bar chart
    x = np.arange(len(metrik_names))
    w = 0.25
    for i, det in enumerate(det_names):
        ax2.bar(x + i * w, metrik_data[det], w, label=det,
                color=warna_det[i % len(warna_det)])

    ax2.set_xticks(x + w)
    ax2.set_xticklabels(metrik_names, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

ax2.set_title('Perbandingan Detektor', fontsize=12, fontweight='bold')

# --- Panel 3: Confidence per Objek per Scene ---
ax3 = axes[1, 0]

# Menyiapkan data confidence
if semua_hasil_deteksi:
    scene_labels = [d['scene']['file'].replace('.jpg', '') for d in semua_hasil_deteksi]
    objek_names = [o['nama'] for o in database_objek]

    # Membuat grouped bar chart confidence
    x = np.arange(len(scene_labels))
    w = 0.25
    for i, obj_nama in enumerate(objek_names):
        conf_vals = []
        for data in semua_hasil_deteksi:
            for h in data['deteksi']:
                if h['nama'] == obj_nama:
                    conf_vals.append(h['confidence'])
                    break
            else:
                conf_vals.append(0)
        # Menentukan warna berdasarkan objek
        warna_obj_chart = ['#2ecc71', '#3498db', '#e74c3c']
        ax3.bar(x + i * w, conf_vals, w, label=obj_nama,
                color=warna_obj_chart[i % len(warna_obj_chart)])

    ax3.set_xticks(x + w)
    ax3.set_xticklabels(scene_labels, fontsize=10)
    ax3.set_ylabel('Confidence', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis='y', alpha=0.3)

ax3.set_title('Confidence per Objek per Scene', fontsize=12, fontweight='bold')

# --- Panel 4: Rekomendasi dan Summary ---
ax4 = axes[1, 1]
ax4.axis('off')

# Menyiapkan teks rekomendasi
teks_rekomendasi = "REKOMENDASI & KESIMPULAN\n"
teks_rekomendasi += "=" * 40 + "\n\n"

# Menentukan detektor terbaik berdasarkan benchmark
if waktu_benchmark:
    teks_rekomendasi += "Benchmark Detektor:\n"
    for det in sorted(waktu_benchmark, key=lambda d: waktu_benchmark[d]['total']):
        teks_rekomendasi += f"  {det}: {waktu_benchmark[det]['total']:.3f}s\n"

    # Detektor tercepat
    fastest = min(waktu_benchmark, key=lambda d: waktu_benchmark[d]['total'])
    teks_rekomendasi += f"\nDetektor Tercepat: {fastest}\n"

# Menambahkan rekomendasi
teks_rekomendasi += "\nRekomendasi Penggunaan:\n"
teks_rekomendasi += "  1. Real-time: ORB (cepat)\n"
teks_rekomendasi += "  2. Akurasi: SIFT (presisi)\n"
teks_rekomendasi += "  3. Balanced: AKAZE\n"
teks_rekomendasi += f"\nTotal Scene Diproses: {len(scene_data)}\n"
teks_rekomendasi += f"Total Objek Database: {len(database_objek)}\n"
teks_rekomendasi += f"Akurasi Deteksi: {akurasi_total:.1f}%\n"

# Menampilkan teks rekomendasi
ax4.text(0.1, 0.95, teks_rekomendasi, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax4.set_title('Rekomendasi', fontsize=12, fontweight='bold')

# Menambahkan judul utama
fig.suptitle('Percobaan 20: PROYEK AKHIR - Laporan Komprehensif\n'
             'Sistem Deteksi dan Lokalisasi Multi-Objek',
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan laporan final
output_final = os.path.join(OUTPUT_DIR, "20_proyek_final.png")
plt.savefig(output_final, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_final}")

# ============================================================
# 9. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 20: PROYEK AKHIR - APLIKASI FEATURE MATCHING")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan tentang sistem
print("1. Sistem deteksi dan lokalisasi multi-objek berhasil dibangun")
print("   menggunakan teknik feature matching dan homography.")

# Menampilkan tentang database
print("2. Database berisi 3 objek (Buku, Poster, Kartu) yang")
print("   masing-masing memiliki fitur SIFT yang diekstrak.")

# Menampilkan tentang deteksi
print("3. Deteksi dilakukan dengan matching fitur, estimasi homography,")
print("   dan filtering berdasarkan minimum inlier threshold.")

# Menampilkan tentang benchmark
print("4. Benchmark menunjukkan trade-off antara kecepatan dan akurasi:")
print("   ORB tercepat, SIFT paling akurat, AKAZE seimbang.")

# Menampilkan tentang false positive
print("5. False positive filtering menggunakan minimum inlier threshold")
print(f"   ({MIN_INLIER_THRESHOLD}) dan validasi area bounding box.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 20_deteksi_multi_objek.png")
print("  - 20_report_tabel.png")
print("  - 20_timing_chart.png")
print("  - 20_proyek_final.png")

# Menampilkan garis penutup
print("=" * 60)

# Menampilkan pesan selesai
print("\n" + "*" * 60)
print("SELAMAT! Semua 20 percobaan Modul 07 telah selesai!")
print("Feature Detection dan Matching telah dipelajari secara lengkap.")
print("*" * 60)
