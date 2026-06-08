"""
==========================================================================
PERCOBAAN 15: IMAGE RETRIEVAL MENGGUNAKAN FEATURE MATCHING
==========================================================================
Program ini membangun sistem image retrieval sederhana menggunakan
feature matching. Sistem ini membuat "database" dari semua gambar di
folder IMAGE_DIR, mengekstrak fitur ORB dari setiap gambar, kemudian
untuk sebuah query image, mencocokkan fitur dengan seluruh database
dan meranking gambar berdasarkan skor kemiripan.

Konsep yang dipelajari:
- Content-Based Image Retrieval (CBIR) menggunakan fitur lokal
- Membangun database fitur dari kumpulan gambar
- Feature extraction menggunakan ORB untuk kecepatan
- Brute-Force matching untuk mencocokkan fitur antar gambar
- Scoring: menghitung skor kemiripan berdasarkan jumlah good matches
- Ranking gambar berdasarkan skor kemiripan tertinggi
- Retrieval evaluation: apakah gambar relevan ditemukan?

Fungsi utama yang dipelajari:
- cv2.ORB_create()        : Membuat detektor dan deskriptor ORB
- cv2.BFMatcher()         : Brute-Force matcher dengan Hamming distance
- detectAndCompute()      : Mendeteksi keypoint dan menghitung deskriptor
- matcher.knnMatch()      : K-Nearest Neighbor matching
- sorted()                : Mengurutkan hasil berdasarkan skor

Hasil: Visualisasi top-5 gambar paling mirip untuk setiap query
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

# Mengimpor glob untuk mencari file dengan pola tertentu
import glob

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
print("PERCOBAAN 15: IMAGE RETRIEVAL MENGGUNAKAN FEATURE MATCHING")
print("=" * 60)

# ============================================================
# 1. Membangun Database Gambar
# ============================================================

# Menampilkan header bagian
print("\n--- 1. Membangun Database Gambar ---")

# Mencari semua file gambar jpg di folder IMAGE_DIR
daftar_file = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

# Memeriksa apakah ada gambar yang ditemukan
if len(daftar_file) == 0:
    print("[ERROR] Tidak ada gambar ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan jumlah gambar yang ditemukan
print(f"[INFO] Ditemukan {len(daftar_file)} gambar di database")

# Menyiapkan list untuk menyimpan data database
database = []

# Membuat detektor ORB dengan jumlah fitur maksimum 1000
orb = cv2.ORB_create(nfeatures=1000)

# Membuat matcher Brute-Force dengan Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Menampilkan proses ekstraksi fitur
print("[INFO] Mengekstrak fitur ORB dari setiap gambar...")

# Melakukan iterasi untuk setiap file gambar
for filepath in daftar_file:
    # Membaca gambar dari file
    img = cv2.imread(filepath)

    # Melewatkan file yang gagal dibaca
    if img is None:
        continue

    # Mengkonversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mendeteksi keypoint dan menghitung deskriptor ORB
    kp, des = orb.detectAndCompute(gray, None)

    # Melewatkan gambar yang tidak memiliki deskriptor
    if des is None:
        continue

    # Mendapatkan nama file saja tanpa path
    nama_file = os.path.basename(filepath)

    # Menyimpan data gambar ke database
    database.append({
        'nama': nama_file,
        'path': filepath,
        'gambar': img,
        'keypoints': kp,
        'deskriptor': des,
        'jumlah_kp': len(kp)
    })

    # Menampilkan informasi gambar yang telah diproses
    print(f"  [OK] {nama_file}: {len(kp)} keypoints, deskriptor shape={des.shape}")

# Menampilkan total gambar dalam database
print(f"\n[INFO] Total gambar dalam database: {len(database)}")

# ============================================================
# 2. Fungsi Retrieval: Mencocokkan Query dengan Database
# ============================================================

# Menampilkan header bagian
print("\n--- 2. Fungsi Image Retrieval ---")


# Mendefinisikan fungsi untuk melakukan image retrieval
def image_retrieval(query_des, database, bf_matcher, rasio_threshold=0.75):
    """
    Mencocokkan deskriptor query dengan seluruh database.
    Mengembalikan list skor kemiripan yang sudah diurutkan.
    """
    # Menyiapkan list untuk menyimpan hasil skor
    hasil_skor = []

    # Melakukan iterasi untuk setiap entri di database
    for i, entry in enumerate(database):
        # Mendapatkan deskriptor dari entri database
        des_db = entry['deskriptor']

        # Memastikan deskriptor query dan database valid
        if query_des is None or des_db is None:
            continue

        # Melakukan KNN matching dengan k=2
        try:
            matches = bf_matcher.knnMatch(query_des, des_db, k=2)
        except cv2.error:
            continue

        # Menerapkan ratio test Lowe untuk menyaring good matches
        good_matches = []
        for m_pair in matches:
            # Memastikan ada 2 tetangga terdekat
            if len(m_pair) == 2:
                m, n = m_pair
                # Memeriksa apakah match terbaik jauh lebih baik dari kedua
                if m.distance < rasio_threshold * n.distance:
                    good_matches.append(m)

        # Menghitung skor kemiripan (jumlah good matches dinormalisasi)
        skor = len(good_matches)

        # Menghitung rasio good matches terhadap total matches
        rasio = skor / max(len(matches), 1)

        # Menyimpan hasil skor untuk entri ini
        hasil_skor.append({
            'index': i,
            'nama': entry['nama'],
            'skor': skor,
            'rasio': rasio,
            'good_matches': good_matches
        })

    # Mengurutkan hasil berdasarkan skor (tertinggi dulu)
    hasil_skor.sort(key=lambda x: x['skor'], reverse=True)

    # Mengembalikan hasil yang sudah diurutkan
    return hasil_skor


# Menampilkan informasi fungsi retrieval selesai didefinisikan
print("[INFO] Fungsi image_retrieval() siap digunakan")

# ============================================================
# 3. Query 1: Mencari Gambar Mirip dengan scene_left.jpg
# ============================================================

# Menampilkan header bagian
print("\n--- 3. Query 1: scene_left.jpg ---")

# Menentukan nama file query pertama
query1_nama = "scene_left.jpg"

# Mencari entri query dalam database
query1_entry = None
for entry in database:
    if entry['nama'] == query1_nama:
        query1_entry = entry
        break

# Memeriksa apakah query ditemukan
if query1_entry is None:
    print(f"[WARNING] {query1_nama} tidak ditemukan di database, menggunakan gambar pertama")
    query1_entry = database[0]

# Menampilkan informasi query
print(f"[INFO] Query: {query1_entry['nama']} ({query1_entry['jumlah_kp']} keypoints)")

# Mengukur waktu mulai retrieval
waktu_mulai_q1 = time.time()

# Melakukan image retrieval untuk query 1
hasil_q1 = image_retrieval(query1_entry['deskriptor'], database, bf)

# Mengukur waktu selesai retrieval
waktu_q1 = time.time() - waktu_mulai_q1

# Menampilkan waktu retrieval
print(f"[INFO] Waktu retrieval: {waktu_q1:.3f} detik")

# Menampilkan top-5 hasil
print(f"\nTop-5 gambar paling mirip dengan {query1_entry['nama']}:")
for i, h in enumerate(hasil_q1[:5]):
    print(f"  {i+1}. {h['nama']}: skor={h['skor']}, rasio={h['rasio']:.3f}")

# Membuat visualisasi hasil query 1
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Menampilkan gambar query di posisi pertama
axes[0, 0].imshow(cv2.cvtColor(query1_entry['gambar'], cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"QUERY: {query1_entry['nama']}", fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Menampilkan top-5 hasil retrieval
for i, h in enumerate(hasil_q1[:5]):
    # Menentukan posisi subplot (baris, kolom)
    row = (i + 1) // 3
    col = (i + 1) % 3

    # Mendapatkan gambar dari database
    img_hasil = database[h['index']]['gambar']

    # Menampilkan gambar hasil
    axes[row, col].imshow(cv2.cvtColor(img_hasil, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f"#{i+1}: {h['nama']}\nSkor={h['skor']}, Rasio={h['rasio']:.3f}", fontsize=10)
    axes[row, col].axis('off')

# Menambahkan judul utama
fig.suptitle(f"Image Retrieval - Query: {query1_entry['nama']} (waktu: {waktu_q1:.3f}s)",
             fontsize=14, fontweight='bold')

# Mengatur layout agar rapi
plt.tight_layout()

# Menyimpan hasil visualisasi query 1
output_path_q1 = os.path.join(OUTPUT_DIR, "15_retrieval_query1.png")
plt.savefig(output_path_q1, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_q1}")

# ============================================================
# 4. Query 2: Mencari Gambar Mirip dengan objek_buku.jpg
# ============================================================

# Menampilkan header bagian
print("\n--- 4. Query 2: objek_buku.jpg ---")

# Menentukan nama file query kedua
query2_nama = "objek_buku.jpg"

# Mencari entri query dalam database
query2_entry = None
for entry in database:
    if entry['nama'] == query2_nama:
        query2_entry = entry
        break

# Memeriksa apakah query ditemukan
if query2_entry is None:
    print(f"[WARNING] {query2_nama} tidak ditemukan, menggunakan gambar kedua")
    query2_entry = database[1] if len(database) > 1 else database[0]

# Menampilkan informasi query
print(f"[INFO] Query: {query2_entry['nama']} ({query2_entry['jumlah_kp']} keypoints)")

# Mengukur waktu mulai retrieval
waktu_mulai_q2 = time.time()

# Melakukan image retrieval untuk query 2
hasil_q2 = image_retrieval(query2_entry['deskriptor'], database, bf)

# Mengukur waktu selesai retrieval
waktu_q2 = time.time() - waktu_mulai_q2

# Menampilkan waktu retrieval
print(f"[INFO] Waktu retrieval: {waktu_q2:.3f} detik")

# Menampilkan top-5 hasil
print(f"\nTop-5 gambar paling mirip dengan {query2_entry['nama']}:")
for i, h in enumerate(hasil_q2[:5]):
    print(f"  {i+1}. {h['nama']}: skor={h['skor']}, rasio={h['rasio']:.3f}")

# Membuat visualisasi hasil query 2
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Menampilkan gambar query di posisi pertama
axes[0, 0].imshow(cv2.cvtColor(query2_entry['gambar'], cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"QUERY: {query2_entry['nama']}", fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Menampilkan top-5 hasil retrieval
for i, h in enumerate(hasil_q2[:5]):
    # Menentukan posisi subplot
    row = (i + 1) // 3
    col = (i + 1) % 3

    # Mendapatkan gambar dari database
    img_hasil = database[h['index']]['gambar']

    # Menampilkan gambar hasil
    axes[row, col].imshow(cv2.cvtColor(img_hasil, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f"#{i+1}: {h['nama']}\nSkor={h['skor']}, Rasio={h['rasio']:.3f}", fontsize=10)
    axes[row, col].axis('off')

# Menambahkan judul utama
fig.suptitle(f"Image Retrieval - Query: {query2_entry['nama']} (waktu: {waktu_q2:.3f}s)",
             fontsize=14, fontweight='bold')

# Mengatur layout agar rapi
plt.tight_layout()

# Menyimpan hasil visualisasi query 2
output_path_q2 = os.path.join(OUTPUT_DIR, "15_retrieval_query2.png")
plt.savefig(output_path_q2, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_q2}")

# ============================================================
# 5. Ranking Keseluruhan: Perbandingan Skor Semua Gambar
# ============================================================

# Menampilkan header bagian
print("\n--- 5. Ranking Keseluruhan ---")

# Menyiapkan data untuk visualisasi ranking
query_names = [query1_entry['nama'], query2_entry['nama']]
semua_hasil = [hasil_q1, hasil_q2]

# Membuat figure untuk ranking chart
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Melakukan iterasi untuk setiap query
for idx, (q_name, hasil) in enumerate(zip(query_names, semua_hasil)):
    # Mengambil top-10 hasil (atau semua jika kurang dari 10)
    top_n = hasil[:min(10, len(hasil))]

    # Mengekstrak nama-nama gambar untuk sumbu x
    nama_list = [h['nama'][:15] for h in top_n]

    # Mengekstrak skor untuk sumbu y
    skor_list = [h['skor'] for h in top_n]

    # Membuat warna berdasarkan ranking (hijau untuk terbaik, merah untuk terburuk)
    warna = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(skor_list)))

    # Membuat bar chart horizontal
    bars = axes[idx].barh(range(len(nama_list)), skor_list, color=warna)

    # Mengatur label sumbu y dengan nama gambar
    axes[idx].set_yticks(range(len(nama_list)))
    axes[idx].set_yticklabels(nama_list, fontsize=9)

    # Menambahkan label skor di ujung bar
    for i, (bar, skor) in enumerate(zip(bars, skor_list)):
        axes[idx].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                       f'{skor}', va='center', fontsize=9)

    # Mengatur label sumbu x
    axes[idx].set_xlabel('Skor Kemiripan (Good Matches)', fontsize=11)

    # Mengatur judul subplot
    axes[idx].set_title(f'Query: {q_name}', fontsize=12, fontweight='bold')

    # Membalik urutan sumbu y agar ranking 1 di atas
    axes[idx].invert_yaxis()

# Menambahkan judul utama
fig.suptitle('Image Retrieval Ranking - Perbandingan Dua Query', fontsize=14, fontweight='bold')

# Mengatur layout agar rapi
plt.tight_layout()

# Menyimpan hasil visualisasi ranking
output_path_ranking = os.path.join(OUTPUT_DIR, "15_retrieval_ranking.png")
plt.savefig(output_path_ranking, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_ranking}")

# ============================================================
# 6. Statistik Database dan Analisis
# ============================================================

# Menampilkan header bagian
print("\n--- 6. Statistik Database ---")

# Menghitung total keypoint di seluruh database
total_kp = sum(entry['jumlah_kp'] for entry in database)

# Menampilkan total keypoint
print(f"[INFO] Total keypoint di database: {total_kp}")

# Menghitung rata-rata keypoint per gambar
rata_kp = total_kp / len(database)

# Menampilkan rata-rata keypoint
print(f"[INFO] Rata-rata keypoint per gambar: {rata_kp:.1f}")

# Mencari gambar dengan keypoint terbanyak
max_entry = max(database, key=lambda x: x['jumlah_kp'])

# Menampilkan gambar dengan keypoint terbanyak
print(f"[INFO] Gambar dengan keypoint terbanyak: {max_entry['nama']} ({max_entry['jumlah_kp']})")

# Mencari gambar dengan keypoint tersedikit
min_entry = min(database, key=lambda x: x['jumlah_kp'])

# Menampilkan gambar dengan keypoint tersedikit
print(f"[INFO] Gambar dengan keypoint tersedikit: {min_entry['nama']} ({min_entry['jumlah_kp']})")

# Menampilkan rata-rata waktu retrieval
rata_waktu = (waktu_q1 + waktu_q2) / 2

# Menampilkan rata-rata waktu
print(f"[INFO] Rata-rata waktu retrieval: {rata_waktu:.3f} detik")

# ============================================================
# 7. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 15: IMAGE RETRIEVAL MENGGUNAKAN FEATURE MATCHING")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan konsep image retrieval
print("1. Image retrieval berbasis fitur lokal bekerja dengan")
print("   mencocokkan deskriptor antara query dan database.")

# Menampilkan tentang skor kemiripan
print("2. Skor kemiripan dihitung berdasarkan jumlah good matches")
print("   setelah ratio test Lowe (threshold 0.75).")

# Menampilkan tentang ORB + BFMatcher
print("3. ORB + BFMatcher dipilih untuk kecepatan tinggi,")
print("   cocok untuk real-time retrieval dengan database besar.")

# Menampilkan tentang ranking
print("4. Gambar diranking dari skor tertinggi ke terendah,")
print("   gambar dengan konten serupa mendapat skor tinggi.")

# Menampilkan tentang keterbatasan
print("5. Keterbatasan: ORB kurang robust terhadap perubahan")
print("   iluminasi dan skala besar dibanding SIFT.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 15_retrieval_query1.png")
print("  - 15_retrieval_query2.png")
print("  - 15_retrieval_ranking.png")

# Menampilkan garis penutup
print("=" * 60)
