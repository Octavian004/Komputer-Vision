"""
==========================================================================
PERCOBAAN 16: DETEKSI AR MARKER DAN OVERLAY
==========================================================================
Program ini mendeteksi AR marker (ar_marker.jpg) dalam sebuah scene
menggunakan feature matching, mengestimasi homography ke marker,
lalu melakukan overlay konten virtual (persegi panjang berwarna)
di atas marker. Teknik warp perspektif dan blending digunakan
untuk menempatkan overlay secara realistis.

Konsep yang dipelajari:
- Augmented Reality (AR) sederhana menggunakan feature matching
- Deteksi marker planar menggunakan fitur lokal
- Estimasi homography untuk menentukan posisi marker
- Warp perspektif: mentransformasi overlay ke sudut pandang scene
- Blending: menggabungkan overlay dengan scene secara halus
- Pengujian pada berbagai sudut dan perspektif

Fungsi utama yang dipelajari:
- cv2.findHomography()     : Estimasi transformasi perspektif (homography)
- cv2.warpPerspective()    : Mentransformasi gambar menggunakan homography
- cv2.addWeighted()        : Blending dua gambar dengan bobot alpha
- cv2.polylines()          : Menggambar batas polyline (bounding box)
- cv2.perspectiveTransform() : Mentransformasi titik menggunakan homography

Hasil: Visualisasi deteksi marker dan overlay konten virtual
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan geometri
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
print("PERCOBAAN 16: DETEKSI AR MARKER DAN OVERLAY")
print("=" * 60)

# ============================================================
# 1. Memuat Gambar AR Marker dan Membuat Scene Sintetis
# ============================================================

# Menampilkan header bagian
print("\n--- 1. Memuat AR Marker dan Membuat Scene ---")

# Membaca gambar AR marker
img_marker = cv2.imread(os.path.join(IMAGE_DIR, "ar_marker.jpg"))

# Memeriksa apakah marker berhasil dimuat
if img_marker is None:
    print("[ERROR] ar_marker.jpg tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan ukuran marker
print(f"[INFO] Ukuran AR marker: {img_marker.shape}")

# Mendapatkan dimensi marker
h_marker, w_marker = img_marker.shape[:2]


# Mendefinisikan fungsi untuk membuat scene sintetis dengan marker
def buat_scene_dengan_marker(marker, sudut_rotasi=0, skala=1.0, posisi_x=250, posisi_y=100):
    """Membuat scene sintetis dengan marker yang ditransformasi."""
    # Membuat background scene
    scene = np.ones((500, 700, 3), dtype=np.uint8) * 180

    # Menambahkan tekstur gradien pada background
    for y in range(500):
        scene[y, :] = np.clip(180 + (y - 250) * 0.2, 100, 220)

    # Menambahkan beberapa elemen dekoratif ke scene
    cv2.rectangle(scene, (10, 10), (690, 490), (150, 150, 150), 3)
    cv2.putText(scene, "AR SCENE", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # Mendapatkan dimensi marker
    h, w = marker.shape[:2]

    # Mendefinisikan titik sudut sumber marker
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Menghitung ukuran marker yang diskalakan
    sw = int(w * skala)
    sh = int(h * skala)

    # Mendefinisikan titik sudut tujuan (dengan rotasi dan posisi)
    cos_a = np.cos(np.radians(sudut_rotasi))
    sin_a = np.sin(np.radians(sudut_rotasi))

    # Menghitung titik sudut tujuan dengan transformasi
    cx = posisi_x + sw // 2
    cy = posisi_y + sh // 2

    # Mendefinisikan 4 sudut relatif terhadap pusat
    corners = np.array([[-sw/2, -sh/2], [sw/2, -sh/2], [sw/2, sh/2], [-sw/2, sh/2]])

    # Merotasi setiap sudut
    rotated_corners = []
    for corner in corners:
        rx = corner[0] * cos_a - corner[1] * sin_a + cx
        ry = corner[0] * sin_a + corner[1] * cos_a + cy
        rotated_corners.append([rx, ry])

    # Mengkonversi ke format float32
    dst_pts = np.float32(rotated_corners)

    # Menghitung homography dari sumber ke tujuan
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Melakukan warp perspektif marker ke scene
    warped = cv2.warpPerspective(marker, M, (700, 500))

    # Membuat mask dari marker yang sudah di-warp
    mask_full = cv2.warpPerspective(np.ones_like(marker) * 255, M, (700, 500))

    # Mengkonversi mask ke boolean
    mask_bool = mask_full > 128

    # Menggabungkan marker yang sudah di-warp ke scene
    scene[mask_bool] = warped[mask_bool]

    # Mengembalikan scene dan titik tujuan
    return scene, dst_pts, M


# Membuat scene dengan marker di posisi default
scene_default, dst_pts_default, M_default = buat_scene_dengan_marker(img_marker)

# Menampilkan informasi scene yang dibuat
print(f"[INFO] Scene default dibuat: {scene_default.shape}")

# ============================================================
# 2. Deteksi AR Marker dalam Scene
# ============================================================

# Menampilkan header bagian
print("\n--- 2. Deteksi AR Marker dalam Scene ---")

# Membuat detektor ORB
orb = cv2.ORB_create(nfeatures=1000)

# Mengkonversi marker ke grayscale
gray_marker = cv2.cvtColor(img_marker, cv2.COLOR_BGR2GRAY)

# Mendeteksi keypoint dan deskriptor pada marker
kp_marker, des_marker = orb.detectAndCompute(gray_marker, None)

# Menampilkan jumlah keypoint pada marker
print(f"[INFO] Keypoint pada marker: {len(kp_marker)}")

# Mengkonversi scene ke grayscale
gray_scene = cv2.cvtColor(scene_default, cv2.COLOR_BGR2GRAY)

# Mendeteksi keypoint dan deskriptor pada scene
kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

# Menampilkan jumlah keypoint pada scene
print(f"[INFO] Keypoint pada scene: {len(kp_scene)}")

# Membuat matcher Brute-Force dengan Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Melakukan KNN matching
matches = bf.knnMatch(des_marker, des_scene, k=2)

# Menerapkan ratio test Lowe
good_matches = []
for m_pair in matches:
    if len(m_pair) == 2:
        m, n = m_pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

# Menampilkan jumlah good matches
print(f"[INFO] Good matches: {len(good_matches)}")

# Menentukan minimum matches untuk homography
MIN_MATCH_COUNT = 10

# Memeriksa apakah cukup matches untuk estimasi homography
if len(good_matches) >= MIN_MATCH_COUNT:
    # Mengekstrak titik sumber dari keypoint marker
    src_pts_match = np.float32([kp_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Mengekstrak titik tujuan dari keypoint scene
    dst_pts_match = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Mengestimasi homography menggunakan RANSAC
    H, mask = cv2.findHomography(src_pts_match, dst_pts_match, cv2.RANSAC, 5.0)

    # Menghitung jumlah inlier
    inliers = mask.ravel().tolist().count(1)
    print(f"[INFO] Homography ditemukan! Inliers: {inliers}/{len(good_matches)}")

    # Mendefinisikan sudut-sudut marker
    h_m, w_m = img_marker.shape[:2]
    corner_pts = np.float32([[0, 0], [w_m, 0], [w_m, h_m], [0, h_m]]).reshape(-1, 1, 2)

    # Mentransformasi sudut-sudut ke koordinat scene
    detected_corners = cv2.perspectiveTransform(corner_pts, H)

    # Menampilkan koordinat sudut yang terdeteksi
    print(f"[INFO] Sudut marker terdeteksi di scene:")
    for i, corner in enumerate(detected_corners):
        print(f"  Sudut {i+1}: ({corner[0][0]:.1f}, {corner[0][1]:.1f})")
else:
    print(f"[WARNING] Tidak cukup matches ({len(good_matches)}/{MIN_MATCH_COUNT})")
    H = None
    detected_corners = None

# ============================================================
# 3. Visualisasi Deteksi Marker
# ============================================================

# Menampilkan header bagian
print("\n--- 3. Visualisasi Deteksi ---")

# Membuat figure untuk deteksi marker
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Menampilkan marker asli dengan keypoint
img_marker_kp = cv2.drawKeypoints(img_marker, kp_marker, None, color=(0, 255, 0))
axes[0].imshow(cv2.cvtColor(img_marker_kp, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'AR Marker\n{len(kp_marker)} keypoints', fontsize=12)
axes[0].axis('off')

# Membuat salinan scene untuk menggambar hasil deteksi
scene_deteksi = scene_default.copy()

# Menggambar bounding box jika homography berhasil ditemukan
if detected_corners is not None:
    # Mengkonversi sudut ke integer
    corners_int = np.int32(detected_corners)

    # Menggambar polyline bounding box
    cv2.polylines(scene_deteksi, [corners_int], True, (0, 255, 0), 3)

    # Menambahkan label "MARKER DETECTED"
    cx_det = int(np.mean(corners_int[:, 0, 0]))
    cy_det = int(np.mean(corners_int[:, 0, 1]))
    cv2.putText(scene_deteksi, "DETECTED", (cx_det - 50, cy_det),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Menampilkan scene dengan hasil deteksi
axes[1].imshow(cv2.cvtColor(scene_deteksi, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Deteksi Marker di Scene\nGood Matches: {len(good_matches)}', fontsize=12)
axes[1].axis('off')

# Menampilkan feature matching menggunakan drawMatches
img_matches = cv2.drawMatches(img_marker, kp_marker, scene_default, kp_scene,
                              good_matches[:30], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
axes[2].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'Feature Matching\n(menampilkan 30 dari {len(good_matches)} matches)', fontsize=12)
axes[2].axis('off')

# Menambahkan judul utama
fig.suptitle('Percobaan 16: Deteksi AR Marker', fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi deteksi
output_path_det = os.path.join(OUTPUT_DIR, "16_ar_deteksi.png")
plt.savefig(output_path_det, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_det}")

# ============================================================
# 4. Overlay Konten Virtual pada Marker
# ============================================================

# Menampilkan header bagian
print("\n--- 4. Overlay Konten Virtual ---")

# Membuat konten virtual (persegi panjang berwarna dengan teks)
overlay_content = np.zeros((h_marker, w_marker, 3), dtype=np.uint8)

# Mengisi overlay dengan gradien warna biru-hijau
for y in range(h_marker):
    for x in range(w_marker):
        overlay_content[y, x] = [int(255 * y / h_marker), int(255 * x / w_marker), 100]

# Menambahkan teks pada overlay
cv2.putText(overlay_content, "VIRTUAL", (30, h_marker // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
cv2.putText(overlay_content, "CONTENT", (25, h_marker // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

# Menambahkan border pada overlay
cv2.rectangle(overlay_content, (5, 5), (w_marker - 5, h_marker - 5), (255, 255, 0), 3)

# Memeriksa apakah homography tersedia untuk overlay
if H is not None:
    # Melakukan warp overlay menggunakan homography yang ditemukan
    overlay_warped = cv2.warpPerspective(overlay_content, H, (scene_default.shape[1], scene_default.shape[0]))

    # Membuat mask dari overlay yang sudah di-warp
    mask_overlay = cv2.warpPerspective(np.ones((h_marker, w_marker), dtype=np.uint8) * 255,
                                        H, (scene_default.shape[1], scene_default.shape[0]))

    # Membuat salinan scene untuk overlay
    scene_overlay = scene_default.copy()

    # Mendapatkan region dimana overlay akan ditempatkan
    mask_bool_overlay = mask_overlay > 128

    # Melakukan blending overlay dengan scene pada daerah mask
    alpha = 0.7  # bobot overlay
    beta = 0.3   # bobot scene

    # Menerapkan overlay pada region mask
    for c in range(3):
        scene_overlay[:, :, c] = np.where(
            mask_bool_overlay,
            cv2.addWeighted(overlay_warped[:, :, c], alpha, scene_default[:, :, c], beta, 0),
            scene_overlay[:, :, c]
        )

    # Menggambar bounding box overlay
    if detected_corners is not None:
        cv2.polylines(scene_overlay, [np.int32(detected_corners)], True, (0, 255, 255), 2)

    # Menampilkan informasi overlay
    print(f"[INFO] Overlay berhasil diterapkan dengan alpha={alpha}")
else:
    # Menggunakan scene tanpa overlay jika homography tidak tersedia
    scene_overlay = scene_default.copy()
    print("[WARNING] Overlay tidak dapat diterapkan (homography tidak tersedia)")

# Membuat figure untuk visualisasi overlay
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Menampilkan konten virtual (overlay)
axes[0].imshow(cv2.cvtColor(overlay_content, cv2.COLOR_BGR2RGB))
axes[0].set_title('Konten Virtual (Overlay)', fontsize=12)
axes[0].axis('off')

# Menampilkan scene asli
axes[1].imshow(cv2.cvtColor(scene_default, cv2.COLOR_BGR2RGB))
axes[1].set_title('Scene dengan Marker', fontsize=12)
axes[1].axis('off')

# Menampilkan scene dengan overlay
axes[2].imshow(cv2.cvtColor(scene_overlay, cv2.COLOR_BGR2RGB))
axes[2].set_title('Scene dengan Overlay Virtual', fontsize=12)
axes[2].axis('off')

# Menambahkan judul utama
fig.suptitle('Percobaan 16: AR Overlay pada Marker', fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi overlay
output_path_overlay = os.path.join(OUTPUT_DIR, "16_ar_overlay.png")
plt.savefig(output_path_overlay, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_overlay}")

# ============================================================
# 5. Pengujian pada Berbagai Sudut dan Perspektif
# ============================================================

# Menampilkan header bagian
print("\n--- 5. Pengujian Multi-Angle ---")

# Mendefinisikan konfigurasi pengujian dengan berbagai transformasi
konfigurasi_test = [
    {'sudut': 0, 'skala': 0.8, 'pos_x': 200, 'pos_y': 100, 'label': 'Normal (0°)'},
    {'sudut': 15, 'skala': 0.7, 'pos_x': 250, 'pos_y': 80, 'label': 'Rotasi 15°'},
    {'sudut': 30, 'skala': 0.6, 'pos_x': 300, 'pos_y': 120, 'label': 'Rotasi 30°'},
    {'sudut': -20, 'skala': 0.9, 'pos_x': 150, 'pos_y': 60, 'label': 'Rotasi -20°'},
    {'sudut': 45, 'skala': 0.5, 'pos_x': 280, 'pos_y': 150, 'label': 'Rotasi 45°'},
    {'sudut': 0, 'skala': 1.2, 'pos_x': 100, 'pos_y': 50, 'label': 'Skala Besar'},
]

# Membuat figure untuk pengujian multi-angle
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Menyiapkan list untuk menyimpan statistik per konfigurasi
statistik_multi = []

# Melakukan iterasi untuk setiap konfigurasi
for idx, config in enumerate(konfigurasi_test):
    # Menentukan posisi subplot
    row = idx // 3
    col = idx % 3

    # Membuat scene dengan marker pada konfigurasi ini
    scene_test, dst_pts_test, M_test = buat_scene_dengan_marker(
        img_marker, config['sudut'], config['skala'], config['pos_x'], config['pos_y']
    )

    # Mengkonversi scene ke grayscale
    gray_test = cv2.cvtColor(scene_test, cv2.COLOR_BGR2GRAY)

    # Mendeteksi keypoint dan deskriptor pada scene test
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    # Mengukur waktu deteksi
    waktu_mulai = time.time()

    # Melakukan matching
    if des_test is not None and des_marker is not None:
        matches_test = bf.knnMatch(des_marker, des_test, k=2)

        # Menerapkan ratio test
        good_test = []
        for m_pair in matches_test:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_test.append(m)
    else:
        good_test = []

    # Menghitung waktu deteksi
    waktu_deteksi = time.time() - waktu_mulai

    # Membuat salinan scene untuk visualisasi
    scene_vis = scene_test.copy()

    # Memeriksa apakah cukup matches untuk homography
    deteksi_berhasil = False
    if len(good_test) >= MIN_MATCH_COUNT:
        # Mengekstrak titik
        src_pts_t = np.float32([kp_marker[m.queryIdx].pt for m in good_test]).reshape(-1, 1, 2)
        dst_pts_t = np.float32([kp_test[m.trainIdx].pt for m in good_test]).reshape(-1, 1, 2)

        # Mengestimasi homography
        H_test, mask_test = cv2.findHomography(src_pts_t, dst_pts_t, cv2.RANSAC, 5.0)

        if H_test is not None:
            # Menghitung sudut marker yang terdeteksi
            corners_test = cv2.perspectiveTransform(
                np.float32([[0, 0], [w_marker, 0], [w_marker, h_marker], [0, h_marker]]).reshape(-1, 1, 2),
                H_test
            )

            # Menggambar bounding box
            cv2.polylines(scene_vis, [np.int32(corners_test)], True, (0, 255, 0), 3)

            # Melakukan overlay
            overlay_w = cv2.warpPerspective(overlay_content, H_test, (scene_test.shape[1], scene_test.shape[0]))
            mask_w = cv2.warpPerspective(np.ones((h_marker, w_marker), dtype=np.uint8) * 255,
                                          H_test, (scene_test.shape[1], scene_test.shape[0]))
            mask_b = mask_w > 128

            for c in range(3):
                scene_vis[:, :, c] = np.where(
                    mask_b,
                    cv2.addWeighted(overlay_w[:, :, c], 0.7, scene_vis[:, :, c], 0.3, 0),
                    scene_vis[:, :, c]
                )

            # Menandai deteksi berhasil
            deteksi_berhasil = True
            inliers_test = mask_test.ravel().tolist().count(1)
        else:
            inliers_test = 0
    else:
        inliers_test = 0

    # Menyimpan statistik
    statistik_multi.append({
        'label': config['label'],
        'matches': len(good_test),
        'inliers': inliers_test,
        'terdeteksi': deteksi_berhasil,
        'waktu': waktu_deteksi
    })

    # Menampilkan hasil di subplot
    axes[row, col].imshow(cv2.cvtColor(scene_vis, cv2.COLOR_BGR2RGB))
    status = "OK" if deteksi_berhasil else "GAGAL"
    warna_title = 'green' if deteksi_berhasil else 'red'
    axes[row, col].set_title(
        f'{config["label"]} [{status}]\nMatches: {len(good_test)}, Inliers: {inliers_test}',
        fontsize=11, color=warna_title
    )
    axes[row, col].axis('off')

    # Menampilkan informasi di konsol
    print(f"  [{status}] {config['label']}: matches={len(good_test)}, inliers={inliers_test}, "
          f"waktu={waktu_deteksi:.3f}s")

# Menambahkan judul utama
fig.suptitle('Percobaan 16: Deteksi AR Marker - Multi Angle/Scale',
             fontsize=14, fontweight='bold')

# Mengatur layout
plt.tight_layout()

# Menyimpan hasil visualisasi multi-angle
output_path_multi = os.path.join(OUTPUT_DIR, "16_ar_multi_angle.png")
plt.savefig(output_path_multi, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Menampilkan konfirmasi penyimpanan
print(f"[SAVED] {output_path_multi}")

# ============================================================
# 6. Statistik Pengujian
# ============================================================

# Menampilkan header bagian
print("\n--- 6. Statistik Pengujian Multi-Angle ---")

# Menghitung jumlah deteksi yang berhasil
total_berhasil = sum(1 for s in statistik_multi if s['terdeteksi'])

# Menampilkan tingkat keberhasilan
print(f"[INFO] Tingkat keberhasilan: {total_berhasil}/{len(statistik_multi)} "
      f"({100 * total_berhasil / len(statistik_multi):.1f}%)")

# Menghitung rata-rata matches
rata_matches = np.mean([s['matches'] for s in statistik_multi])

# Menampilkan rata-rata matches
print(f"[INFO] Rata-rata good matches: {rata_matches:.1f}")

# Menghitung rata-rata waktu
rata_waktu = np.mean([s['waktu'] for s in statistik_multi])

# Menampilkan rata-rata waktu
print(f"[INFO] Rata-rata waktu deteksi: {rata_waktu:.3f} detik")

# ============================================================
# 7. Ringkasan Percobaan
# ============================================================

# Menampilkan garis pemisah ringkasan
print("\n" + "=" * 60)

# Menampilkan judul ringkasan
print("RINGKASAN PERCOBAAN 16: DETEKSI AR MARKER DAN OVERLAY")

# Menampilkan garis pemisah
print("=" * 60)

# Menampilkan penjelasan deteksi marker
print("1. AR Marker dideteksi menggunakan feature matching (ORB)")
print("   dan homography estimation dengan RANSAC.")

# Menampilkan penjelasan overlay
print("2. Konten virtual di-overlay menggunakan warp perspektif")
print("   (cv2.warpPerspective) dan blending (cv2.addWeighted).")

# Menampilkan penjelasan sudut
print("3. Deteksi bekerja pada berbagai sudut rotasi dan skala,")
print("   namun robustness menurun pada sudut/skala ekstrem.")

# Menampilkan penjelasan threshold
print("4. Minimum 10 good matches diperlukan untuk estimasi")
print("   homography yang reliable.")

# Menampilkan daftar file output
print("\nFile output yang dihasilkan:")
print("  - 16_ar_deteksi.png")
print("  - 16_ar_overlay.png")
print("  - 16_ar_multi_angle.png")

# Menampilkan garis penutup
print("=" * 60)
