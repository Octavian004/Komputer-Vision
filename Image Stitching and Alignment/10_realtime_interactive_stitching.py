"""
==========================================================================
PERCOBAAN 10: REAL-TIME / INTERACTIVE STITCHING
==========================================================================
Program ini membangun aplikasi stitching interaktif menggunakan webcam
atau simulasi frame video dari gambar panorama yang sudah ada. Program
mengimplementasikan pipeline stitching cepat menggunakan ORB (Oriented
FAST and Rotated BRIEF) yang lebih cepat dari SIFT untuk skenario
real-time.

Konsep yang dipelajari:
- Pipeline stitching real-time (deteksi → match → warp → composite)
- Perbandingan ORB vs SIFT untuk kecepatan stitching
- Homography caching untuk stabilisasi real-time
- Feather blending sederhana untuk stitching cepat
- Simulasi video frame dari panorama images
- Pengukuran FPS dan performa pipeline

Fungsi utama yang dipelajari:
- cv2.ORB_create()       : Detektor fitur cepat untuk real-time
- cv2.BFMatcher()        : Matcher brute-force untuk ORB (binary desc)
- cv2.findHomography()   : Estimasi homography real-time
- cv2.warpPerspective()  : Warping real-time
- cv2.VideoCapture()     : Menangkap video dari webcam/file
- time.time()            : Mengukur FPS dan latency
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar dan computer vision
import cv2

# Mengimpor library NumPy untuk operasi array dan matriks
import numpy as np

# Mengimpor library os untuk operasi path file dan folder
import os

# Mengimpor matplotlib untuk visualisasi dan grid perbandingan
import matplotlib.pyplot as plt

# Mengimpor modul time untuk mengukur FPS dan waktu eksekusi
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
print("PERCOBAAN 10: REAL-TIME / INTERACTIVE STITCHING")
print("=" * 65)


# ============================================================
# LANGKAH 1: Konfigurasi Mode Operasi
# ============================================================
print("\n[LANGKAH 1] Mengonfigurasi mode operasi...")

# Mode operasi: 'simulation' (default) atau 'webcam'
# Simulasi menggunakan gambar panorama yang sudah ada sebagai frame
MODE = 'simulation'

# Mencoba mendeteksi webcam
print("  Memeriksa ketersediaan webcam...")
webcam_available = False
try:
    cap_test = cv2.VideoCapture(0)
    if cap_test.isOpened():
        ret, frame = cap_test.read()
        if ret:
            webcam_available = True
            print(f"  [OK] Webcam terdeteksi: {frame.shape[1]}x{frame.shape[0]}")
    cap_test.release()
except Exception as e:
    print(f"  [INFO] Webcam tidak tersedia: {e}")

if not webcam_available:
    print("  [INFO] Menggunakan mode SIMULASI (gambar panorama sebagai frame)")
    MODE = 'simulation'
else:
    # Tetap gunakan simulasi sebagai default untuk konsistensi output
    print("  [INFO] Webcam tersedia, tetap menggunakan mode SIMULASI untuk demo")
    MODE = 'simulation'


# ============================================================
# LANGKAH 2: Memuat Gambar untuk Simulasi Frame
# ============================================================
print("\n[LANGKAH 2] Memuat gambar untuk simulasi frame video...")

# Memuat gambar panorama outdoor sebagai "frame" video
sim_frames = []
frame_sources = []

# Prioritas 1: panorama outdoor (3 gambar)
for i in range(1, 4):
    path = os.path.join(IMAGE_DIR, f"panorama_outdoor_{i}.jpg")
    img = cv2.imread(path)
    if img is not None:
        sim_frames.append(img)
        frame_sources.append(f"panorama_outdoor_{i}.jpg")

# Prioritas 2: jika tidak cukup, gunakan pair images
if len(sim_frames) < 2:
    for name in ["pair_left.jpg", "pair_right.jpg"]:
        path = os.path.join(IMAGE_DIR, name)
        img = cv2.imread(path)
        if img is not None:
            sim_frames.append(img)
            frame_sources.append(name)

# Memastikan ada cukup frame
if len(sim_frames) < 2:
    print("  [ERROR] Tidak cukup gambar untuk simulasi!")
    print("  Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan frame yang dimuat
for i, (frame, src) in enumerate(zip(sim_frames, frame_sources)):
    print(f"  Frame {i + 1}: {src} ({frame.shape[1]}x{frame.shape[0]})")

print(f"  Total frame simulasi: {len(sim_frames)}")


# ============================================================
# LANGKAH 3: Implementasi Pipeline Stitching Cepat (ORB)
# ============================================================
print("\n[LANGKAH 3] Mengimplementasikan pipeline stitching cepat (ORB)...")


class FastStitcher:
    """
    Kelas untuk stitching real-time menggunakan ORB features.
    Lebih cepat dari SIFT tapi kurang akurat untuk beberapa skenario.
    """

    def __init__(self, n_features=1000, match_threshold=0.75):
        """
        Inisialisasi fast stitcher.

        Parameter:
        - n_features      : Jumlah fitur ORB yang dideteksi
        - match_threshold : Threshold untuk distance ratio test
        """
        # Membuat detektor ORB dengan jumlah fitur yang dikonfigurasi
        self.orb = cv2.ORB_create(nfeatures=n_features)

        # Membuat BFMatcher untuk deskriptor binary (ORB menggunakan BRIEF)
        # cv2.NORM_HAMMING cocok untuk deskriptor binary
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Menyimpan threshold dan cache
        self.match_threshold = match_threshold
        self.cached_H = None          # Homography yang di-cache
        self.cache_inlier_count = 0   # Jumlah inlier terakhir
        self.reference_frame = None   # Frame referensi
        self.reference_kp = None      # Keypoints referensi
        self.reference_desc = None    # Deskriptor referensi

        # Statistik performa
        self.stats = {
            'total_frames': 0,
            'total_matches': 0,
            'total_inliers': 0,
            'total_time': 0,
            'cache_hits': 0,
            'successful_stitches': 0
        }

    def set_reference(self, frame):
        """
        Menetapkan frame referensi untuk stitching.

        Parameter:
        - frame : Gambar BGR sebagai referensi
        """
        # Menyimpan frame referensi
        self.reference_frame = frame.copy()

        # Mengkonversi ke grayscale untuk deteksi fitur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mendeteksi ORB keypoints dan deskriptor
        self.reference_kp, self.reference_desc = self.orb.detectAndCompute(gray, None)

        # Mereset cache homography
        self.cached_H = None
        self.cache_inlier_count = 0

        n_kp = len(self.reference_kp) if self.reference_kp is not None else 0
        print(f"    Referensi ditetapkan: {n_kp} keypoints ORB")

    def match_and_homography(self, frame, use_cache=True):
        """
        Mencocokkan frame baru dengan referensi dan menghitung homography.

        Parameter:
        - frame     : Frame baru yang akan di-stitch
        - use_cache : Gunakan homography cache jika stabil

        Returns:
        - H         : Matriks homography 3x3
        - n_matches : Jumlah good matches
        - n_inliers : Jumlah inlier RANSAC
        - dt        : Waktu proses (detik)
        """
        t_start = time.time()

        # Mendeteksi fitur pada frame baru
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_new, desc_new = self.orb.detectAndCompute(gray, None)

        # Memeriksa validitas deskriptor
        if (desc_new is None or self.reference_desc is None or
                len(desc_new) < 4 or len(self.reference_desc) < 4):
            dt = time.time() - t_start
            return None, 0, 0, dt

        # Melakukan kNN matching (k=2) untuk ratio test
        matches = self.bf.knnMatch(desc_new, self.reference_desc, k=2)

        # Menerapkan ratio test
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.match_threshold * n.distance:
                    good.append(m)

        n_matches = len(good)

        # Memastikan ada cukup kecocokan
        if n_matches < 4:
            dt = time.time() - t_start
            # Jika ada cache, gunakan cache
            if use_cache and self.cached_H is not None:
                self.stats['cache_hits'] += 1
                return self.cached_H, n_matches, 0, dt
            return None, n_matches, 0, dt

        # Mengekstrak titik korespondensi
        src_pts = np.float32([kp_new[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.reference_kp[m.trainIdx].pt
                              for m in good]).reshape(-1, 1, 2)

        # Mengestimasi homography dengan RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        n_inliers = int(mask.ravel().sum()) if mask is not None else 0

        dt = time.time() - t_start

        if H is not None:
            # Caching: simpan homography jika jumlah inlier stabil
            if use_cache:
                if (self.cached_H is not None and
                        abs(n_inliers - self.cache_inlier_count) < 10):
                    # Homography stabil, gunakan rata-rata berbobot
                    alpha = 0.7  # Bobot untuk H baru
                    self.cached_H = alpha * H + (1 - alpha) * self.cached_H
                else:
                    self.cached_H = H.copy()
                self.cache_inlier_count = n_inliers

            # Update statistik
            self.stats['total_matches'] += n_matches
            self.stats['total_inliers'] += n_inliers
            self.stats['successful_stitches'] += 1

        self.stats['total_frames'] += 1
        self.stats['total_time'] += dt

        return H, n_matches, n_inliers, dt

    def stitch(self, frame, H=None):
        """
        Melakukan stitching frame baru dengan frame referensi.

        Parameter:
        - frame : Frame baru
        - H     : Homography (jika None, akan dihitung otomatis)

        Returns:
        - result    : Gambar panorama hasil stitching
        - H_used    : Homography yang digunakan
        - n_matches : Jumlah matches
        - n_inliers : Jumlah inliers
        - dt        : Waktu proses
        """
        if self.reference_frame is None:
            return frame.copy(), None, 0, 0, 0

        if H is None:
            H, n_matches, n_inliers, dt = self.match_and_homography(frame)
        else:
            n_matches, n_inliers, dt = 0, 0, 0

        if H is None:
            return self.reference_frame.copy(), None, n_matches, n_inliers, dt

        h_ref, w_ref = self.reference_frame.shape[:2]
        h_new, w_new = frame.shape[:2]

        # Menghitung batas canvas
        corners_new = np.float32([[0, 0], [w_new, 0], [w_new, h_new],
                                   [0, h_new]]).reshape(-1, 1, 2)
        corners_trans = cv2.perspectiveTransform(corners_new, H)

        all_corners = np.concatenate([
            np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref],
                         [0, h_ref]]).reshape(-1, 1, 2),
            corners_trans
        ])

        x_min = int(np.floor(all_corners[:, 0, 0].min()))
        y_min = int(np.floor(all_corners[:, 0, 1].min()))
        x_max = int(np.ceil(all_corners[:, 0, 0].max()))
        y_max = int(np.ceil(all_corners[:, 0, 1].max()))

        canvas_w = min(x_max - x_min, 4000)
        canvas_h = min(y_max - y_min, 2000)

        # Matriks translasi
        H_tr = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]],
                          dtype=np.float64)

        # Warping frame baru
        warped_new = cv2.warpPerspective(frame, H_tr @ H,
                                          (canvas_w, canvas_h))

        # Menempatkan referensi
        result = warped_new.copy()
        ox, oy = -x_min, -y_min
        y1 = max(0, oy)
        y2 = min(canvas_h, oy + h_ref)
        x1 = max(0, ox)
        x2 = min(canvas_w, ox + w_ref)
        sy1, sx1 = max(0, -oy), max(0, -ox)
        ah, aw = y2 - y1, x2 - x1

        if ah > 0 and aw > 0:
            ref_region = self.reference_frame[sy1:sy1 + ah, sx1:sx1 + aw]
            canvas_region = result[y1:y1 + ah, x1:x1 + aw]

            # Feather blending sederhana
            mask_ref = (cv2.cvtColor(ref_region, cv2.COLOR_BGR2GRAY) > 0)
            mask_warp = (cv2.cvtColor(canvas_region, cv2.COLOR_BGR2GRAY) > 0)
            overlap = mask_ref & mask_warp
            only_ref = mask_ref & ~mask_warp

            # Di area overlap: rata-rata
            canvas_region[overlap] = (
                (ref_region[overlap].astype(np.float32) +
                 canvas_region[overlap].astype(np.float32)) / 2
            ).astype(np.uint8)

            # Di area hanya referensi: gunakan referensi
            canvas_region[only_ref] = ref_region[only_ref]

            result[y1:y1 + ah, x1:x1 + aw] = canvas_region

        return result, H, n_matches, n_inliers, dt

    def get_stats_string(self):
        """Mengembalikan string statistik performa."""
        s = self.stats
        if s['total_frames'] == 0:
            return "Belum ada frame yang diproses"
        avg_time = s['total_time'] / s['total_frames']
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_matches = s['total_matches'] / max(s['successful_stitches'], 1)
        avg_inliers = s['total_inliers'] / max(s['successful_stitches'], 1)
        return (f"Frames: {s['total_frames']}, "
                f"FPS: {avg_fps:.1f}, "
                f"Avg matches: {avg_matches:.0f}, "
                f"Avg inliers: {avg_inliers:.0f}, "
                f"Cache hits: {s['cache_hits']}")


# Membuat instance stitcher
fast_stitcher = FastStitcher(n_features=1500, match_threshold=0.75)
print("  FastStitcher (ORB) berhasil diinisialisasi.")
print(f"    Fitur ORB: 1500")
print(f"    Matcher: BFMatcher (NORM_HAMMING)")


# ============================================================
# LANGKAH 4: Perbandingan Kecepatan ORB vs SIFT
# ============================================================
print("\n[LANGKAH 4] Membandingkan kecepatan ORB vs SIFT...")

# Menggunakan frame pertama untuk benchmark
bench_frame = sim_frames[0]
gray_bench = cv2.cvtColor(bench_frame, cv2.COLOR_BGR2GRAY)

# Benchmark ORB
orb_test = cv2.ORB_create(nfeatures=1500)
t_start = time.time()
for _ in range(10):
    kp_orb, desc_orb = orb_test.detectAndCompute(gray_bench, None)
t_orb = (time.time() - t_start) / 10
n_kp_orb = len(kp_orb) if kp_orb else 0

# Benchmark SIFT
sift_test = cv2.SIFT_create()
t_start = time.time()
for _ in range(10):
    kp_sift, desc_sift = sift_test.detectAndCompute(gray_bench, None)
t_sift = (time.time() - t_start) / 10
n_kp_sift = len(kp_sift) if kp_sift else 0

# Menampilkan hasil benchmark
print(f"  ORB:  {t_orb * 1000:.1f} ms/frame, {n_kp_orb} keypoints")
print(f"  SIFT: {t_sift * 1000:.1f} ms/frame, {n_kp_sift} keypoints")
speedup = t_sift / t_orb if t_orb > 0 else 0
print(f"  ORB {speedup:.1f}x lebih cepat dari SIFT")


# ============================================================
# LANGKAH 5: Simulasi Real-Time Stitching (2 Frame)
# ============================================================
print("\n[LANGKAH 5] Menjalankan simulasi real-time stitching (2 frame)...")

# Menetapkan frame pertama sebagai referensi
fast_stitcher.set_reference(sim_frames[0])

# Melakukan stitching dengan frame kedua
result_2, H_2, matches_2, inliers_2, dt_2 = fast_stitcher.stitch(sim_frames[1])

if H_2 is not None:
    fps_2 = 1.0 / dt_2 if dt_2 > 0 else 0

    # Menambahkan info FPS dan matches pada hasil
    vis_2 = result_2.copy()
    cv2.putText(vis_2, f"FPS: {fps_2:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis_2, f"Matches: {matches_2} | Inliers: {inliers_2}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Crop border hitam
    gray_r2 = cv2.cvtColor(result_2, cv2.COLOR_BGR2GRAY)
    _, thresh_r2 = cv2.threshold(gray_r2, 1, 255, cv2.THRESH_BINARY)
    cnt_r2, _ = cv2.findContours(thresh_r2, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if cnt_r2:
        lg = max(cnt_r2, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(lg)
        result_2_crop = result_2[y:y + h, x:x + w]
    else:
        result_2_crop = result_2

    cv2.imwrite(os.path.join(OUTPUT_DIR, "10_realtime_2frame.jpg"), result_2_crop)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "10_realtime_2frame_info.jpg"), vis_2)
    print(f"  [OK] 2-frame stitch: {dt_2 * 1000:.1f}ms, "
          f"FPS={fps_2:.1f}, matches={matches_2}, inliers={inliers_2}")
else:
    print("  [WARNING] 2-frame stitching gagal (tidak cukup matches)")
    result_2_crop = sim_frames[0]


# ============================================================
# LANGKAH 6: Stitching 3 Frame Berurutan
# ============================================================
print("\n[LANGKAH 6] Melakukan stitching 3 frame berurutan...")

if len(sim_frames) >= 3:
    # Reset stitcher
    fast_stitcher_3 = FastStitcher(n_features=1500, match_threshold=0.75)

    # Frame 1 sebagai referensi
    fast_stitcher_3.set_reference(sim_frames[0])

    # Stitch frame 2
    result_12, H_12, _, _, dt_12 = fast_stitcher_3.stitch(sim_frames[1])

    if H_12 is not None and result_12 is not None:
        # Gunakan hasil 1+2 sebagai referensi baru
        fast_stitcher_3.set_reference(result_12)

        # Stitch frame 3
        result_123, H_123, matches_3, inliers_3, dt_123 = fast_stitcher_3.stitch(
            sim_frames[2])

        if result_123 is not None:
            # Crop
            gray_r3 = cv2.cvtColor(result_123, cv2.COLOR_BGR2GRAY)
            _, thresh_r3 = cv2.threshold(gray_r3, 1, 255, cv2.THRESH_BINARY)
            cnt_r3, _ = cv2.findContours(thresh_r3, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            if cnt_r3:
                lg = max(cnt_r3, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(lg)
                result_123_crop = result_123[y:y + h, x:x + w]
            else:
                result_123_crop = result_123

            cv2.imwrite(os.path.join(OUTPUT_DIR, "10_realtime_3frame.jpg"),
                        result_123_crop)
            total_dt = dt_12 + dt_123
            print(f"  [OK] 3-frame stitch: total {total_dt * 1000:.1f}ms")
            print(f"  Ukuran: {result_123_crop.shape[1]}x{result_123_crop.shape[0]}")
        else:
            print("  [WARNING] 3-frame stitching gagal pada frame 3")
            result_123_crop = result_12
    else:
        print("  [WARNING] 3-frame stitching gagal pada frame 2")
        result_123_crop = sim_frames[0]
else:
    print("  [INFO] Hanya 2 frame tersedia, lewati 3-frame stitch")
    result_123_crop = result_2_crop


# ============================================================
# LANGKAH 7: Simulasi Video Stream (Banyak Frame)
# ============================================================
print("\n[LANGKAH 7] Menjalankan simulasi video stream...")

# Membuat variasi frame dari gambar yang ada (simulasi pergerakan kamera)
# Menggunakan crop dari gambar lebar untuk simulasi panning
sim_video_frames = []

for img in sim_frames:
    h, w = img.shape[:2]
    # Membuat beberapa variasi: crop kiri, tengah, kanan
    crop_w = int(w * 0.8)
    offsets = [0, (w - crop_w) // 4, (w - crop_w) // 2,
               3 * (w - crop_w) // 4, w - crop_w]
    for offset in offsets:
        crop = img[:, offset:offset + crop_w].copy()
        sim_video_frames.append(crop)

print(f"  Total frame simulasi: {len(sim_video_frames)}")

# Menjalankan stitching pipeline pada semua frame
pipeline_stitcher = FastStitcher(n_features=1500, match_threshold=0.75)
pipeline_stitcher.set_reference(sim_video_frames[0])

fps_log = []
match_log = []
inlier_log = []

for i in range(1, min(len(sim_video_frames), 10)):
    _, H_tmp, n_m, n_i, dt_tmp = pipeline_stitcher.match_and_homography(
        sim_video_frames[i], use_cache=True)
    fps = 1.0 / dt_tmp if dt_tmp > 0 else 0
    fps_log.append(fps)
    match_log.append(n_m)
    inlier_log.append(n_i)
    print(f"  Frame {i}: FPS={fps:.1f}, matches={n_m}, inliers={n_i}")

# Menampilkan statistik
print(f"\n  Statistik Pipeline:")
print(f"    Rata-rata FPS:     {np.mean(fps_log):.1f}")
print(f"    Rata-rata matches: {np.mean(match_log):.0f}")
print(f"    Rata-rata inliers: {np.mean(inlier_log):.0f}")
print(f"    {pipeline_stitcher.get_stats_string()}")


# ============================================================
# LANGKAH 8: Homography Caching Test
# ============================================================
print("\n[LANGKAH 8] Menguji efek homography caching...")

# Membandingkan dengan dan tanpa cache
stitcher_no_cache = FastStitcher(n_features=1500, match_threshold=0.75)
stitcher_no_cache.set_reference(sim_frames[0])

stitcher_with_cache = FastStitcher(n_features=1500, match_threshold=0.75)
stitcher_with_cache.set_reference(sim_frames[0])

# Menjalankan beberapa kali untuk mengukur stabilitas
n_runs = 5
fps_no_cache = []
fps_with_cache = []

for run in range(n_runs):
    # Tanpa cache
    _, _, _, _, dt_nc = stitcher_no_cache.match_and_homography(
        sim_frames[1], use_cache=False)
    fps_no_cache.append(1.0 / dt_nc if dt_nc > 0 else 0)

    # Dengan cache
    _, _, _, _, dt_wc = stitcher_with_cache.match_and_homography(
        sim_frames[1], use_cache=True)
    fps_with_cache.append(1.0 / dt_wc if dt_wc > 0 else 0)

print(f"  Tanpa cache:  avg FPS = {np.mean(fps_no_cache):.1f}")
print(f"  Dengan cache: avg FPS = {np.mean(fps_with_cache):.1f}")
print(f"    Cache hits: {stitcher_with_cache.stats['cache_hits']}")


# ============================================================
# LANGKAH 9: Perbandingan ORB vs SIFT Full Pipeline
# ============================================================
print("\n[LANGKAH 9] Membandingkan pipeline lengkap ORB vs SIFT...")


def sift_pipeline_stitch(img_ref, img_new):
    """
    Pipeline stitching menggunakan SIFT + FLANN.
    Untuk perbandingan dengan ORB.
    """
    t_start = time.time()

    # Deteksi SIFT
    sift = cv2.SIFT_create()
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

    kp1, desc1 = sift.detectAndCompute(gray_ref, None)
    kp2, desc2 = sift.detectAndCompute(gray_new, None)

    if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
        return None, 0, 0, time.time() - t_start

    # FLANN matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc2, desc1, k=2)

    # Ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        return None, len(good), 0, time.time() - t_start

    # Homography
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    n_inliers = int(mask.ravel().sum()) if mask is not None else 0

    dt = time.time() - t_start
    return H, len(good), n_inliers, dt


# Pipeline ORB
stitcher_orb = FastStitcher(n_features=1500)
stitcher_orb.set_reference(sim_frames[0])
_, _, orb_matches, orb_inliers, orb_dt = stitcher_orb.match_and_homography(
    sim_frames[1])

# Pipeline SIFT
H_sift, sift_matches, sift_inliers, sift_dt = sift_pipeline_stitch(
    sim_frames[0], sim_frames[1])

print(f"\n  Perbandingan Pipeline Penuh:")
print(f"  {'Metode':<10} | {'Waktu (ms)':>12} | {'Matches':>8} | "
      f"{'Inliers':>8} | {'FPS':>6}")
print(f"  {'-' * 10}-+-{'-' * 12}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}")
orb_fps = 1.0 / orb_dt if orb_dt > 0 else 0
sift_fps = 1.0 / sift_dt if sift_dt > 0 else 0
print(f"  {'ORB':<10} | {orb_dt * 1000:>11.1f} | {orb_matches:>8} | "
      f"{orb_inliers:>8} | {orb_fps:>5.1f}")
print(f"  {'SIFT':<10} | {sift_dt * 1000:>11.1f} | {sift_matches:>8} | "
      f"{sift_inliers:>8} | {sift_fps:>5.1f}")


# ============================================================
# LANGKAH 10: Menyimpan Hasil Panorama Final
# ============================================================
print("\n[LANGKAH 10] Menyimpan hasil panorama final...")

# Stitching terbaik menggunakan SIFT (lebih akurat untuk hasil akhir)
if H_sift is not None:
    h_ref, w_ref = sim_frames[0].shape[:2]
    h_new, w_new = sim_frames[1].shape[:2]

    corners_new = np.float32([[0, 0], [w_new, 0], [w_new, h_new],
                               [0, h_new]]).reshape(-1, 1, 2)
    corners_trans = cv2.perspectiveTransform(corners_new, H_sift)

    all_c = np.concatenate([
        np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref],
                     [0, h_ref]]).reshape(-1, 1, 2),
        corners_trans
    ])

    xm = int(np.floor(all_c[:, 0, 0].min()))
    ym = int(np.floor(all_c[:, 0, 1].min()))
    xmx = int(np.ceil(all_c[:, 0, 0].max()))
    ymx = int(np.ceil(all_c[:, 0, 1].max()))

    cw = min(xmx - xm, 4000)
    ch = min(ymx - ym, 2000)
    H_t = np.array([[1, 0, -xm], [0, 1, -ym], [0, 0, 1]], dtype=np.float64)

    warped = cv2.warpPerspective(sim_frames[1], H_t @ H_sift, (cw, ch))
    result_final = warped.copy()
    ox, oy = -xm, -ym
    y1 = max(0, oy)
    y2 = min(ch, oy + h_ref)
    x1 = max(0, ox)
    x2 = min(cw, ox + w_ref)
    sy1, sx1 = max(0, -oy), max(0, -ox)
    ah, aw = y2 - y1, x2 - x1

    if ah > 0 and aw > 0:
        ref_reg = sim_frames[0][sy1:sy1 + ah, sx1:sx1 + aw]
        mask_r = (cv2.cvtColor(ref_reg, cv2.COLOR_BGR2GRAY) > 0)
        mask_w = (cv2.cvtColor(result_final[y1:y2, x1:x2],
                                cv2.COLOR_BGR2GRAY) > 0)
        ovl = mask_r & mask_w
        only_r = mask_r & ~mask_w
        result_final[y1:y2, x1:x2][ovl] = (
            (ref_reg[ovl].astype(np.float32) +
             result_final[y1:y2, x1:x2][ovl].astype(np.float32)) / 2
        ).astype(np.uint8)
        result_final[y1:y2, x1:x2][only_r] = ref_reg[only_r]

    # Crop
    gf = cv2.cvtColor(result_final, cv2.COLOR_BGR2GRAY)
    _, tf = cv2.threshold(gf, 1, 255, cv2.THRESH_BINARY)
    cf, _ = cv2.findContours(tf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cf:
        lg = max(cf, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(lg)
        result_final_crop = result_final[y:y + h, x:x + w]
    else:
        result_final_crop = result_final

    cv2.imwrite(os.path.join(OUTPUT_DIR, "10_final_panorama.jpg"),
                result_final_crop)
    print(f"  [OK] Panorama final disimpan: "
          f"{result_final_crop.shape[1]}x{result_final_crop.shape[0]}")
else:
    result_final_crop = sim_frames[0]
    print("  [WARNING] Tidak dapat membuat panorama final")


# ============================================================
# LANGKAH 11: Membuat Grid Perbandingan ORB vs SIFT
# ============================================================
print("\n[LANGKAH 11] Membuat grid perbandingan ORB vs SIFT...")

# Visualisasi keypoints ORB dan SIFT
img_orb_kp = sim_frames[0].copy()
img_sift_kp = sim_frames[0].copy()

# ORB keypoints
orb_vis = cv2.ORB_create(nfeatures=500)
kp_orb_vis, _ = orb_vis.detectAndCompute(
    cv2.cvtColor(sim_frames[0], cv2.COLOR_BGR2GRAY), None
)
cv2.drawKeypoints(sim_frames[0], kp_orb_vis, img_orb_kp,
                  color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# SIFT keypoints
sift_vis = cv2.SIFT_create()
kp_sift_vis, _ = sift_vis.detectAndCompute(
    cv2.cvtColor(sim_frames[0], cv2.COLOR_BGR2GRAY), None
)
# Ambil 500 keypoints terkuat untuk visualisasi
kp_sift_vis = sorted(kp_sift_vis, key=lambda k: k.response, reverse=True)[:500]
cv2.drawKeypoints(sim_frames[0], kp_sift_vis, img_sift_kp,
                  color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Grid perbandingan
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))

# Keypoints ORB
axes1[0, 0].imshow(cv2.cvtColor(img_orb_kp, cv2.COLOR_BGR2RGB))
axes1[0, 0].set_title(f"ORB Keypoints ({len(kp_orb_vis)} kp, "
                       f"{t_orb * 1000:.1f}ms)", fontsize=11)
axes1[0, 0].axis("off")

# Keypoints SIFT
axes1[0, 1].imshow(cv2.cvtColor(img_sift_kp, cv2.COLOR_BGR2RGB))
axes1[0, 1].set_title(f"SIFT Keypoints ({len(kp_sift_vis)} kp, "
                       f"{t_sift * 1000:.1f}ms)", fontsize=11)
axes1[0, 1].axis("off")

# Hasil stitching 2-frame
axes1[1, 0].imshow(cv2.cvtColor(result_2_crop, cv2.COLOR_BGR2RGB))
axes1[1, 0].set_title(f"Stitching ORB (2 frame)\n"
                       f"matches={orb_matches}, inliers={orb_inliers}, "
                       f"FPS={orb_fps:.1f}", fontsize=10)
axes1[1, 0].axis("off")

# Hasil final SIFT
axes1[1, 1].imshow(cv2.cvtColor(result_final_crop, cv2.COLOR_BGR2RGB))
axes1[1, 1].set_title(f"Stitching SIFT (2 frame)\n"
                       f"matches={sift_matches}, inliers={sift_inliers}, "
                       f"FPS={sift_fps:.1f}", fontsize=10)
axes1[1, 1].axis("off")

plt.suptitle("Percobaan 10: Perbandingan ORB vs SIFT untuk Real-Time Stitching",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_grid_orb_vs_sift.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid ORB vs SIFT disimpan.")
plt.close()


# ============================================================
# LANGKAH 12: Membuat Grid Performa Pipeline
# ============================================================
print("\n[LANGKAH 12] Membuat grid performa pipeline...")

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# Grafik FPS
if fps_log:
    axes2[0].plot(range(1, len(fps_log) + 1), fps_log, 'go-', linewidth=2,
                  markersize=6)
    axes2[0].axhline(y=np.mean(fps_log), color='r', linestyle='--',
                     label=f"Mean: {np.mean(fps_log):.1f} FPS")
    axes2[0].set_xlabel("Frame ke-", fontsize=10)
    axes2[0].set_ylabel("FPS", fontsize=10)
    axes2[0].set_title("FPS per Frame (ORB Pipeline)", fontsize=12)
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

# Grafik matches per frame
if match_log:
    axes2[1].bar(range(1, len(match_log) + 1), match_log, color='steelblue',
                 alpha=0.7)
    axes2[1].set_xlabel("Frame ke-", fontsize=10)
    axes2[1].set_ylabel("Jumlah Good Matches", fontsize=10)
    axes2[1].set_title("Matches per Frame (ORB)", fontsize=12)
    axes2[1].grid(True, alpha=0.3)

# Perbandingan waktu ORB vs SIFT
methods = ['ORB\n(Deteksi)', 'SIFT\n(Deteksi)', 'ORB\n(Full)', 'SIFT\n(Full)']
times = [t_orb * 1000, t_sift * 1000, orb_dt * 1000, sift_dt * 1000]
colors = ['green', 'blue', 'green', 'blue']
bars = axes2[2].bar(methods, times, color=colors, alpha=0.7)
axes2[2].set_ylabel("Waktu (ms)", fontsize=10)
axes2[2].set_title("Perbandingan Waktu ORB vs SIFT", fontsize=12)
for bar, val in zip(bars, times):
    axes2[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  f"{val:.1f}", ha='center', va='bottom', fontsize=9)
axes2[2].grid(True, alpha=0.3, axis='y')

plt.suptitle("Percobaan 10: Statistik Performa Real-Time Stitching",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_grid_performance_stats.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid performa disimpan.")
plt.close()


# ============================================================
# LANGKAH 13: Grid Hasil Stitching Multi-Frame
# ============================================================
print("\n[LANGKAH 13] Membuat grid hasil stitching multi-frame...")

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))

# Input frames
montage = sim_frames[0].copy()
for fr in sim_frames[1:]:
    h_m = montage.shape[0]
    w_new = int(fr.shape[1] * h_m / fr.shape[0])
    fr_r = cv2.resize(fr, (w_new, h_m))
    montage = np.hstack([montage, fr_r])
axes3[0, 0].imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
axes3[0, 0].set_title(f"Input: {len(sim_frames)} Frame Simulasi", fontsize=11)
axes3[0, 0].axis("off")

# 2-frame result
axes3[0, 1].imshow(cv2.cvtColor(result_2_crop, cv2.COLOR_BGR2RGB))
axes3[0, 1].set_title("Real-Time Stitch: 2 Frame (ORB)", fontsize=11)
axes3[0, 1].axis("off")

# 3-frame result
axes3[1, 0].imshow(cv2.cvtColor(result_123_crop, cv2.COLOR_BGR2RGB))
axes3[1, 0].set_title("Real-Time Stitch: 3 Frame (ORB)", fontsize=11)
axes3[1, 0].axis("off")

# Final panorama
axes3[1, 1].imshow(cv2.cvtColor(result_final_crop, cv2.COLOR_BGR2RGB))
axes3[1, 1].set_title("Final Panorama (SIFT - kualitas terbaik)", fontsize=11)
axes3[1, 1].axis("off")

plt.suptitle("Percobaan 10: Hasil Stitching Real-Time / Interactive",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_grid_multiframe_results.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Grid multi-frame results disimpan.")
plt.close()


# ============================================================
# LANGKAH 14: Ringkasan dan Statistik
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN PERCOBAAN 10: REAL-TIME / INTERACTIVE STITCHING")
print("=" * 65)

# Tabel perbandingan ORB vs SIFT
print("\n  Tabel Perbandingan ORB vs SIFT:")
print(f"  {'Aspek':<20} | {'ORB':>12} | {'SIFT':>12}")
print(f"  {'-' * 20}-+-{'-' * 12}-+-{'-' * 12}")
print(f"  {'Deteksi (ms)':<20} | {t_orb * 1000:>12.1f} | {t_sift * 1000:>12.1f}")
print(f"  {'Full pipeline (ms)':<20} | {orb_dt * 1000:>12.1f} | {sift_dt * 1000:>12.1f}")
print(f"  {'Potential FPS':<20} | {orb_fps:>12.1f} | {sift_fps:>12.1f}")
print(f"  {'Good matches':<20} | {orb_matches:>12} | {sift_matches:>12}")
print(f"  {'Inliers':<20} | {orb_inliers:>12} | {sift_inliers:>12}")
print(f"  {'Deskriptor':<20} | {'Binary':>12} | {'Float':>12}")
print(f"  {'Matcher':<20} | {'BFMatcher':>12} | {'FLANN':>12}")

# Statistik simulasi video
if fps_log:
    print(f"\n  Statistik Simulasi Video Stream:")
    print(f"    Rata-rata FPS:     {np.mean(fps_log):.1f}")
    print(f"    Min FPS:           {min(fps_log):.1f}")
    print(f"    Max FPS:           {max(fps_log):.1f}")
    print(f"    Rata-rata matches: {np.mean(match_log):.0f}")
    print(f"    Rata-rata inliers: {np.mean(inlier_log):.0f}")

# Penjelasan konsep
print("\n  Konsep Real-Time Stitching:")
print("  - ORB ~{:.0f}x lebih cepat dari SIFT untuk deteksi fitur".format(speedup))
print("  - ORB menggunakan deskriptor binary → BFMatcher (Hamming distance)")
print("  - SIFT menggunakan deskriptor float → FLANN (kd-tree)")
print("  - Homography caching meningkatkan stabilitas dan kecepatan")
print("  - Trade-off: ORB cepat tapi kurang akurat, SIFT akurat tapi lambat")
print("  - Untuk real-time (>15 FPS): gunakan ORB")
print("  - Untuk kualitas akhir: gunakan SIFT")

# Daftar output
print("\n  File output yang dihasilkan:")
output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("10_")])
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    - {f} ({size_kb:.1f} KB)")

print("\n  Fungsi utama yang dipelajari:")
print("    cv2.ORB_create()       → Detektor fitur cepat (binary)")
print("    cv2.BFMatcher()        → Brute-force matcher (Hamming)")
print("    cv2.findHomography()   → Estimasi homography (RANSAC)")
print("    cv2.warpPerspective()  → Warping perspektif gambar")
print("    cv2.drawKeypoints()    → Visualisasi keypoints")
print("    time.time()            → Pengukuran FPS dan latency")
print("=" * 65)
