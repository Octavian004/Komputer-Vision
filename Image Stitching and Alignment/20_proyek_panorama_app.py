

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 20: PROYEK - APLIKASI PANORAMA MAKER
    ==========================================================================
    Program ini adalah proyek akhir yang membuat aplikasi panorama maker
    lengkap dengan berbagai mode dan fitur. Mencakup:
    - Auto mode (OpenCV Stitcher)
    - Manual mode (full pipeline dengan konfigurasi)
    - Fast mode (ORB-based untuk kecepatan)

    Aplikasi mendukung berbagai skenario: outdoor, indoor, wide panorama,
    dan document stitching. Dilengkapi dengan quality report, batch
    processing, dan gallery generator.

    Konsep yang dipelajari:
    - Multi-mode stitching (auto, manual, fast)
    - Konfigurasi detektor (SIFT vs ORB)
    - Konfigurasi blending (none, feather, multiband)
    - Batch processing multiple image sets
    - Quality metrics dan timing comparison
    - Gallery generation untuk semua panorama

    Fungsi utama yang dipelajari:
    - cv2.Stitcher_create()     : Stitcher API untuk mode auto
    - cv2.SIFT_create()          : Deteksi fitur SIFT (akurat)
    - cv2.ORB_create()           : Deteksi fitur ORB (cepat)
    - cv2.findHomography()       : Estimasi transformasi perspektif
    - cv2.warpPerspective()      : Warping gambar
    - cv2.pyrDown() / cv2.pyrUp(): Multi-band blending
    - cv2.BFMatcher()            : Brute-force matcher untuk ORB
    - cv2.FlannBasedMatcher()    : FLANN matcher untuk SIFT
    - cv2.imwrite()              : Menyimpan hasil panorama
    ==========================================================================
    """

    # Mengimpor library OpenCV untuk pemrosesan gambar dan computer vision
    import cv2

    # Mengimpor library NumPy untuk operasi array dan matriks numerik
    import numpy as np

    # Mengimpor library os untuk operasi path file dan folder
    import os

    # Mengimpor matplotlib untuk visualisasi grafik dan grid perbandingan
    import matplotlib.pyplot as plt

    # Mengimpor modul time untuk mengukur waktu eksekusi setiap tahap
    import time

    # Mengimpor modul math untuk perhitungan logaritma (PSNR)
    import math

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
    print("PERCOBAAN 20: PROYEK - APLIKASI PANORAMA MAKER")
    print("=" * 65)


    # ============================================================
    # CLASS: PanoramaMaker
    # ============================================================

    class PanoramaMaker:
        """
        Aplikasi Panorama Maker dengan tiga mode operasi:
        1. Auto   : Menggunakan OpenCV Stitcher API
        2. Manual : Pipeline lengkap SIFT + FLANN + multi-band
        3. Fast   : Pipeline cepat ORB + BFMatcher + simple blend

        Setiap mode memiliki kelebihan dan kekurangan:
        - Auto   : Paling mudah, kualitas baik, tapi kurang kontrol
        - Manual : Kontrol penuh, kualitas terbaik, tapi lambat
        - Fast   : Cepat, cocok untuk real-time, kualitas cukup
        """

        def __init__(self, config=None):
            """
            Inisialisasi PanoramaMaker dengan konfigurasi.

            Parameter config:
            - detector      : 'sift' atau 'orb'
            - n_features    : Jumlah fitur maksimum
            - ratio_thresh  : Threshold Lowe's ratio test
            - blend_method  : 'none', 'feather', atau 'multiband'
            - blend_levels  : Jumlah level piramida (untuk multiband)
            - ransac_thresh : Threshold RANSAC
            """
            # Konfigurasi default
            default_config = {
                'detector': 'sift',
                'n_features': 2000,
                'ratio_thresh': 0.75,
                'blend_method': 'multiband',
                'blend_levels': 4,
                'ransac_thresh': 5.0,
            }

            # Menggabungkan konfigurasi user dengan default
            self.config = default_config
            if config:
                self.config.update(config)

            # Menyimpan hasil dan statistik
            self.results = {}
            self.all_timings = {}
            self.all_stats = {}

        def _create_detector(self):
            """Membuat detektor fitur sesuai konfigurasi."""
            if self.config['detector'] == 'sift':
                # SIFT: Scale-Invariant Feature Transform
                # Akurat tetapi relatif lambat
                return cv2.SIFT_create(nfeatures=self.config['n_features'])
            elif self.config['detector'] == 'orb':
                # ORB: Oriented FAST and Rotated BRIEF
                # Cepat tetapi kurang robust terhadap skala besar
                return cv2.ORB_create(nfeatures=self.config['n_features'])
            else:
                return cv2.SIFT_create(nfeatures=self.config['n_features'])

        def _create_matcher(self):
            """Membuat matcher sesuai tipe detektor."""
            if self.config['detector'] == 'sift':
                # FLANN matcher optimal untuk deskriptor float (SIFT)
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=100)
                return cv2.FlannBasedMatcher(index_params, search_params)
            elif self.config['detector'] == 'orb':
                # BFMatcher dengan Hamming distance untuk deskriptor binary (ORB)
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=100)
                return cv2.FlannBasedMatcher(index_params, search_params)

        def _detect_and_match(self, img1, img2):
            """
            Mendeteksi fitur dan mencocokkan antar dua gambar.

            Returns:
            - H         : Matriks homography 3x3
            - n_match   : Jumlah good matches
            - n_inlier  : Jumlah inliers RANSAC
            """
            # Mengkonversi ke grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Membuat detektor dan matcher
            detector = self._create_detector()
            matcher = self._create_matcher()

            # Mendeteksi fitur
            kp1, desc1 = detector.detectAndCompute(gray1, None)
            kp2, desc2 = detector.detectAndCompute(gray2, None)

            # Validasi deskriptor
            if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
                return np.eye(3, dtype=np.float64), 0, 0

            # Konversi deskriptor ORB ke float32 jika menggunakan FLANN
            # (ORB menghasilkan uint8, FLANN butuh float32)
            if self.config['detector'] == 'orb':
                # Untuk BFMatcher, deskriptor binary sudah OK
                matches = matcher.knnMatch(desc1, desc2, k=2)
            else:
                matches = matcher.knnMatch(desc1, desc2, k=2)

            # Ratio test
            good = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.config['ratio_thresh'] * n.distance:
                        good.append(m)

            if len(good) < 10:
                return np.eye(3, dtype=np.float64), len(good), 0

            # Menghitung homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                          self.config['ransac_thresh'])
            n_inlier = int(mask.ravel().sum()) if mask is not None else 0

            if H is None:
                H = np.eye(3, dtype=np.float64)

            return H, len(good), n_inlier

        def stitch_auto(self, images, label=""):
            """
            Mode Auto: Menggunakan OpenCV Stitcher API.
            Stitcher menangani seluruh pipeline secara otomatis.

            Parameter:
            - images : List gambar BGR
            - label  : Label untuk identifikasi

            Returns:
            - result : Panorama hasil, atau None jika gagal
            """
            print(f"\n    [AUTO] Stitching {len(images)} gambar...")
            t0 = time.time()

            try:
                # Membuat Stitcher dengan mode PANORAMA
                stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

                # Menjalankan stitching otomatis
                status, result = stitcher.stitch(images)
                elapsed = time.time() - t0

                # Mengecek status
                if status == cv2.Stitcher_OK:
                    print(f"    [AUTO] Berhasil: {result.shape[1]}x{result.shape[0]}, "
                          f"waktu={elapsed:.3f}s")
                    return result, elapsed
                else:
                    status_msgs = {
                        1: "NEED_MORE_IMGS",
                        2: "HOMOGRAPHY_EST_FAIL",
                        3: "CAMERA_PARAMS_ADJUST_FAIL"
                    }
                    msg = status_msgs.get(status, f"code={status}")
                    print(f"    [AUTO] Gagal: {msg}")
                    return None, elapsed

            except Exception as e:
                elapsed = time.time() - t0
                print(f"    [AUTO] Error: {e}")
                return None, elapsed

        def stitch_manual(self, images, label=""):
            """
            Mode Manual: Full pipeline SIFT/ORB + blending.
            Memberikan kontrol penuh atas setiap tahap.

            Returns:
            - result  : Panorama hasil
            - elapsed : Waktu total
            """
            print(f"\n    [MANUAL] Stitching {len(images)} gambar "
                  f"(detector={self.config['detector']}, blend={self.config['blend_method']})...")
            t0 = time.time()

            n = len(images)
            if n < 2:
                return images[0].copy() if n == 1 else None, 0

            # Tahap 1: Estimasi homography antar pasangan
            H_pairs = []
            total_matches = 0
            total_inliers = 0

            for i in range(n - 1):
                H, n_match, n_inlier = self._detect_and_match(images[i], images[i + 1])
                H_pairs.append(H)
                total_matches += n_match
                total_inliers += n_inlier
                print(f"      Pair {i + 1}-{i + 2}: {n_match} matches, {n_inlier} inliers")

            # Tahap 2: Homography kumulatif dari referensi (tengah)
            ref_idx = n // 2
            H_cum = [None] * n
            H_cum[ref_idx] = np.eye(3, dtype=np.float64)

            for i in range(ref_idx - 1, -1, -1):
                H_cum[i] = H_cum[i + 1] @ H_pairs[i]

            for i in range(ref_idx + 1, n):
                H_inv = np.linalg.inv(H_pairs[i - 1]) if np.linalg.det(H_pairs[i - 1]) != 0 else np.eye(3)
                H_cum[i] = H_cum[i - 1] @ H_inv

            # Tahap 3: Menghitung canvas size
            all_corners = []
            for i in range(n):
                h, w = images[i].shape[:2]
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                corners_t = cv2.perspectiveTransform(corners, H_cum[i])
                all_corners.append(corners_t)

            all_corners = np.concatenate(all_corners, axis=0)
            x_min = int(np.floor(all_corners[:, :, 0].min()))
            y_min = int(np.floor(all_corners[:, :, 1].min()))
            x_max = int(np.ceil(all_corners[:, :, 0].max()))
            y_max = int(np.ceil(all_corners[:, :, 1].max()))

            canvas_w = min(x_max - x_min, 10000)
            canvas_h = min(y_max - y_min, 5000)

            # Memastikan ukuran habis dibagi factor piramida
            factor = 2 ** self.config['blend_levels']
            canvas_w = (canvas_w // factor) * factor
            canvas_h = (canvas_h // factor) * factor

            T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

            # Tahap 4: Warping semua gambar
            warped = []
            masks = []
            for i in range(n):
                w_img = cv2.warpPerspective(images[i], T @ H_cum[i], (canvas_w, canvas_h))
                m_img = (cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
                warped.append(w_img)
                masks.append(m_img)

            # Tahap 5: Blending
            if self.config['blend_method'] == 'none':
                result = self._blend_none(warped, masks)
            elif self.config['blend_method'] == 'feather':
                result = self._blend_feather(warped, masks)
            elif self.config['blend_method'] == 'multiband':
                result = self._blend_multiband(warped, masks)
            else:
                result = self._blend_feather(warped, masks)

            # Tahap 6: Auto-crop
            result = self._auto_crop(result)

            elapsed = time.time() - t0
            if result is not None:
                print(f"    [MANUAL] Berhasil: {result.shape[1]}x{result.shape[0]}, "
                      f"waktu={elapsed:.3f}s")
            print(f"      Stats: {total_matches} matches, {total_inliers} inliers")

            return result, elapsed

        def stitch_fast(self, images, label=""):
            """
            Mode Fast: Pipeline cepat menggunakan ORB + simple feather blend.
            Optimized untuk kecepatan, bukan kualitas maksimal.

            Returns:
            - result  : Panorama hasil
            - elapsed : Waktu total
            """
            print(f"\n    [FAST] Stitching {len(images)} gambar (ORB)...")

            # Menyimpan config asli dan ganti ke ORB
            orig_det = self.config['detector']
            orig_blend = self.config['blend_method']
            orig_feat = self.config['n_features']

            self.config['detector'] = 'orb'
            self.config['blend_method'] = 'feather'
            self.config['n_features'] = 1000  # Kurangi fitur untuk kecepatan

            # Jalankan pipeline manual dengan konfigurasi fast
            result, elapsed = self.stitch_manual(images, label)

            # Kembalikan config asli
            self.config['detector'] = orig_det
            self.config['blend_method'] = orig_blend
            self.config['n_features'] = orig_feat

            return result, elapsed

        def _blend_none(self, warped_list, mask_list):
            """
            Blending tanpa transisi: gambar ditimpa langsung.
            Cepat tetapi menghasilkan seam yang terlihat.
            """
            canvas = np.zeros_like(warped_list[0])
            for w in warped_list:
                mask = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY) > 0
                canvas[mask] = w[mask]
            return canvas

        def _blend_feather(self, warped_list, mask_list):
            """
            Feather blending: transisi gradual menggunakan distance transform.
            Keseimbangan antara kecepatan dan kualitas.
            """
            canvas_h, canvas_w = warped_list[0].shape[:2]
            result = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
            weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float64)

            for i in range(len(warped_list)):
                # Distance transform untuk weight
                dist = cv2.distanceTransform(mask_list[i], cv2.DIST_L2, 5).astype(np.float64)

                for c in range(3):
                    result[:, :, c] += warped_list[i][:, :, c].astype(np.float64) * dist
                weight_sum += dist

            weight_sum = np.maximum(weight_sum, 1e-10)
            for c in range(3):
                result[:, :, c] /= weight_sum

            return np.clip(result, 0, 255).astype(np.uint8)

        def _blend_multiband(self, warped_list, mask_list):
            """
            Multi-band blending: Laplacian pyramid untuk transisi terbaik.
            Paling halus tetapi paling lambat.
            """
            levels = self.config['blend_levels']
            n = len(warped_list)
            canvas_h, canvas_w = warped_list[0].shape[:2]

            # Weight maps
            weight_maps = []
            for m in mask_list:
                dist = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float64)
                weight_maps.append(dist)

            ws = sum(weight_maps)
            ws = np.maximum(ws, 1e-10)
            norm_weights = [w / ws for w in weight_maps]

            def build_gp(img, lvl):
                gp = [img.astype(np.float64)]
                for _ in range(lvl):
                    gp.append(cv2.pyrDown(gp[-1]))
                return gp

            def build_lp(img, lvl):
                gp = build_gp(img, lvl)
                lp = []
                for i in range(lvl):
                    up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
                    lp.append(gp[i] - up)
                lp.append(gp[lvl])
                return lp

            # Build pyramids
            img_pyramids = [build_lp(w, levels) for w in warped_list]
            wgt_pyramids = [build_gp(w, levels) for w in norm_weights]

            # Blend pyramids
            blended = []
            for lvl in range(levels + 1):
                b = np.zeros_like(img_pyramids[0][lvl])
                for i in range(n):
                    w = wgt_pyramids[i][lvl]
                    if len(b.shape) == 3:
                        w = np.stack([w] * 3, axis=-1)
                    b += img_pyramids[i][lvl] * w
                blended.append(b)

            # Reconstruct
            result = blended[levels]
            for i in range(levels - 1, -1, -1):
                result = cv2.pyrUp(result, dstsize=(blended[i].shape[1], blended[i].shape[0]))
                result += blended[i]

            return np.clip(result, 0, 255).astype(np.uint8)

        def _auto_crop(self, image):
            """Auto-crop area hitam dari panorama."""
            if image is None:
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                biggest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(biggest)
                margin = 3
                x = max(0, x + margin)
                y = max(0, y + margin)
                w = min(w - 2 * margin, image.shape[1] - x)
                h = min(h - 2 * margin, image.shape[0] - y)
                return image[y:y + h, x:x + w]

            return image

        def process_set(self, images, set_name=""):
            """
            Memproses satu set gambar dengan ketiga mode.

            Parameter:
            - images   : List gambar BGR
            - set_name : Nama set untuk identifikasi

            Returns:
            - results : Dict dengan hasil dari setiap mode
            """
            print(f"\n{'=' * 55}")
            print(f"  PROCESSING: {set_name} ({len(images)} gambar)")
            print(f"{'=' * 55}")

            results = {}

            # Mode 1: Auto (OpenCV Stitcher)
            result_auto, time_auto = self.stitch_auto(images, set_name)
            results['Auto'] = {'result': result_auto, 'time': time_auto}

            # Mode 2: Manual (SIFT + multiband)
            result_manual, time_manual = self.stitch_manual(images, set_name)
            results['Manual'] = {'result': result_manual, 'time': time_manual}

            # Mode 3: Fast (ORB + feather)
            result_fast, time_fast = self.stitch_fast(images, set_name)
            results['Fast'] = {'result': result_fast, 'time': time_fast}

            # Menyimpan results
            self.results[set_name] = results

            return results

        def generate_quality_report(self, set_name, reference_image=None):
            """
            Menghasilkan quality report untuk satu set.

            Returns:
            - report : Dict berisi metrik per mode
            """
            if set_name not in self.results:
                return {}

            report = {}
            results = self.results[set_name]

            for mode, data in results.items():
                pano = data['result']
                if pano is None:
                    report[mode] = {'status': 'GAGAL'}
                    continue

                metrics = {
                    'status': 'OK',
                    'width': pano.shape[1],
                    'height': pano.shape[0],
                    'pixels': pano.shape[0] * pano.shape[1],
                    'time': data['time']
                }

                # Menghitung statistik gambar
                gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
                metrics['mean_brightness'] = float(np.mean(gray))
                metrics['std_brightness'] = float(np.std(gray))
                metrics['min_val'] = int(np.min(gray))
                metrics['max_val'] = int(np.max(gray))

                # Menghitung entropy sebagai ukuran detail/informasi
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = hist.flatten() / hist.sum()
                hist = hist[hist > 0]
                metrics['entropy'] = float(-np.sum(hist * np.log2(hist)))

                report[mode] = metrics

            return report


    # ============================================================
    # LANGKAH 1: Memuat Semua Set Gambar
    # ============================================================
    print("\n[LANGKAH 1] Memuat semua set gambar yang tersedia...")

    # Mendefinisikan semua set gambar
    all_image_sets = {
        "Outdoor (3 img)": [f"panorama_outdoor_{i}.jpg" for i in range(1, 4)],
        "Indoor (4 img)": [f"panorama_indoor_{i}.jpg" for i in range(1, 5)],
        "Wide (5 img)": [f"panorama_wide_{i}.jpg" for i in range(1, 6)],
        "Document": ["dokumen_1.jpg", "dokumen_2.jpg", "dokumen_3.jpg"],
    }

    loaded_sets = {}

    for set_name, files in all_image_sets.items():
        images = []
        for f in files:
            img = cv2.imread(os.path.join(IMAGE_DIR, f))
            if img is not None:
                images.append(img)

        if len(images) >= 2:
            loaded_sets[set_name] = images
            print(f"  {set_name}: {len(images)} gambar dimuat")
        else:
            print(f"  {set_name}: SKIP (hanya {len(images)} gambar)")

    if not loaded_sets:
        print("[ERROR] Tidak ada set gambar yang berhasil dimuat!")
        print("Jalankan download_image.py terlebih dahulu.")
        exit()

    print(f"\n  Total set yang tersedia: {len(loaded_sets)}")


    # ============================================================
    # LANGKAH 2: Membuat PanoramaMaker dan Batch Processing
    # ============================================================
    print("\n[LANGKAH 2] Batch processing semua set gambar...")

    # Membuat instance PanoramaMaker dengan konfigurasi default
    maker = PanoramaMaker(config={
        'detector': 'sift',
        'n_features': 2000,
        'ratio_thresh': 0.75,
        'blend_method': 'multiband',
        'blend_levels': 4,
        'ransac_thresh': 5.0,
    })

    # Memproses setiap set gambar
    all_reports = {}

    for set_name, images in loaded_sets.items():
        try:
            # Memproses dengan ketiga mode
            results = maker.process_set(images, set_name)

            # Menyimpan hasil ke file
            for mode, data in results.items():
                if data['result'] is not None:
                    safe_set = set_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                    safe_mode = mode.lower()
                    filename = f"20_{safe_set}_{safe_mode}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), data['result'])
                    print(f"    Saved: {filename}")

            # Generate quality report
            report = maker.generate_quality_report(set_name)
            all_reports[set_name] = report

        except Exception as e:
            print(f"  [ERROR] Gagal memproses {set_name}: {e}")


    # ============================================================
    # LANGKAH 3: Quality Report
    # ============================================================
    print("\n[LANGKAH 3] Quality Report")
    print("=" * 90)
    print(f"{'Set':<20} {'Mode':<8} {'Status':<8} {'Ukuran':<15} "
          f"{'Waktu(s)':<10} {'Brightness':<12} {'Entropy':<10}")
    print("-" * 90)

    for set_name, report in all_reports.items():
        for mode, metrics in report.items():
            if metrics.get('status') == 'OK':
                size = f"{metrics['width']}x{metrics['height']}"
                print(f"{set_name:<20} {mode:<8} {'OK':<8} {size:<15} "
                      f"{metrics['time']:>7.3f}   "
                      f"{metrics['mean_brightness']:>8.1f}    "
                      f"{metrics['entropy']:>7.3f}")
            else:
                print(f"{set_name:<20} {mode:<8} {'GAGAL':<8}")

    print("-" * 90)


    # ============================================================
    # LANGKAH 4: Timing Comparison per Set
    # ============================================================
    print("\n[LANGKAH 4] Membuat timing comparison...")

    try:
        n_sets = len(all_reports)
        fig, axes = plt.subplots(1, n_sets, figsize=(7 * n_sets, 5))
        if n_sets == 1:
            axes = [axes]

        colors_mode = {'Auto': '#3498db', 'Manual': '#e74c3c', 'Fast': '#2ecc71'}

        for idx, (set_name, report) in enumerate(all_reports.items()):
            modes = []
            times = []
            clrs = []

            for mode, metrics in report.items():
                if metrics.get('status') == 'OK':
                    modes.append(mode)
                    times.append(metrics['time'])
                    clrs.append(colors_mode.get(mode, '#95a5a6'))

            if modes:
                bars = axes[idx].bar(modes, times, color=clrs, edgecolor='black')
                for bar, t in zip(bars, times):
                    axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                                   f'{t:.3f}s', ha='center', fontsize=10)

            axes[idx].set_title(f"{set_name}", fontsize=11)
            axes[idx].set_ylabel("Waktu (detik)")

        plt.suptitle("Perbandingan Waktu: Auto vs Manual vs Fast",
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "20_timing_comparison.png"),
                dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Timing comparison disimpan.")
    except Exception as e:
        print(f"  [WARNING] Gagal membuat timing comparison: {e}")


    # ============================================================
    # LANGKAH 5: Per-Set Comparison Grid
    # ============================================================
    print("\n[LANGKAH 5] Membuat comparison grid untuk setiap set...")

    for set_name in maker.results:
        try:
            results = maker.results[set_name]
            valid = {k: v for k, v in results.items() if v['result'] is not None}
            n = len(valid)

            if n == 0:
                continue

            fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
            if n == 1:
                axes = [axes]

            for i, (mode, data) in enumerate(valid.items()):
                pano_rgb = cv2.cvtColor(data['result'], cv2.COLOR_BGR2RGB)
                axes[i].imshow(pano_rgb)
                h, w = data['result'].shape[:2]
                axes[i].set_title(f"{mode}\n{w}x{h} | {data['time']:.3f}s", fontsize=11)
                axes[i].axis('off')

            plt.suptitle(f"Perbandingan Mode: {set_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()

            safe_set = set_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(
                os.path.join(OUTPUT_DIR, f"20_comparison_{safe_set}.png"),
                dpi=150,
                bbox_inches='tight'
            )
            plt.show()
            plt.close()
            print(f"  Comparison grid untuk {set_name} disimpan.")
        except Exception as e:
            print(f"  [WARNING] Gagal membuat comparison grid: {e}")


    # ============================================================
    # LANGKAH 6: Gallery of All Panoramas
    # ============================================================
    print("\n[LANGKAH 6] Membuat gallery semua panorama...")

    try:
        # Mengumpulkan semua panorama yang berhasil
        all_panoramas = []
        for set_name, results in maker.results.items():
            for mode, data in results.items():
                if data['result'] is not None:
                    all_panoramas.append({
                        'name': f"{set_name} - {mode}",
                        'image': data['result'],
                        'time': data['time']
                    })

        n_panos = len(all_panoramas)
        print(f"  Total panorama berhasil: {n_panos}")

        if n_panos > 0:
            # Menghitung layout grid
            cols = min(3, n_panos)
            rows = (n_panos + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows))
            axes = np.array(axes).flatten() if n_panos > 1 else [axes]

            for i in range(len(axes)):
                if i < n_panos:
                    pano_rgb = cv2.cvtColor(all_panoramas[i]['image'], cv2.COLOR_BGR2RGB)
                    axes[i].imshow(pano_rgb)
                    h, w = all_panoramas[i]['image'].shape[:2]
                    axes[i].set_title(
                        f"{all_panoramas[i]['name']}\n{w}x{h} | {all_panoramas[i]['time']:.3f}s",
                        fontsize=10
                    )
                axes[i].axis('off')

            plt.suptitle("Gallery Semua Panorama (Percobaan 20: Proyek Panorama Maker)",
                          fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "20_gallery_panorama.png"),
                        dpi=120, bbox_inches='tight')
            plt.show()
            plt.close()
            print("  Gallery panorama disimpan.")
    except Exception as e:
        print(f"  [WARNING] Gagal membuat gallery: {e}")


    # ============================================================
    # LANGKAH 7: Analisis Detail per Mode
    # ============================================================
    print("\n[LANGKAH 7] Analisis detail per mode...")

    try:
        # Mengumpulkan data per mode
        mode_data = {'Auto': [], 'Manual': [], 'Fast': []}

        for set_name, report in all_reports.items():
            for mode, metrics in report.items():
                if metrics.get('status') == 'OK' and mode in mode_data:
                    mode_data[mode].append(metrics)

        # Statistik per mode
        print(f"\n{'Mode':<10} {'Avg Time(s)':<14} {'Avg Brightness':<18} "
              f"{'Avg Entropy':<14} {'Success Rate':<14}")
        print("-" * 70)

        for mode in ['Auto', 'Manual', 'Fast']:
            data_list = mode_data[mode]
            total_attempts = sum(1 for r in all_reports.values() if mode in r)
            success = len(data_list)

            if success > 0:
                avg_time = np.mean([d['time'] for d in data_list])
                avg_bright = np.mean([d['mean_brightness'] for d in data_list])
                avg_entropy = np.mean([d['entropy'] for d in data_list])
                rate = f"{success}/{total_attempts}"
            else:
                avg_time = 0
                avg_bright = 0
                avg_entropy = 0
                rate = f"0/{total_attempts}"

            print(f"{mode:<10} {avg_time:>10.3f}     {avg_bright:>12.1f}      "
                  f"{avg_entropy:>10.3f}     {rate:<14}")

        print("-" * 70)
    except Exception as e:
        print(f"  [WARNING] Gagal membuat analisis: {e}")


    # ============================================================
    # LANGKAH 8: Mode Recommendation Engine
    # ============================================================
    print("\n[LANGKAH 8] Rekomendasi mode untuk setiap skenario...")

    for set_name, report in all_reports.items():
        print(f"\n  {set_name}:")

        # Menentukan rekomendasi berdasarkan metrik
        best_quality_mode = None
        best_quality_entropy = 0
        best_speed_mode = None
        best_speed_time = float('inf')

        for mode, metrics in report.items():
            if metrics.get('status') != 'OK':
                continue

            # Mode dengan entropy tertinggi = detail terbaik
            if metrics['entropy'] > best_quality_entropy:
                best_quality_entropy = metrics['entropy']
                best_quality_mode = mode

            # Mode tercepat
            if metrics['time'] < best_speed_time:
                best_speed_time = metrics['time']
                best_speed_mode = mode

        if best_quality_mode:
            print(f"    Kualitas terbaik : {best_quality_mode} "
                  f"(entropy={best_quality_entropy:.3f})")
        if best_speed_mode:
            print(f"    Tercepat         : {best_speed_mode} "
                  f"(waktu={best_speed_time:.3f}s)")

        # Rekomendasi
        n_images = len(loaded_sets.get(set_name, []))
        if n_images <= 3:
            print(f"    Rekomendasi      : Manual (gambar sedikit, prioritas kualitas)")
        elif n_images <= 5:
            print(f"    Rekomendasi      : Auto (keseimbangan kecepatan & kualitas)")
        else:
            print(f"    Rekomendasi      : Fast (gambar banyak, prioritas kecepatan)")


    # ============================================================
    # LANGKAH 9: Comprehensive Summary Figure
    # ============================================================
    print("\n[LANGKAH 9] Membuat comprehensive summary figure...")

    try:
        fig = plt.figure(figsize=(24, 20))

        # Bagian 1: Timing comparison bar chart
        ax_time = fig.add_subplot(3, 2, 1)
        x_labels = []
        time_auto = []
        time_manual = []
        time_fast = []

        for set_name, report in all_reports.items():
            x_labels.append(set_name.replace(" ", "\n"))
            time_auto.append(report.get('Auto', {}).get('time', 0))
            time_manual.append(report.get('Manual', {}).get('time', 0))
            time_fast.append(report.get('Fast', {}).get('time', 0))

        x = np.arange(len(x_labels))
        width = 0.25
        ax_time.bar(x - width, time_auto, width, label='Auto', color='#3498db', edgecolor='black')
        ax_time.bar(x, time_manual, width, label='Manual', color='#e74c3c', edgecolor='black')
        ax_time.bar(x + width, time_fast, width, label='Fast', color='#2ecc71', edgecolor='black')
        ax_time.set_xticks(x)
        ax_time.set_xticklabels(x_labels, fontsize=8)
        ax_time.set_ylabel("Waktu (detik)")
        ax_time.set_title("Perbandingan Waktu Eksekusi", fontsize=12)
        ax_time.legend(fontsize=9)

        # Bagian 2: Entropy comparison
        ax_ent = fig.add_subplot(3, 2, 2)
        ent_auto = []
        ent_manual = []
        ent_fast = []

        for set_name, report in all_reports.items():
            ent_auto.append(report.get('Auto', {}).get('entropy', 0))
            ent_manual.append(report.get('Manual', {}).get('entropy', 0))
            ent_fast.append(report.get('Fast', {}).get('entropy', 0))

        ax_ent.bar(x - width, ent_auto, width, label='Auto', color='#3498db', edgecolor='black')
        ax_ent.bar(x, ent_manual, width, label='Manual', color='#e74c3c', edgecolor='black')
        ax_ent.bar(x + width, ent_fast, width, label='Fast', color='#2ecc71', edgecolor='black')
        ax_ent.set_xticks(x)
        ax_ent.set_xticklabels(x_labels, fontsize=8)
        ax_ent.set_ylabel("Entropy")
        ax_ent.set_title("Perbandingan Image Entropy (Detail)", fontsize=12)
        ax_ent.legend(fontsize=9)

        # Bagian 3-6: Best panorama per set
        for idx, (set_name, results) in enumerate(maker.results.items()):
            if idx >= 4:
                break
            ax = fig.add_subplot(3, 2, 3 + idx)

            # Pilih panorama terbaik (Manual jika ada, kalau tidak Auto)
            best = None
            best_mode = ""
            for mode in ['Manual', 'Auto', 'Fast']:
                if mode in results and results[mode]['result'] is not None:
                    best = results[mode]['result']
                    best_mode = mode
                    break

            if best is not None:
                ax.imshow(cv2.cvtColor(best, cv2.COLOR_BGR2RGB))
                h, w = best.shape[:2]
                t = results[best_mode]['time']
                ax.set_title(f"{set_name}\n({best_mode}) {w}x{h} | {t:.3f}s", fontsize=10)
            else:
                ax.set_title(f"{set_name}\n(Tidak tersedia)", fontsize=10)
            ax.axis('off')

        plt.suptitle("Comprehensive Summary: Panorama Maker (Percobaan 20)",
                      fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "20_comprehensive_summary.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Comprehensive summary disimpan.")
    except Exception as e:
        print(f"  [WARNING] Gagal membuat summary: {e}")


    # ============================================================
    # LANGKAH 10: Final Summary
    # ============================================================
    print("\n" + "=" * 65)
    print("RINGKASAN PERCOBAAN 20: PROYEK PANORAMA MAKER")
    print("=" * 65)

    # Hitung total statistik
    total_panoramas = sum(
        1 for results in maker.results.values()
        for data in results.values()
        if data['result'] is not None
    )

    total_failed = sum(
        1 for results in maker.results.values()
        for data in results.values()
        if data['result'] is None
    )

    print(f"""
    Statistik Keseluruhan:
    - Set gambar diproses : {len(loaded_sets)}
    - Total panorama berhasil : {total_panoramas}
    - Total panorama gagal    : {total_failed}
    - Mode yang diuji         : Auto, Manual (SIFT), Fast (ORB)

    Perbandingan Mode:
    ┌──────────┬─────────────────────────────────────────────────────┐
    │  Auto    │ OpenCV Stitcher API. Mudah digunakan, kualitas     │
    │          │ umumnya baik. Kurang kontrol atas parameter.        │
    ├──────────┼─────────────────────────────────────────────────────┤
    │  Manual  │ SIFT + FLANN + multi-band blend. Kontrol penuh,    │
    │          │ kualitas terbaik. Lebih lambat dari Auto.           │
    ├──────────┼─────────────────────────────────────────────────────┤
    │  Fast    │ ORB + BFMatcher + feather blend. Paling cepat,     │
    │          │ cocok untuk real-time. Kualitas cukup baik.         │
    └──────────┴─────────────────────────────────────────────────────┘

    Teknik yang Dipelajari:
    1. cv2.Stitcher_create() - Stitching otomatis (mode PANORAMA)
    2. cv2.SIFT_create()     - Deteksi fitur yang robust dan akurat
    3. cv2.ORB_create()      - Deteksi fitur cepat (binary descriptors)
    4. cv2.FlannBasedMatcher - Matching cepat untuk deskriptor float
    5. cv2.BFMatcher         - Matching untuk deskriptor binary (ORB)
    6. cv2.findHomography    - Estimasi transformasi perspektif (RANSAC)
    7. cv2.warpPerspective   - Warping gambar ke canvas panorama
    8. cv2.pyrDown/pyrUp     - Multi-band Laplacian pyramid blending
    9. cv2.distanceTransform - Weight map untuk feather blending
    10. cv2.imwrite          - Menyimpan hasil dalam berbagai format

    Rekomendasi Penggunaan:
    - Gambar sedikit (2-3)     → Manual mode (prioritas kualitas)
    - Gambar sedang (3-5)      → Auto mode (keseimbangan)
    - Gambar banyak (5+)       → Fast mode (prioritas kecepatan)
    - Real-time / mobile       → Fast mode + resolusi rendah
    - Cetak / publikasi        → Manual mode + resolusi tinggi
    """)

    # Menampilkan daftar file output
    print("File output yang dihasilkan:")
    try:
        output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("20_")])
        for f in output_files:
            path = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(path) / 1024
            print(f"  {f:<50} ({size_kb:>8.1f} KB)")
    except Exception as e:
        print(f"  [WARNING] Gagal listing file: {e}")

    print("\n" + "=" * 65)
    print("Program selesai dijalankan.")
    print("=" * 65)



if __name__ == "__main__":
    main()
