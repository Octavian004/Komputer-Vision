

def main():
    """Fungsi utama yang menjalankan seluruh percobaan."""
    """
    ==========================================================================
    PERCOBAAN 19: PANORAMA PIPELINE LENGKAP
    ==========================================================================
    Program ini mengintegrasikan seluruh teknik yang dipelajari menjadi satu
    pipeline panorama profesional yang lengkap:
      detection → matching → homography → compensation → warp → seam → blend → crop

    Pipeline ini diimplementasikan sebagai class PanoramaPipeline agar
    modular, mudah diuji, dan bisa digunakan ulang.

    Konsep yang dipelajari:
    - Class-based pipeline design untuk panorama
    - Modularisasi setiap tahap stitching
    - Profiling waktu eksekusi per tahap
    - Exposure compensation sederhana
    - Seam finding menggunakan difference map
    - Multi-band blending dengan Laplacian pyramid
    - Auto-cropping area hitam pada hasil panorama
    - Perbandingan dengan OpenCV Stitcher API

    Fungsi utama yang dipelajari:
    - cv2.SIFT_create()         : Deteksi fitur SIFT
    - cv2.FlannBasedMatcher()   : Feature matching cepat
    - cv2.findHomography()      : Estimasi homography dengan RANSAC
    - cv2.warpPerspective()     : Image warping perspektif
    - cv2.pyrDown() / cv2.pyrUp()  : Piramida Gaussian untuk multi-band blend
    - cv2.Stitcher_create()     : OpenCV Stitcher API untuk referensi
    - cv2.distanceTransform()   : Menghitung jarak ke tepi mask
    - cv2.createCLAHE()         : Exposure compensation adaptif
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
    print("PERCOBAAN 19: PANORAMA PIPELINE LENGKAP")
    print("=" * 65)


    # ============================================================
    # CLASS: PanoramaPipeline
    # ============================================================

    class PanoramaPipeline:
        """
        Pipeline panorama stitching lengkap dengan tahapan:
        1. detect_features()        - Deteksi fitur SIFT pada semua gambar
        2. match_features()         - Pencocokan fitur antar pasangan bersebelahan
        3. estimate_homographies()  - Estimasi homography kumulatif
        4. compute_canvas_size()    - Menghitung ukuran canvas akhir
        5. warp_images()            - Warping semua gambar ke canvas
        6. exposure_compensate()    - Kompensasi exposure
        7. find_seam()              - Menemukan posisi seam optimal
        8. blend_multiband()        - Multi-band blending dengan Laplacian pyramid
        9. crop_result()            - Auto-cropping area hitam

        Setiap tahap mencatat waktu eksekusi untuk profiling.
        """

        def __init__(self, n_features=2000, ratio_thresh=0.75, ransac_thresh=5.0,
                     blend_levels=4):
            """
            Inisialisasi pipeline dengan parameter yang bisa dikonfigurasi.

            Parameter:
            - n_features     : Jumlah fitur SIFT maksimum per gambar
            - ratio_thresh   : Threshold untuk Lowe's ratio test
            - ransac_thresh  : Threshold RANSAC untuk homography (piksel)
            - blend_levels   : Jumlah level piramida untuk multi-band blend
            """
            # Menyimpan parameter konfigurasi
            self.n_features = n_features
            self.ratio_thresh = ratio_thresh
            self.ransac_thresh = ransac_thresh
            self.blend_levels = blend_levels

            # Menyimpan data intermediate untuk setiap tahap
            self.images = []             # Gambar input
            self.keypoints = []          # Keypoints per gambar
            self.descriptors = []        # Deskriptor per gambar
            self.pair_matches = []       # Match antar pasangan
            self.homographies = []       # Homography kumulatif
            self.canvas_size = (0, 0)    # Ukuran canvas (w, h)
            self.translation = None      # Matriks translasi
            self.warped_images = []      # Gambar yang sudah di-warp
            self.warped_masks = []       # Mask gambar yang di-warp
            self.compensated = []        # Gambar setelah exposure compensation
            self.seam_masks = []         # Mask seam per gambar
            self.blended = None          # Hasil blending
            self.cropped = None          # Hasil setelah cropping

            # Waktu eksekusi per tahap
            self.timings = {}

            # Statistik
            self.stats = {}

        def detect_features(self):
            """
            Tahap 1: Mendeteksi fitur SIFT pada semua gambar input.

            SIFT (Scale-Invariant Feature Transform) mendeteksi keypoints
            yang invariant terhadap perubahan skala, rotasi, dan pencahayaan.
            Setiap keypoint memiliki deskriptor 128-dimensi.
            """
            print("\n  [Tahap 1] Deteksi fitur SIFT...")
            t0 = time.time()

            # Membuat detektor SIFT dengan jumlah fitur maksimum
            sift = cv2.SIFT_create(nfeatures=self.n_features)

            self.keypoints = []
            self.descriptors = []

            for i, img in enumerate(self.images):
                # Mengkonversi ke grayscale untuk deteksi fitur
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Mendeteksi keypoints dan menghitung deskriptor 128-D
                kp, desc = sift.detectAndCompute(gray, None)

                self.keypoints.append(kp)
                self.descriptors.append(desc)

                print(f"    Gambar {i + 1}: {len(kp)} keypoints terdeteksi")

            # Mencatat waktu dan statistik
            self.timings['detect'] = time.time() - t0
            self.stats['total_keypoints'] = sum(len(kp) for kp in self.keypoints)

            print(f"    Total keypoints: {self.stats['total_keypoints']}")
            print(f"    Waktu: {self.timings['detect']:.3f} detik")

        def match_features(self):
            """
            Tahap 2: Mencocokkan fitur antar pasangan gambar bersebelahan.

            Menggunakan FLANN matcher (Fast Library for Approximate Nearest
            Neighbors) untuk pencocokan cepat, diikuti Lowe's ratio test
            untuk memfilter match yang reliable.
            """
            print("\n  [Tahap 2] Pencocokan fitur antar pasangan...")
            t0 = time.time()

            # Konfigurasi FLANN matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)

            self.pair_matches = []
            n = len(self.images)

            total_matches = 0
            total_inliers = 0

            for i in range(n - 1):
                desc1 = self.descriptors[i]
                desc2 = self.descriptors[i + 1]

                # Validasi deskriptor
                if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
                    self.pair_matches.append({'good': [], 'H': np.eye(3), 'inliers': 0})
                    print(f"    Pasangan {i + 1}-{i + 2}: Skip (deskriptor kosong)")
                    continue

                # Membuat FLANN matcher untuk pasangan ini
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                # KNN matching (k=2 untuk ratio test)
                matches = flann.knnMatch(desc1, desc2, k=2)

                # Lowe's ratio test
                good = [m for m, n_match in matches if m.distance < self.ratio_thresh * n_match.distance]

                # Menghitung homography jika cukup matches
                if len(good) >= 10:
                    src_pts = np.float32(
                        [self.keypoints[i][m.queryIdx].pt for m in good]
                    ).reshape(-1, 1, 2)
                    dst_pts = np.float32(
                        [self.keypoints[i + 1][m.trainIdx].pt for m in good]
                    ).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                                  self.ransac_thresh)
                    n_inlier = int(mask.ravel().sum()) if mask is not None else 0
                else:
                    H = np.eye(3, dtype=np.float64)
                    n_inlier = 0

                if H is None:
                    H = np.eye(3, dtype=np.float64)

                self.pair_matches.append({
                    'good': good,
                    'H': H,
                    'inliers': n_inlier
                })

                total_matches += len(good)
                total_inliers += n_inlier

                print(f"    Pasangan {i + 1}-{i + 2}: {len(good)} matches, "
                      f"{n_inlier} inliers")

            # Mencatat statistik
            self.timings['match'] = time.time() - t0
            self.stats['total_matches'] = total_matches
            self.stats['total_inliers'] = total_inliers

            print(f"    Total matches: {total_matches}, Total inliers: {total_inliers}")
            print(f"    Waktu: {self.timings['match']:.3f} detik")

        def estimate_homographies(self):
            """
            Tahap 3: Menghitung homography kumulatif dari setiap gambar ke referensi.

            Gambar tengah dipilih sebagai referensi untuk meminimalkan distorsi.
            Homography dihitung secara berantai (chain multiplication).
            """
            print("\n  [Tahap 3] Estimasi homography kumulatif...")
            t0 = time.time()

            n = len(self.images)

            # Memilih gambar tengah sebagai referensi
            ref_idx = n // 2
            self.stats['ref_idx'] = ref_idx
            print(f"    Referensi: Gambar {ref_idx + 1} (tengah)")

            # Inisialisasi homography kumulatif
            self.homographies = [None] * n
            self.homographies[ref_idx] = np.eye(3, dtype=np.float64)

            # Chain ke kiri: ref-1, ref-2, ...
            for i in range(ref_idx - 1, -1, -1):
                H_pair = self.pair_matches[i]['H']
                self.homographies[i] = self.homographies[i + 1] @ H_pair

            # Chain ke kanan: ref+1, ref+2, ...
            for i in range(ref_idx + 1, n):
                H_pair = self.pair_matches[i - 1]['H']
                H_inv = np.linalg.inv(H_pair) if np.linalg.det(H_pair) != 0 else np.eye(3)
                self.homographies[i] = self.homographies[i - 1] @ H_inv

            # Menampilkan homography
            for i, H in enumerate(self.homographies):
                if H is not None:
                    tx = H[0, 2]
                    ty = H[1, 2]
                    print(f"    Gambar {i + 1}: tx={tx:>8.1f}, ty={ty:>8.1f}")

            self.timings['homography'] = time.time() - t0
            print(f"    Waktu: {self.timings['homography']:.3f} detik")

        def compute_canvas_size(self):
            """
            Tahap 4: Menghitung ukuran canvas akhir berdasarkan
            transformasi semua gambar.
            """
            print("\n  [Tahap 4] Menghitung ukuran canvas...")
            t0 = time.time()

            # Mengumpulkan semua corner setelah transformasi
            all_corners = []
            for i, img in enumerate(self.images):
                h, w = img.shape[:2]
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                if self.homographies[i] is not None:
                    corners_t = cv2.perspectiveTransform(corners, self.homographies[i])
                    all_corners.append(corners_t)

            all_corners = np.concatenate(all_corners, axis=0)

            # Menentukan batas canvas
            x_min = int(np.floor(all_corners[:, :, 0].min()))
            y_min = int(np.floor(all_corners[:, :, 1].min()))
            x_max = int(np.ceil(all_corners[:, :, 0].max()))
            y_max = int(np.ceil(all_corners[:, :, 1].max()))

            # Membatasi ukuran canvas
            canvas_w = min(x_max - x_min, 10000)
            canvas_h = min(y_max - y_min, 5000)

            # Memastikan ukuran habis dibagi 2^levels
            factor = 2 ** self.blend_levels
            canvas_w = (canvas_w // factor) * factor
            canvas_h = (canvas_h // factor) * factor

            self.canvas_size = (canvas_w, canvas_h)

            # Matriks translasi untuk menggeser koordinat negatif
            self.translation = np.array([
                [1, 0, -x_min],
                [0, 1, -y_min],
                [0, 0, 1]
            ], dtype=np.float64)

            self.timings['canvas'] = time.time() - t0
            print(f"    Canvas: {canvas_w}x{canvas_h}")
            print(f"    Offset: ({-x_min}, {-y_min})")
            print(f"    Waktu: {self.timings['canvas']:.3f} detik")

        def warp_images(self):
            """
            Tahap 5: Warping semua gambar ke canvas menggunakan homography.
            Juga membuat mask untuk setiap gambar yang di-warp.
            """
            print("\n  [Tahap 5] Warping gambar ke canvas...")
            t0 = time.time()

            canvas_w, canvas_h = self.canvas_size
            self.warped_images = []
            self.warped_masks = []

            for i, img in enumerate(self.images):
                if self.homographies[i] is None:
                    self.warped_images.append(np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8))
                    self.warped_masks.append(np.zeros((canvas_h, canvas_w), dtype=np.uint8))
                    continue

                # Warping gambar menggunakan perspektif transform + translasi
                warped = cv2.warpPerspective(img, self.translation @ self.homographies[i],
                                              (canvas_w, canvas_h))

                # Membuat mask (area valid = piksel > 0)
                mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255

                self.warped_images.append(warped)
                self.warped_masks.append(mask)

                print(f"    Gambar {i + 1}: warped ke canvas {canvas_w}x{canvas_h}")

            self.timings['warp'] = time.time() - t0
            print(f"    Waktu: {self.timings['warp']:.3f} detik")

        def exposure_compensate(self):
            """
            Tahap 6: Kompensasi exposure antar gambar.
            Menggunakan gain normalization pada area overlap.
            """
            print("\n  [Tahap 6] Kompensasi exposure...")
            t0 = time.time()

            n = len(self.warped_images)
            self.compensated = [w.copy() for w in self.warped_images]

            # Menghitung rata-rata brightness untuk setiap gambar di area valid
            brightness = []
            for i in range(n):
                mask = self.warped_masks[i]
                if np.sum(mask > 0) > 0:
                    # Mengkonversi ke LAB space (L channel = brightness)
                    lab = cv2.cvtColor(self.compensated[i], cv2.COLOR_BGR2LAB)
                    l_channel = lab[:, :, 0].astype(np.float64)
                    # Rata-rata brightness hanya di area valid
                    mean_bright = np.mean(l_channel[mask > 0])
                    brightness.append(mean_bright)
                else:
                    brightness.append(128.0)

            # Menghitung target brightness (rata-rata global)
            target_brightness = np.mean(brightness) if brightness else 128.0

            # Menerapkan gain compensation ke setiap gambar
            for i in range(n):
                if brightness[i] > 0:
                    # Menghitung gain factor
                    gain = target_brightness / brightness[i]
                    gain = np.clip(gain, 0.5, 2.0)  # Membatasi range gain

                    # Menerapkan gain pada LAB space (hanya L channel)
                    lab = cv2.cvtColor(self.compensated[i], cv2.COLOR_BGR2LAB).astype(np.float64)
                    lab[:, :, 0] *= gain
                    lab = np.clip(lab, 0, 255).astype(np.uint8)
                    self.compensated[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                    print(f"    Gambar {i + 1}: brightness={brightness[i]:.1f} → "
                          f"gain={gain:.3f}")

            self.timings['compensate'] = time.time() - t0
            print(f"    Target brightness: {target_brightness:.1f}")
            print(f"    Waktu: {self.timings['compensate']:.3f} detik")

        def find_seam(self):
            """
            Tahap 7: Menemukan posisi seam optimal menggunakan distance transform.
            Seam ditempatkan di area dimana perbedaan antar gambar minimal.
            """
            print("\n  [Tahap 7] Finding optimal seam...")
            t0 = time.time()

            n = len(self.compensated)
            canvas_w, canvas_h = self.canvas_size

            # Membuat weight map menggunakan distance transform
            weight_maps = []
            for i in range(n):
                mask = self.warped_masks[i]
                # Distance transform: piksel di tengah gambar mendapat weight tinggi
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float64)
                weight_maps.append(dist)

            # Menentukan seam berdasarkan weight terbesar
            self.seam_masks = []
            for i in range(n):
                seam_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                for y in range(canvas_h):
                    for x in range(min(canvas_w, 200)):
                        pass  # Skip per-pixel untuk efisiensi

                # Menggunakan argmax per-piksel (winning source)
                if i == 0:
                    stack = np.stack(weight_maps, axis=0)
                    winner = np.argmax(stack, axis=0)

                    for j in range(n):
                        mask_j = ((winner == j) & (self.warped_masks[j] > 0)).astype(np.uint8) * 255
                        self.seam_masks.append(mask_j)
                    break  # Hanya iterasi sekali

            print(f"    Seam masks dibuat untuk {n} gambar")
            self.timings['seam'] = time.time() - t0
            print(f"    Waktu: {self.timings['seam']:.3f} detik")

        def blend_multiband(self):
            """
            Tahap 8: Multi-band blending menggunakan Laplacian pyramid.

            Multi-band blending memblend frekuensi rendah (warna) dan
            frekuensi tinggi (detail) secara terpisah, menghasilkan
            transisi yang very smooth.
            """
            print("\n  [Tahap 8] Multi-band blending...")
            t0 = time.time()

            n = len(self.compensated)
            canvas_w, canvas_h = self.canvas_size
            levels = self.blend_levels

            # Membuat weight maps (normalized) untuk blending
            weight_maps = []
            for i in range(n):
                mask = self.warped_masks[i]
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float64)
                weight_maps.append(dist)

            # Normalisasi weight maps (total weight harus = 1 per piksel)
            weight_sum = sum(weight_maps)
            weight_sum = np.maximum(weight_sum, 1e-10)
            normalized_weights = [w / weight_sum for w in weight_maps]

            def build_gaussian_pyramid(img, lvl):
                """Membangun piramida Gaussian."""
                gp = [img.astype(np.float64)]
                for _ in range(lvl):
                    gp.append(cv2.pyrDown(gp[-1]))
                return gp

            def build_laplacian_pyramid(img, lvl):
                """Membangun piramida Laplacian dari piramida Gaussian."""
                gp = build_gaussian_pyramid(img, lvl)
                lp = []
                for i in range(lvl):
                    expanded = cv2.pyrUp(gp[i + 1],
                                          dstsize=(gp[i].shape[1], gp[i].shape[0]))
                    lp.append(gp[i] - expanded)
                lp.append(gp[lvl])
                return lp

            # Membangun Laplacian pyramid untuk setiap gambar
            image_pyramids = []
            for i in range(n):
                lp = build_laplacian_pyramid(self.compensated[i], levels)
                image_pyramids.append(lp)

            # Membangun Gaussian pyramid untuk setiap weight map
            weight_pyramids = []
            for i in range(n):
                gp = build_gaussian_pyramid(normalized_weights[i], levels)
                weight_pyramids.append(gp)

            # Menggabungkan piramida berdasarkan weight
            blended_pyramid = []
            for lvl in range(levels + 1):
                blended_level = np.zeros_like(image_pyramids[0][lvl])
                for i in range(n):
                    w = weight_pyramids[i][lvl]
                    if len(blended_level.shape) == 3:
                        w_3d = np.stack([w] * 3, axis=-1)
                    else:
                        w_3d = w
                    blended_level += image_pyramids[i][lvl] * w_3d
                blended_pyramid.append(blended_level)

            # Merekonstruksi gambar dari piramida
            result = blended_pyramid[levels]
            for i in range(levels - 1, -1, -1):
                result = cv2.pyrUp(result,
                                    dstsize=(blended_pyramid[i].shape[1],
                                             blended_pyramid[i].shape[0]))
                result = result + blended_pyramid[i]

            self.blended = np.clip(result, 0, 255).astype(np.uint8)

            self.timings['blend'] = time.time() - t0
            print(f"    Blending selesai: {self.blended.shape[1]}x{self.blended.shape[0]}")
            print(f"    Waktu: {self.timings['blend']:.3f} detik")

        def crop_result(self):
            """
            Tahap 9: Auto-cropping area hitam pada hasil panorama.

            Menghilangkan border hitam yang muncul akibat warping
            dengan menemukan bounding rectangle dari area non-hitam terbesar.
            """
            print("\n  [Tahap 9] Auto-cropping area hitam...")
            t0 = time.time()

            if self.blended is None:
                print("    [ERROR] Tidak ada gambar untuk di-crop!")
                return

            # Mengkonversi ke grayscale dan threshold
            gray = cv2.cvtColor(self.blended, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

            # Menemukan kontur dari area non-hitam
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                self.cropped = self.blended.copy()
                print("    Tidak ada area untuk di-crop")
            else:
                # Mengambil bounding rectangle dari kontur terbesar
                biggest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(biggest)

                # Menambahkan margin kecil (5 piksel)
                margin = 5
                x = max(0, x + margin)
                y = max(0, y + margin)
                w = min(w - 2 * margin, self.blended.shape[1] - x)
                h = min(h - 2 * margin, self.blended.shape[0] - y)

                # Melakukan cropping
                self.cropped = self.blended[y:y + h, x:x + w]
                print(f"    Crop: ({x},{y}) → ({x + w},{y + h})")

            self.timings['crop'] = time.time() - t0
            print(f"    Ukuran setelah crop: {self.cropped.shape[1]}x{self.cropped.shape[0]}")
            print(f"    Waktu: {self.timings['crop']:.3f} detik")

        def run(self, images, label=""):
            """
            Menjalankan seluruh pipeline secara berurutan.

            Parameter:
            - images : List gambar BGR
            - label  : Label untuk logging

            Returns:
            - result : Gambar panorama hasil akhir
            """
            print(f"\n{'=' * 50}")
            print(f"  PIPELINE: {label}")
            print(f"  Input: {len(images)} gambar")
            print(f"{'=' * 50}")

            self.images = images
            t_total = time.time()

            # Menjalankan setiap tahap pipeline secara berurutan
            self.detect_features()
            self.match_features()
            self.estimate_homographies()
            self.compute_canvas_size()
            self.warp_images()
            self.exposure_compensate()
            self.find_seam()
            self.blend_multiband()
            self.crop_result()

            self.timings['total'] = time.time() - t_total
            print(f"\n  [SELESAI] Total waktu: {self.timings['total']:.3f} detik")

            return self.cropped

        def print_report(self, label=""):
            """
            Mencetak laporan lengkap pipeline: statistik dan timing.
            """
            print(f"\n{'=' * 55}")
            print(f"  LAPORAN PIPELINE: {label}")
            print(f"{'=' * 55}")

            # Statistik fitur
            print(f"\n  Statistik Fitur:")
            print(f"    Total keypoints : {self.stats.get('total_keypoints', 'N/A')}")
            print(f"    Total matches   : {self.stats.get('total_matches', 'N/A')}")
            print(f"    Total inliers   : {self.stats.get('total_inliers', 'N/A')}")
            print(f"    Gambar referensi: {self.stats.get('ref_idx', 'N/A')}")

            # Timing
            print(f"\n  Waktu Eksekusi per Tahap:")
            print(f"    {'Tahap':<20} {'Waktu (detik)':<15} {'%':<8}")
            print(f"    {'-' * 43}")

            total = self.timings.get('total', 1.0)
            for stage, t in self.timings.items():
                if stage != 'total':
                    pct = t / total * 100
                    print(f"    {stage:<20} {t:>10.3f}     {pct:>5.1f}%")

            print(f"    {'-' * 43}")
            print(f"    {'TOTAL':<20} {total:>10.3f}     100.0%")

            # Ukuran output
            if self.cropped is not None:
                print(f"\n  Ukuran Output: {self.cropped.shape[1]}x{self.cropped.shape[0]}")


    # ============================================================
    # LANGKAH 1: Memuat Set Gambar
    # ============================================================
    print("\n[LANGKAH 1] Memuat set gambar panorama...")

    # Mendefinisikan semua set gambar yang akan diproses
    image_sets = {
        "Outdoor (3 gambar)": [f"panorama_outdoor_{i}.jpg" for i in range(1, 4)],
        "Wide (5 gambar)": [f"panorama_wide_{i}.jpg" for i in range(1, 6)],
        "Indoor (4 gambar)": [f"panorama_indoor_{i}.jpg" for i in range(1, 5)],
    }

    loaded_sets = {}

    for set_name, files in image_sets.items():
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
        exit()


    # ============================================================
    # LANGKAH 2: Menjalankan Pipeline pada Setiap Set
    # ============================================================
    print("\n[LANGKAH 2] Menjalankan pipeline pada setiap set gambar...")

    pipeline_results = {}

    for set_name, images in loaded_sets.items():
        try:
            # Membuat instance pipeline baru untuk setiap set
            pipeline = PanoramaPipeline(
                n_features=2000,
                ratio_thresh=0.75,
                ransac_thresh=5.0,
                blend_levels=4
            )

            # Menjalankan pipeline
            result = pipeline.run(images, set_name)

            if result is not None:
                # Menyimpan hasil
                safe_name = set_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"19_pipeline_{safe_name}.jpg"), result)

                # Mencetak laporan
                pipeline.print_report(set_name)

                pipeline_results[set_name] = {
                    'result': result,
                    'pipeline': pipeline,
                    'timings': pipeline.timings.copy(),
                    'stats': pipeline.stats.copy()
                }
            else:
                print(f"  [WARNING] Pipeline gagal untuk {set_name}")

        except Exception as e:
            print(f"  [ERROR] Pipeline gagal untuk {set_name}: {e}")


    # ============================================================
    # LANGKAH 3: Visualisasi Intermediate Results per Tahap
    # ============================================================
    print("\n[LANGKAH 3] Membuat visualisasi intermediate results...")

    for set_name, data in pipeline_results.items():
        try:
            pipeline = data['pipeline']
            n_imgs = len(pipeline.images)

            fig = plt.figure(figsize=(20, 16))

            # Baris 1: Input images
            for i in range(min(n_imgs, 5)):
                ax = fig.add_subplot(4, max(n_imgs, 5), i + 1)
                ax.imshow(cv2.cvtColor(pipeline.images[i], cv2.COLOR_BGR2RGB))
                kp_count = len(pipeline.keypoints[i]) if i < len(pipeline.keypoints) else 0
                ax.set_title(f"Input {i + 1}\n{kp_count} kp", fontsize=9)
                ax.axis('off')

            # Baris 2: Warped images
            cols = max(n_imgs, 5)
            for i in range(min(n_imgs, 5)):
                if i < len(pipeline.warped_images):
                    ax = fig.add_subplot(4, cols, cols + i + 1)
                    warped_rgb = cv2.cvtColor(pipeline.warped_images[i], cv2.COLOR_BGR2RGB)
                    ax.imshow(warped_rgb)
                    ax.set_title(f"Warped {i + 1}", fontsize=9)
                    ax.axis('off')

            # Baris 3: Compensated images
            for i in range(min(n_imgs, 5)):
                if i < len(pipeline.compensated):
                    ax = fig.add_subplot(4, cols, 2 * cols + i + 1)
                    comp_rgb = cv2.cvtColor(pipeline.compensated[i], cv2.COLOR_BGR2RGB)
                    ax.imshow(comp_rgb)
                    ax.set_title(f"Compensated {i + 1}", fontsize=9)
                    ax.axis('off')

            # Baris 4: Final result
            ax_final = fig.add_subplot(4, 1, 4)
            if data['result'] is not None:
                ax_final.imshow(cv2.cvtColor(data['result'], cv2.COLOR_BGR2RGB))
            ax_final.set_title(f"Hasil Pipeline: {set_name}", fontsize=12)
            ax_final.axis('off')

            safe_name = set_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            plt.suptitle(f"Pipeline Stages: {set_name}",
                          fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"19_stages_{safe_name}.png"), dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()
            print(f"  Stages visualization untuk {set_name} disimpan.")

        except Exception as e:
            print(f"  [WARNING] Gagal membuat visualisasi untuk {set_name}: {e}")


    # ============================================================
    # LANGKAH 4: Perbandingan dengan OpenCV Stitcher
    # ============================================================
    print("\n[LANGKAH 4] Membandingkan dengan OpenCV Stitcher API...")

    stitcher_results = {}

    for set_name, images in loaded_sets.items():
        print(f"\n  --- {set_name} ---")

        try:
            # Membuat OpenCV Stitcher
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

            # Menjalankan stitcher
            t0 = time.time()
            status, pano_cv = stitcher.stitch(images)
            t_cv = time.time() - t0

            if status == cv2.Stitcher_OK:
                safe_name = set_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"19_cv_stitcher_{safe_name}.jpg"), pano_cv)
                stitcher_results[set_name] = {
                    'result': pano_cv,
                    'time': t_cv
                }
                print(f"    OpenCV Stitcher: OK ({pano_cv.shape[1]}x{pano_cv.shape[0]})")
                print(f"    Waktu: {t_cv:.3f} detik")
            else:
                status_msg = {
                    1: "ERR_NEED_MORE_IMGS",
                    2: "ERR_HOMOGRAPHY_EST_FAIL",
                    3: "ERR_CAMERA_PARAMS_ADJUST_FAIL"
                }
                print(f"    OpenCV Stitcher: GAGAL ({status_msg.get(status, f'code={status}')})")

        except Exception as e:
            print(f"    [ERROR] OpenCV Stitcher gagal: {e}")

    # Visualisasi perbandingan
    try:
        fig_rows = len(pipeline_results)
        if fig_rows > 0:
            fig, axes = plt.subplots(fig_rows, 2, figsize=(20, 6 * fig_rows))
            if fig_rows == 1:
                axes = axes.reshape(1, 2)

            for idx, (set_name, data) in enumerate(pipeline_results.items()):
                # Pipeline result
                if data['result'] is not None:
                    axes[idx, 0].imshow(cv2.cvtColor(data['result'], cv2.COLOR_BGR2RGB))
                axes[idx, 0].set_title(
                    f"Pipeline Manual: {set_name}\n"
                    f"Waktu: {data['timings'].get('total', 0):.3f}s",
                    fontsize=11
                )
                axes[idx, 0].axis('off')

                # OpenCV Stitcher result
                if set_name in stitcher_results:
                    axes[idx, 1].imshow(
                        cv2.cvtColor(stitcher_results[set_name]['result'], cv2.COLOR_BGR2RGB)
                    )
                    axes[idx, 1].set_title(
                        f"OpenCV Stitcher: {set_name}\n"
                        f"Waktu: {stitcher_results[set_name]['time']:.3f}s",
                        fontsize=11
                    )
                else:
                    axes[idx, 1].set_title(f"OpenCV Stitcher: GAGAL\n{set_name}", fontsize=11)
                axes[idx, 1].axis('off')

            plt.suptitle("Pipeline Manual vs OpenCV Stitcher",
                          fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "19_pipeline_vs_stitcher.png"), dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()
            print("\n  Perbandingan pipeline vs stitcher disimpan.")
    except Exception as e:
        print(f"  [WARNING] Gagal membuat perbandingan: {e}")


    # ============================================================
    # LANGKAH 5: Profiling Timing Bar Chart
    # ============================================================
    print("\n[LANGKAH 5] Membuat profiling timing...")

    try:
        fig, axes = plt.subplots(1, len(pipeline_results), figsize=(8 * len(pipeline_results), 6))
        if len(pipeline_results) == 1:
            axes = [axes]

        stages_order = ['detect', 'match', 'homography', 'canvas', 'warp',
                        'compensate', 'seam', 'blend', 'crop']
        colors = plt.cm.Set3(np.linspace(0, 1, len(stages_order)))

        for idx, (set_name, data) in enumerate(pipeline_results.items()):
            timings = data['timings']

            # Mengambil waktu per tahap
            stage_times = [timings.get(s, 0) for s in stages_order]

            # Membuat bar chart horizontal
            bars = axes[idx].barh(stages_order, stage_times, color=colors, edgecolor='black')
            axes[idx].set_xlabel("Waktu (detik)")
            axes[idx].set_title(f"{set_name}\nTotal: {timings.get('total', 0):.3f}s", fontsize=11)

            # Menambahkan label waktu pada setiap bar
            for bar, t in zip(bars, stage_times):
                if t > 0.001:
                    axes[idx].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                                   f'{t:.3f}s', va='center', fontsize=9)

        plt.suptitle("Profiling Waktu Eksekusi per Tahap Pipeline",
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "19_profiling_timing.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Profiling timing disimpan.")
    except Exception as e:
        print(f"  [WARNING] Gagal membuat profiling: {e}")


    # ============================================================
    # LANGKAH 6: Pipeline Flow Diagram
    # ============================================================
    print("\n[LANGKAH 6] Membuat pipeline flow diagram...")

    try:
        fig, ax = plt.subplots(1, 1, figsize=(18, 5))

        # Tahapan pipeline
        stages = [
            "Input\nImages", "Detect\nFeatures", "Match\nFeatures",
            "Estimate\nHomography", "Compute\nCanvas", "Warp\nImages",
            "Exposure\nCompensate", "Find\nSeam", "Multi-band\nBlend",
            "Auto\nCrop", "OUTPUT\nPanorama"
        ]

        n_stages = len(stages)
        x_positions = np.linspace(0.05, 0.95, n_stages)

        for i, (x, stage) in enumerate(zip(x_positions, stages)):
            # Menggambar kotak tahap
            color = '#3498db' if i > 0 and i < n_stages - 1 else '#e74c3c'
            bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
            ax.text(x, 0.5, stage, transform=ax.transAxes, fontsize=9,
                    ha='center', va='center', bbox=bbox, color='white',
                    fontweight='bold')

            # Menggambar panah antar tahap
            if i < n_stages - 1:
                x_next = x_positions[i + 1]
                ax.annotate('', xy=(x_next - 0.025, 0.5), xytext=(x + 0.025, 0.5),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='black', lw=2))

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Pipeline Panorama Stitching", fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "19_pipeline_flow.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  Pipeline flow diagram disimpan.")
    except Exception as e:
        print(f"  [WARNING] Gagal membuat flow diagram: {e}")


    # ============================================================
    # LANGKAH 7: Comprehensive Report Table
    # ============================================================
    print("\n[LANGKAH 7] Comprehensive Report")
    print("=" * 75)
    print(f"{'Set':<25} {'Gambar':<8} {'KP':<8} {'Match':<8} "
          f"{'Inlier':<8} {'Waktu(s)':<10} {'Ukuran':<15}")
    print("-" * 75)

    for set_name, data in pipeline_results.items():
        stats = data['stats']
        timings = data['timings']
        result = data['result']
        n_imgs = len(loaded_sets[set_name])
        size_str = f"{result.shape[1]}x{result.shape[0]}" if result is not None else "N/A"

        print(f"{set_name:<25} {n_imgs:<8} {stats.get('total_keypoints', 0):<8} "
              f"{stats.get('total_matches', 0):<8} {stats.get('total_inliers', 0):<8} "
              f"{timings.get('total', 0):>7.3f}   {size_str:<15}")

    # OpenCV Stitcher comparison
    print(f"\n{'OpenCV Stitcher:'}")
    print(f"{'Set':<25} {'Waktu(s)':<10} {'Ukuran':<15}")
    print("-" * 50)
    for set_name, data in stitcher_results.items():
        size_str = f"{data['result'].shape[1]}x{data['result'].shape[0]}"
        print(f"{set_name:<25} {data['time']:>7.3f}   {size_str:<15}")

    print("-" * 75)


    # ============================================================
    # RINGKASAN PROGRAM
    # ============================================================
    print("\n" + "=" * 65)
    print("RINGKASAN PERCOBAAN 19")
    print("=" * 65)
    print(f"""
    Apa yang telah dipelajari:
    1. Pipeline Design:
       - Class PanoramaPipeline dengan 9 tahap modular
       - Setiap tahap independen dan terukur waktunya

    2. Feature Detection & Matching:
       - SIFT_create() untuk deteksi fitur scale-invariant
       - FlannBasedMatcher untuk pencocokan cepat
       - Lowe's ratio test untuk filtering

    3. Homography Chain:
       - Estimasi homography per pasangan dengan RANSAC
       - Akumulasi homography dari referensi (gambar tengah)

    4. Image Warping & Compensation:
       - warpPerspective ke canvas bersama
       - LAB-space gain compensation untuk exposure

    5. Multi-band Blending:
       - Laplacian pyramid decomposition
       - Distance-based weight maps
       - Gaussian pyramid untuk weight propagation

    6. Auto-cropping:
       - Threshold → findContours → boundingRect
       - Menghilangkan border hitam dari warping

    7. Perbandingan dengan OpenCV Stitcher:
       - Pipeline manual vs Stitcher API
       - Trade-off: kontrol vs kemudahan

    File output disimpan di folder: output/
    """)

    print("Program selesai dijalankan.")
    print("=" * 65)



if __name__ == "__main__":
    main()
