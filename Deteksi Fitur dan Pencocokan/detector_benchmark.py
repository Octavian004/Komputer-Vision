import cv2
import numpy as np
import time
from utils.visualization import draw_keypoints_side_by_side, show_image_grid


class DetectorBenchmark:
    def __init__(self, detectors=None):
        """Inisialisasi dengan nama detector list."""
        self.detect_funcs = {
            'Harris': self._harris,
            'Shi-Tomasi': self._shi_tomasi,
            'SIFT': self._sift,
            'ORB': self._orb,
            'AKAZE': self._akaze,
            'FAST': self._fast,
            'BRISK': self._brisk,
        }
        self.detector_names = detectors or list(self.detect_funcs.keys())

    def run_detector(self, name, image):
        if name not in self.detect_funcs:
            raise ValueError(f"Detektor tidak tersedia: {name}")
        start = time.time()
        kps, desc = self.detect_funcs[name](image)
        elapsed = (time.time() - start) * 1000.0
        return dict(name=name, keypoints=kps, descriptors=desc, time_ms=elapsed)

    def compare_all(self, image):
        results = []
        for name in self.detector_names:
            try:
                res = self.run_detector(name, image)
                results.append(res)
            except Exception as e:
                print(f"Gagal menjalankan {name}: {e}")
        return results

    def visualization(self, image, results, save_path=None):
        figs = []
        titles = []
        for r in results:
            img_kp = draw_keypoints_side_by_side(image, r['keypoints'], title=f"{r['name']} ({len(r['keypoints'])})")
            figs.append(img_kp)
            titles.append(f"{r['name']}")
        show_image_grid(figs, titles, cols=2, figsize=(14, 10), save_path=save_path)

    def _harris(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        thresh = 0.01 * dst.max()
        kps = [cv2.KeyPoint(float(x[1]), float(x[0]), 3) for x in np.argwhere(dst > thresh)]
        return kps, None

    def _shi_tomasi(self, image, max_corners=500):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
        kps = []
        if corners is not None:
            for c in corners:
                x, y = c.ravel()
                kps.append(cv2.KeyPoint(float(x), float(y), 3))
        return kps, None

    def _sift(self, image):
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(image, None)

    def _orb(self, image):
        orb = cv2.ORB_create()
        return orb.detectAndCompute(image, None)

    def _akaze(self, image):
        akaze = cv2.AKAZE_create()
        return akaze.detectAndCompute(image, None)

    def _fast(self, image):
        fast = cv2.FastFeatureDetector_create()
        kps = fast.detect(image, None)
        return kps, None

    def _brisk(self, image):
        brisk = cv2.BRISK_create()
        return brisk.detectAndCompute(image, None)
