import cv2
import numpy as np
import time
from utils.visualization import draw_matches


class RobustMatcher:
    def __init__(self, detector='SIFT', matcher='FLANN', ratio=0.75, ransac_thresh=5.0):
        self.detector_name = detector
        self.matcher_name = matcher
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh
        self.detector = self._create_detector(detector)
        self.matcher = self._create_matcher(matcher, detector)

    def _create_detector(self, name):
        if name == 'SIFT':
            return cv2.SIFT_create()
        if name == 'ORB':
            return cv2.ORB_create()
        if name == 'AKAZE':
            return cv2.AKAZE_create()
        raise ValueError(f"Detector tidak disupport: {name}")

    def _create_matcher(self, matcher, detector):
        if matcher == 'FLANN':
            if detector in ['SIFT']:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
            else:
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        if matcher == 'BF':
            norm = cv2.NORM_L2 if detector in ['SIFT'] else cv2.NORM_HAMMING
            return cv2.BFMatcher(norm, crossCheck=False)
        raise ValueError(f"Matcher tidak disupport: {matcher}")

    def detect_and_compute(self, img):
        if img is None:
            raise ValueError('Image tidak boleh None')
        kp, des = self.detector.detectAndCompute(img, None)
        return kp, des

    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []
        raw = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in raw:
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return good, raw

    def verify_geometry(self, kp1, kp2, matches):
        if len(matches) < 4:
            return None, None, []
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_thresh)
        if mask is None:
            return H, None, []
        inliers = [m for i, m in enumerate(matches) if mask[i]]
        return H, mask.ravel().tolist(), inliers

    def visualize(self, img1, img2, kp1, kp2, matches, mask, save_path=None):
        vis = draw_matches(img1, kp1, img2, kp2, matches, mask=mask)
        if save_path:
            cv2.imwrite(save_path, vis)
        return vis

    def match_and_verify(self, img1, img2):
        t0 = time.time()
        kp1, des1 = self.detect_and_compute(img1)
        kp2, des2 = self.detect_and_compute(img2)
        t1 = time.time()
        matches, raw = self.match(des1, des2)
        t2 = time.time()
        H, mask, inliers = self.verify_geometry(kp1, kp2, matches)
        t3 = time.time()

        log = {
            'kp1': len(kp1),
            'kp2': len(kp2),
            'raw_matches': len(raw),
            'good_matches': len(matches),
            'inliers': len(inliers),
            'time_detect': (t1 - t0) * 1000,
            'time_match': (t2 - t1) * 1000,
            'time_ransac': (t3 - t2) * 1000,
        }
        return log, kp1, kp2, matches, mask, inliers, H
