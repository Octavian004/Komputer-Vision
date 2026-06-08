import os
import cv2
import numpy as np
from robust_matcher import RobustMatcher
from utils.io_utils import read_image
from utils.visualization import draw_bbox


class ObjectRecognizer:
    def __init__(self, detector='SIFT', matcher='FLANN', ratio=0.75, ransac_thresh=5.0):
        self.templates = {}
        self.matcher = RobustMatcher(detector, matcher, ratio, ransac_thresh)

    def add_template(self, name, image_path):
        img = read_image(image_path)
        kp, des = self.matcher.detect_and_compute(img)
        if des is None or len(kp) == 0:
            raise ValueError(f"Template tidak memiliki fitur cukup: {name}")
        self.templates[name] = {'image': img, 'kp': kp, 'des': des}

    def recognize(self, scene_image, min_inlier_ratio=0.1):
        scene_kp, scene_des = self.matcher.detect_and_compute(scene_image)
        if scene_des is None or len(scene_kp) == 0:
            return []
        detected = []

        for name, templ in self.templates.items():
            matches, _ = self.matcher.match(templ['des'], scene_des)
            H, mask, inliers = self.matcher.verify_geometry(templ['kp'], scene_kp, matches)
            if len(inliers) >= 4:
                score = len(inliers) / max(len(matches), 1)
                status = 'OK' if score >= min_inlier_ratio else 'REJECTED'
                detected.append({'name': name, 'score': score, 'inliers': len(inliers), 'matches': len(matches), 'H': H, 'mask': mask, 'status': status})
        detected.sort(key=lambda x: x['score'], reverse=True)
        return detected

    def annotate_scene(self, scene_image, detections):
        out = scene_image.copy()
        h, w = scene_image.shape[:2]
        for d in detections:
            if d['H'] is None or d['status'] == 'REJECTED':
                continue
            timg = self.templates[d['name']]['image']
            th, tw = timg.shape[:2]
            corners = np.float32([[0,0],[tw,0],[tw,th],[0,th]]).reshape(-1,1,2)
            corners_trans = cv2.perspectiveTransform(corners, d['H'])
            out = draw_bbox(out, corners_trans)
            cv2.putText(out, f"{d['name']} {d['score']:.2f}", tuple(corners_trans[0][0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return out
