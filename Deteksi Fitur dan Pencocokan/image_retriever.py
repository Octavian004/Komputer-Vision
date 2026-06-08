import os
import cv2
import numpy as np
from utils.io_utils import read_image, save_pickle, load_pickle
from utils.visualization import show_image_grid


class ImageRetriever:
    def __init__(self, detector='SIFT'):
        self.detector_name = detector
        self.detector = self._create_detector(detector)
        self.index = []

    def _create_detector(self, name):
        if name == 'SIFT':
            return cv2.SIFT_create()
        if name == 'ORB':
            return cv2.ORB_create()
        if name == 'AKAZE':
            return cv2.AKAZE_create()
        raise ValueError(f"Detector tidak disupport: {name}")

    def build_index(self, database_folder):
        if not os.path.isdir(database_folder):
            raise FileNotFoundError("Folder database tidak ditemukan")
        self.index = []
        for fname in os.listdir(database_folder):
            path = os.path.join(database_folder, fname)
            try:
                img = read_image(path)
            except Exception:
                continue
            kp, des = self.detector.detectAndCompute(img, None)
            if des is None:
                continue
            self.index.append({'name': fname, 'path': path, 'kp': kp, 'des': des})
        return len(self.index)

    def query(self, query_image, top_k=5):
        qkp, qdes = self.detector.detectAndCompute(query_image, None)
        if qdes is None:
            return []
        bf = cv2.BFMatcher(cv2.NORM_L2 if self.detector_name=='SIFT' else cv2.NORM_HAMMING, crossCheck=False)
        scores = []
        for item in self.index:
            matches = bf.knnMatch(qdes, item['des'], k=2)
            good = [m for m,n in matches if m.distance < 0.75*n.distance]
            scores.append({'name': item['name'], 'path': item['path'], 'topk': len(good), 'matches': len(matches)})
        scores.sort(key=lambda x: x['topk'], reverse=True)
        return scores[:top_k]

    def save_index(self, path):
        save_pickle(path, self.index)

    def load_index(self, path):
        self.index = load_pickle(path)

    def visualize_results(self, query_image, results, save_path=None):
        """Visualize query image and top results in a grid"""
        images = [query_image]
        titles = [f'Query\n({self.detector_name})']
        
        for r in results:
            try:
                img = read_image(r['path'])
                images.append(img)
                title = f"{os.path.basename(r['name'])}\nMatches: {r['topk']}"
                titles.append(title)
            except Exception:
                continue
        
        show_image_grid(images, titles, cols=2, figsize=(12, 8), save_path=save_path)
