import os
import cv2
import numpy as np
import argparse
from detector_benchmark import DetectorBenchmark
from robust_matcher import RobustMatcher
from object_recognizer import ObjectRecognizer
from image_retriever import ImageRetriever
from utils.io_utils import read_image, save_image

np.random.seed(42)
cv2.setRNGSeed(42)


def run_benchmark(cfg):
    image = read_image(cfg['benchmark_image'])
    db = DetectorBenchmark()
    results = db.compare_all(image)
    print("Detektor | N Keypoints | Waktu (ms)")
    for r in results:
        print(f"{r['name']}: {len(r['keypoints'])} | {r['time_ms']:.2f}")
    db.visualization(image, results, save_path=os.path.join(cfg['results_folder'], 'benchmark_comparison.png'))


def run_matcher(cfg):
    img1 = read_image(cfg['matcher_image1'])
    img2 = read_image(cfg['matcher_image2'])
    rm = RobustMatcher(detector=cfg['matcher_detector'], matcher=cfg['matcher_matcher'], ratio=cfg['matcher_ratio'], ransac_thresh=cfg['matcher_ransac_thresh'])
    log, kp1, kp2, matches, mask, inliers, H = rm.match_and_verify(img1, img2)
    print(log)
    vis = rm.visualize(img1, img2, kp1, kp2, matches, mask, save_path=os.path.join(cfg['results_folder'], 'matcher_result.png'))
    save_image(os.path.join(cfg['results_folder'], 'matcher_result.png'), vis)
    print('Hasil robust matching disimpan di results/')


def run_recognition(cfg):
    orc = ObjectRecognizer(detector=cfg['recognizer_detector'], matcher=cfg['recognizer_matcher'], ratio=cfg['recognizer_ratio'], ransac_thresh=cfg['recognizer_ransac_thresh'])
    tmpl_folder = cfg['templates_folder']
    for f in os.listdir(tmpl_folder):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(tmpl_folder, f)
            orc.add_template(f, path)
    scene = read_image(cfg['recognizer_scene'])
    det = orc.recognize(scene, min_inlier_ratio=cfg['recognizer_min_inlier_ratio'])
    print('Objek Terdeteksi:')
    for i, d in enumerate(det, 1):
        print(f"[{i}] {d['name']} - Score: {d['score']:.2f} - Inliers: {d['inliers']}/{d['matches']} ({d['status']})")
    annotated = orc.annotate_scene(scene, det)
    save_image(os.path.join(cfg['results_folder'], 'recognition.png'), annotated)


def run_retrieval(cfg):
    retriever = ImageRetriever(detector=cfg['retriever_detector'])
    n = retriever.build_index(cfg['database_folder'])
    print(f"Index dibuat: {n} gambar")
    qimg = read_image(cfg['query_image'])
    res = retriever.query(qimg, top_k=cfg['retriever_top_k'])
    print('Hasil query:')
    for i, r in enumerate(res, 1):
        print(f"[{i}] {r['name']} -> matches: {r['topk']}")
    retriever.visualize_results(qimg, res, save_path=os.path.join(cfg['results_folder'], 'retrieval_results.png'))


def main():
    parser = argparse.ArgumentParser(description='FeatureMatch Vision App')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--matcher', action='store_true')
    parser.add_argument('--recognize', action='store_true')
    parser.add_argument('--retrieve', action='store_true')
    args = parser.parse_args()

    cfg = {
        'benchmark_image': 'data/query/3D Plant.jpeg',
        'matcher_image1': 'data/query/3D Plant.jpeg',
        'matcher_image2': 'data/query/Pemandangan.webp',
        'matcher_detector': 'SIFT',
        'matcher_matcher': 'FLANN',
        'matcher_ratio': 0.75,
        'matcher_ransac_thresh': 5.0,
        'templates_folder': 'data/templates',
        'recognizer_scene': 'data/query/Pemandangan.webp',
        'recognizer_detector': 'SIFT',
        'recognizer_matcher': 'FLANN',
        'recognizer_ratio': 0.75,
        'recognizer_ransac_thresh': 5.0,
        'recognizer_min_inlier_ratio': 0.1,
        'database_folder': 'data/database',
        'query_image': 'data/query/3D Plant.jpeg',
        'retriever_detector': 'SIFT',
        'retriever_top_k': 5,
        'results_folder': 'results'
    }

    os.makedirs(cfg['results_folder'], exist_ok=True)

    if args.benchmark:
        run_benchmark(cfg)
    if args.matcher:
        run_matcher(cfg)
    if args.recognize:
        run_recognition(cfg)
    if args.retrieve:
        run_retrieval(cfg)
    if not (args.benchmark or args.matcher or args.recognize or args.retrieve):
        print('Tambah flag --benchmark, --matcher, --recognize atau --retrieve')


if __name__ == '__main__':
    main()
