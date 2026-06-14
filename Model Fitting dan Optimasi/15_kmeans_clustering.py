"""
==========================================================================
PERCOBAAN 15: K-MEANS CLUSTERING UNTUK SEGMENTASI CITRA
==========================================================================
Program ini mempelajari k-means clustering untuk segmentasi citra.
Praktikum 15 - K-Means Clustering untuk Segmentasi Citra
Modul 05: Model Fitting dan Optimasi

Topik: cv2.kmeans(), quantisasi warna, segmentasi berbasis cluster
Referensi: Machine Learning for OpenCV Ch.4 (Beyeler),
           Learning OpenCV Ch.16, Mastering OpenCV 4 Ch.5

Hasil: Visualisasi dan analisis disimpan ke folder output/
==========================================================================
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)



def demo_manual_kmeans():
    """Ilustrasi algoritma K-Means dari awal (tanpa library)."""
    np.random.seed(42)
    # Buat data 2D dengan 3 cluster
    cluster1 = np.random.randn(80, 2) * 0.6 + [2, 2]
    cluster2 = np.random.randn(80, 2) * 0.7 + [6, 6]
    cluster3 = np.random.randn(80, 2) * 0.5 + [2, 8]
    data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    colors_plot = ['red', 'green', 'blue']
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.5, s=20)
    plt.title("Data Asli (Belum Cluster)"); plt.grid(True)

    plt.subplot(1, 2, 2)
    for i in range(k):
        mask = (labels.flatten() == i)
        plt.scatter(data[mask, 0], data[mask, 1], c=colors_plot[i], alpha=0.5, s=20, label=f"Cluster {i}")
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', s=200, zorder=5, label='Centroid')
    plt.legend(); plt.title("Hasil K-Means (k=3)"); plt.grid(True)
    plt.tight_layout(); plt.savefig("output_15_kmeans_2d.png", dpi=100); plt.show()

    print(f"[K-Means 2D] k={k}, centroid:\n{centers}")


def demo_color_quantization(image_path=None):
    """Kuantisasi warna (color quantization) dengan K-Means."""
    if image_path and cv2.haveImageReader(image_path):
        img = cv2.imread(image_path)
    else:
        # Buat gambar gradien berwarna
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        for x in range(400):
            color = [int(x * 255 / 400), int((400 - x) * 200 / 400), 100]
            img[:, x] = color
        for y in range(200):
            img[y, :, 2] = int(y * 255 / 200)

    # Reshape ke daftar piksel
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Gambar Asli"); axes[0, 0].axis('off')

    for idx, k in enumerate([2, 4, 8, 16, 32]):
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(img.shape)
        ax = axes[(idx + 1) // 3][(idx + 1) % 3]
        ax.imshow(cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB))
        ax.set_title(f"k={k} warna"); ax.axis('off')
        print(f"  k={k}: {k} warna dominan (dari 16.7 juta warna)")

    plt.suptitle("Kuantisasi Warna dengan K-Means")
    plt.tight_layout(); plt.savefig("output_15_color_quantization.png", dpi=100); plt.show()


def demo_image_segmentation_kmeans(image_path=None):
    """Segmentasi gambar dengan K-Means sebagai model fitting."""
    if image_path and cv2.haveImageReader(image_path):
        img = cv2.imread(image_path)
    else:
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(img, (120, 150), 90, (180, 60, 60), -1)
        cv2.rectangle(img, (240, 60), (370, 240), (60, 180, 60), -1)
        img[:, :] = np.clip(img.astype(int) + np.random.randint(-20, 20, img.shape), 0, 255).astype(np.uint8)
        img[150:, :180] = [60, 60, 180]

    # Feature: pixel position + color
    h, w = img.shape[:2]
    yx = np.mgrid[0:h, 0:w].reshape(2, -1).T.astype(np.float32)
    colors = img.reshape(-1, 3).astype(np.float32)
    # Gabungkan fitur (posisi dinormalisasi + warna)
    yx_norm = yx / [h, w] * 50
    features = np.concatenate([colors, yx_norm], axis=1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    k = 4
    _, labels, _ = cv2.kmeans(features, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    # Buat segmentation map berwarna
    label_img = labels.reshape(h, w)
    seg_colored = np.zeros_like(img)
    seg_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i in range(k):
        seg_colored[label_img == i] = seg_colors[i % len(seg_colors)]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Gambar Asli"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(seg_colored, cv2.COLOR_BGR2RGB)); plt.title(f"Segmentasi K-Means (k={k})"); plt.axis('off')
    plt.tight_layout(); plt.savefig("output_15_segmentation_kmeans.png", dpi=100); plt.show()


def demo_elbow_method():
    """Metode Elbow untuk memilih nilai K optimal."""
    np.random.seed(42)
    # Data dengan 4 cluster sejati
    centers_true = [(1, 1), (5, 1), (1, 5), (5, 5)]
    data = np.vstack([np.random.randn(100, 2) * 0.5 + c for c in centers_true]).astype(np.float32)

    inertias = []
    K_range = range(1, 11)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    for k in K_range:
        compactness, _, _ = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        inertias.append(compactness)
        print(f"  k={k}: inertia={compactness:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(K_range, inertias, 'bo-', markersize=8)
    plt.axvline(x=4, color='r', linestyle='--', label='K optimal = 4')
    plt.xlabel('Jumlah Cluster (K)'); plt.ylabel('Inertia (Within-cluster SS)')
    plt.title('Metode Elbow untuk Memilih K Optimal')
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("output_15_elbow.png", dpi=100); plt.show()


if __name__ == "__main__":
    print("=" * 55)
    print("PRAKTIKUM 15: K-MEANS CLUSTERING UNTUK SEGMENTASI CITRA")
    print("=" * 55)

    print("\n[1] K-Means pada Data 2D")
    demo_manual_kmeans()

    print("\n[2] Kuantisasi Warna (Color Quantization)")
    demo_color_quantization()

    print("\n[3] Segmentasi Gambar dengan K-Means")
    demo_image_segmentation_kmeans()

    print("\n[4] Metode Elbow - Memilih K Optimal")
    demo_elbow_method()

    print("\n[SELESAI] Semua demo K-Means clustering berhasil dijalankan.")
