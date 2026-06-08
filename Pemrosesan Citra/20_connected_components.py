"""
Praktikum 20 - Analisis Komponen Terhubung (Connected Components)
Modul 03: Pemrosesan Citra

Topik: cv2.connectedComponentsWithStats(), labeling, blob analysis
Referensi: OpenCV-Python Tutorial Ch.4, Learning OpenCV Ch.8,
           Mastering OpenCV 4 Ch.3
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
from matplotlib import cm


def demo_basic_connected_components():
    """Connected components dasar dengan cv2.connectedComponents()."""
    # Buat gambar biner dengan beberapa blob
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (130, 120), (255, 255, 255), -1)
    cv2.circle(img, (250, 80), 55, (255, 255, 255), -1)
    cv2.rectangle(img, (350, 40), (470, 140), (255, 255, 255), -1)
    cv2.circle(img, (80, 230), 45, (255, 255, 255), -1)
    cv2.ellipse(img, (300, 240), (100, 50), 15, 0, 360, (255, 255, 255), -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Versi sederhana (hanya jumlah dan label)
    num_labels, labels = cv2.connectedComponents(binary)
    print(f"[connectedComponents] Jumlah komponen (inkl. background): {num_labels}")
    print(f"  Objek ditemukan: {num_labels - 1}")

    # Visualisasi dengan warna berbeda per komponen
    colormap = (cm.get_cmap('tab10', num_labels)(np.arange(num_labels)) * 255).astype(np.uint8)
    label_colored = colormap[labels][:, :, :3]  # RGB
    label_colored[labels == 0] = [40, 40, 40]   # background gelap

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Gambar Asli"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(binary, cmap='gray'); plt.title("Citra Biner"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(label_colored); plt.title(f"Labels ({num_labels-1} komponen)"); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_20_connected_basic.png"), dpi=100); plt.show()
    return binary


def demo_connected_components_with_stats():
    """connectedComponentsWithStats: area, bounding box, centroid per komponen."""
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    shapes = [
        ("Persegi", lambda: cv2.rectangle(img, (30, 30), (130, 120), (255, 255, 255), -1)),
        ("Lingkaran", lambda: cv2.circle(img, (250, 80), 55, (255, 255, 255), -1)),
        ("Persegi pj", lambda: cv2.rectangle(img, (350, 40), (470, 140), (255, 255, 255), -1)),
        ("Kecil", lambda: cv2.circle(img, (80, 230), 45, (255, 255, 255), -1)),
        ("Elips", lambda: cv2.ellipse(img, (300, 240), (100, 50), 15, 0, 360, (255, 255, 255), -1)),
    ]
    for _, fn in shapes:
        fn()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # connectedComponentsWithStats: label, stats, centroids
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    print(f"\n{'Label':>6} {'Area':>8} {'X':>5} {'Y':>5} {'W':>5} {'H':>5} {'CX':>7} {'CY':>7}")
    print("-" * 52)

    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 200, 0), (0, 200, 255), (200, 0, 255)]

    for label in range(1, num_labels):  # skip 0 = background
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]
        color = colors[(label - 1) % len(colors)]

        print(f"{label:>6} {area:>8} {x:>5} {y:>5} {w:>5} {h:>5} {cx:>7.1f} {cy:>7.1f}")
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.circle(result, (int(cx), int(cy)), 4, (0, 255, 255), -1)
        cv2.putText(result, f"L{label}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(binary, cmap='gray'); plt.title("Citra Biner"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB));
    plt.title("BoundingBox + Centroid per Komponen"); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_20_cc_stats.png"), dpi=100); plt.show()


def demo_filter_by_area():
    """Filter komponen berdasarkan ukuran area (hapus noise kecil)."""
    # Simulasi gambar dengan noise titik-titik kecil
    img = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(img, (100, 150), 70, 255, -1)
    cv2.rectangle(img, (220, 80), (380, 230), 255, -1)

    # Tambah noise acak
    np.random.seed(42)
    noise_pts = np.random.randint(0, [400, 300], size=(200, 2))
    for pt in noise_pts:
        img[pt[1], pt[0]] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Filter: hanya simpan komponen dengan area > 500 piksel
    min_area = 500
    clean = np.zeros_like(img)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255

    print(f"  Total komponen: {num_labels - 1}")
    print(f"  Komponen area >= {min_area}: {np.sum([stats[l, cv2.CC_STAT_AREA] >= min_area for l in range(1, num_labels)])}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title(f"Asli (noise)\n{num_labels-1} komponen"); plt.axis('off')
    area_map = np.array([stats[labels[r, c], cv2.CC_STAT_AREA] if labels[r, c] > 0 else 0
                         for r in range(img.shape[0]) for c in range(img.shape[1])]).reshape(img.shape)
    plt.subplot(1, 3, 2); plt.imshow(area_map, cmap='hot'); plt.title("Heat Map Area"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(clean, cmap='gray'); plt.title(f"Setelah Filter Area ≥ {min_area}"); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_20_filter_area.png"), dpi=100); plt.show()


def demo_connectivity_4vs8():
    """Perbedaan konektivitas 4-tetangga vs 8-tetangga."""
    # Gambar diagonal
    img = np.zeros((10, 10), dtype=np.uint8)
    for i in range(10):
        img[i, i] = 255
    img_large = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)

    n4, labels4 = cv2.connectedComponents(img, connectivity=4)
    n8, labels8 = cv2.connectedComponents(img, connectivity=8)

    print(f"  Konektivitas-4: {n4 - 1} komponen")
    print(f"  Konektivitas-8: {n8 - 1} komponen (diagonal dianggap terhubung)")

    cmap4 = np.zeros((*img.shape, 3), dtype=np.uint8)
    cmap8 = np.zeros((*img.shape, 3), dtype=np.uint8)
    pal = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)] * 5
    for l in range(1, n4):
        cmap4[labels4 == l] = pal[l % len(pal)]
    for l in range(1, n8):
        cmap8[labels8 == l] = pal[l % len(pal)]

    cmap4_large = cv2.resize(cmap4, (200, 200), interpolation=cv2.INTER_NEAREST)
    cmap8_large = cv2.resize(cmap8, (200, 200), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(img_large, cmap='gray'); plt.title("Diagonal Image"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(cmap4_large); plt.title(f"Conn-4: {n4-1} komponen\n(diagonal TIDAK terhubung)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cmap8_large); plt.title(f"Conn-8: {n8-1} komponen\n(diagonal terhubung)"); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_20_connectivity.png"), dpi=100); plt.show()


def demo_blob_analysis():
    """Analisis blob lengkap: filter, sort, shape descriptor."""
    img = np.zeros((350, 600, 3), dtype=np.uint8)
    blobs = [
        ((80, 100), 60, (255, 255, 255)),
        ((250, 100), 30, (255, 255, 255)),
        ((380, 100), 80, (255, 255, 255)),
        ((530, 100), 20, (255, 255, 255)),
        ((150, 270), 50, (255, 255, 255)),
        ((350, 270), 40, (255, 255, 255)),
        ((500, 260), 70, (255, 255, 255)),
    ]
    for (cx, cy), r, c in blobs:
        cv2.circle(img, (cx, cy), r, c, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Sort berdasarkan area (terbesar ke terkecil)
    blob_info = [(label, stats[label, cv2.CC_STAT_AREA], centroids[label])
                 for label in range(1, num_labels)]
    blob_info.sort(key=lambda x: x[1], reverse=True)

    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    print(f"\n{'Rank':>4} {'Label':>6} {'Area':>8} {'CX':>7} {'CY':>7}")
    print("-" * 35)
    for rank, (label, area, (cx, cy)) in enumerate(blob_info):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 200, 0), 1)
        cv2.putText(result, f"#{rank+1}", (int(cx)-10, int(cy)+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(f"{rank+1:>4} {label:>6} {area:>8} {cx:>7.1f} {cy:>7.1f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(binary, cmap='gray'); plt.title("Citra Biner"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title("Ranked by Area"); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_20_blob_analysis.png"), dpi=100); plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("PRAKTIKUM 20: ANALISIS KOMPONEN TERHUBUNG (CONNECTED COMPONENTS)")
    print("=" * 60)

    print("\n[1] Connected Components Dasar")
    demo_basic_connected_components()

    print("\n[2] Stats per Komponen (Area, BBox, Centroid)")
    demo_connected_components_with_stats()

    print("\n[3] Filter Komponen berdasarkan Area (hapus noise)")
    demo_filter_by_area()

    print("\n[4] Konektivitas 4 vs 8")
    demo_connectivity_4vs8()

    print("\n[5] Analisis Blob Lengkap")
    demo_blob_analysis()

    print("\n[SELESAI] Semua demo connected components berhasil dijalankan.")
