"""
==========================================================================
PERCOBAAN 17: CONVEX HULL DAN SHAPE FITTING
==========================================================================
Program ini mempelajari convex hull dan shape fitting.
Praktikum 17 - Convex Hull dan Shape Fitting
Modul 05: Model Fitting dan Optimasi

Topik: cv2.convexHull(), cv2.fitLine(), cv2.fitEllipse(),
       cv2.minAreaRect(), cv2.minEnclosingCircle()
Referensi: Learning OpenCV Ch.8 (Bradski & Kaehler),
           OpenCV-Python Tutorial Ch.4

Hasil: Visualisasi dan analisis disimpan ke folder output/
==========================================================================
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)



def demo_convex_hull():
    """Convex Hull: selubung cembung dari sekumpulan titik."""
    np.random.seed(42)
    pts = np.random.randint(50, 350, (30, 2))
    # Tambah beberapa outlier jauh
    pts = np.vstack([pts, [[10, 10], [390, 10], [390, 390], [10, 390]]])

    hull = cv2.convexHull(pts.astype(np.int32))
    hull_pts = hull.reshape(-1, 2)

    img = np.ones((400, 400, 3), dtype=np.uint8) * 250
    for pt in pts:
        cv2.circle(img, tuple(pt), 5, (50, 50, 50), -1)
    cv2.polylines(img, [hull_pts], True, (0, 0, 200), 2)
    for i, pt in enumerate(hull_pts):
        cv2.circle(img, tuple(pt), 8, (0, 200, 0), -1)
        cv2.putText(img, str(i), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)

    # Defek konveksitas
    defects = None
    try:
        hull_idx = cv2.convexHull(pts.astype(np.int32), returnPoints=False)
        defects = cv2.convexityDefects(pts.astype(np.int32), hull_idx)
        if defects is not None:
            print(f"  Convexity defects: {len(defects)}")
    except Exception:
        pass

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Convex Hull ({len(hull_pts)} vertex dari {len(pts)} titik)"); plt.axis('off')
    plt.tight_layout(); plt.savefig("output_17_convex_hull.png", dpi=100); plt.show()
    print(f"  Hull vertices: {len(hull_pts)}, Total points: {len(pts)}")


def demo_fitline():
    """cv2.fitLine() — fitting garis robust dengan berbagai norm."""
    np.random.seed(10)
    x = np.linspace(0, 300, 60)
    y_true = 0.7 * x + 50
    noise = np.random.normal(0, 20, len(x))
    y_noisy = y_true + noise
    # Tambah outlier
    outlier_idx = np.random.choice(len(x), 10, replace=False)
    y_noisy[outlier_idx] += np.random.choice([-150, 150], 10)
    pts = np.column_stack([x, y_noisy]).astype(np.float32)

    dist_types = {
        'DIST_L2 (OLS)': cv2.DIST_L2,
        'DIST_L1 (Median)': cv2.DIST_L1,
        'DIST_L12': cv2.DIST_L12,
        'DIST_HUBER': cv2.DIST_HUBER,
    }

    def line_y(vx, vy, x0, y0, x_vals):
        return y0 + (vy / (vx + 1e-9)) * (x_vals - x0)

    plt.figure(figsize=(14, 5))
    for i, (name, dist) in enumerate(dist_types.items()):
        line = cv2.fitLine(pts, dist, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()
        y_fit = line_y(vx, vy, x0, y0, x)
        plt.subplot(1, 4, i + 1)
        plt.scatter(x, y_noisy, c='lightblue', s=20, alpha=0.6, label='Data+outlier')
        plt.scatter(x[outlier_idx], y_noisy[outlier_idx], c='red', s=40, label='Outlier')
        plt.plot(x, y_true, 'g--', linewidth=1.5, label='True')
        plt.plot(x, y_fit, 'b-', linewidth=2, label=name)
        plt.legend(fontsize=7); plt.title(name); plt.grid(True, alpha=0.3)
    plt.suptitle("cv2.fitLine() — Perbandingan Norm (robust vs OLS)")
    plt.tight_layout(); plt.savefig("output_17_fitline.png", dpi=100); plt.show()


def demo_shape_fitting():
    """Fitting bounding shapes: minAreaRect, minEnclosingCircle, fitEllipse."""
    shapes = {
        'Lingkaran': lambda: cv2.circle(
            np.zeros((300, 400, 3), np.uint8), (200, 150), 100, (255, 255, 255), -1),
        'Persegi Panjang Miring': None,
        'Elips Miring': None,
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    results = []

    # Shape 1: Lingkaran
    img1 = np.zeros((300, 400, 3), np.uint8)
    cv2.circle(img1, (200, 150), 100, (255, 255, 255), -1)
    results.append(img1)

    # Shape 2: Persegi panjang miring 45°
    img2 = np.zeros((300, 400, 3), np.uint8)
    box_pts = cv2.boxPoints(((200, 150), (180, 80), 45))
    cv2.fillPoly(img2, [np.intp(box_pts)], (255, 255, 255))
    results.append(img2)

    # Shape 3: Elips miring
    img3 = np.zeros((300, 400, 3), np.uint8)
    cv2.ellipse(img3, (200, 150), (130, 60), 30, 0, 360, (255, 255, 255), -1)
    results.append(img3)

    titles = ["Lingkaran", "Persegi Miring 45°", "Elips Miring 30°"]
    for i, (img, title) in enumerate(zip(results, titles)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        vis = img.copy()

        # minAreaRect
        rect = cv2.minAreaRect(cnt)
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)

        # minEnclosingCircle
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(vis, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)

        # fitEllipse (perlu ≥ 5 titik)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(vis, ellipse, (255, 200, 0), 2)

        area = cv2.contourArea(cnt)
        min_rect_area = rect[1][0] * rect[1][1]
        min_circ_area = np.pi * radius ** 2
        print(f"\n  [{title}]")
        print(f"    Contour area: {area:.1f}")
        print(f"    minAreaRect: {rect[1][0]:.1f}x{rect[1][1]:.1f}, angle={rect[2]:.1f}°, area={min_rect_area:.1f}")
        print(f"    minEnclosingCircle: r={radius:.1f}, area={min_circ_area:.1f}")

        axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{title}\n🟥 minAreaRect  🟢 minCircle  🟡 fitEllipse")
        axes[i].axis('off')

    plt.suptitle("Shape Fitting: minAreaRect, minEnclosingCircle, fitEllipse")
    plt.tight_layout(); plt.savefig("output_17_shape_fitting.png", dpi=100); plt.show()


def demo_convexity_defects():
    """Deteksi convexity defects — aplikasi: menghitung jari tangan."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Buat bentuk seperti tangan (polygon sederhana)
    hand_pts = np.array([
        [180, 380], [180, 250], [160, 180], [140, 120], [120, 70],   # jari telunjuk
        [130, 60],  [150, 110], [165, 165],
        [175, 130], [160, 80],  [145, 40],  [130, 78],   # jari tengah
        [145, 120], [165, 175],
        [185, 150], [175, 95],  [160, 55],  [150, 93],   # jari manis
        [165, 140], [185, 180],
        [200, 200], [215, 170], [205, 140],
        [225, 160], [230, 200], [220, 240],               # kelingking
        [215, 290], [220, 380],
    ], dtype=np.int32)
    cv2.fillPoly(img, [hand_pts], (255, 255, 255))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("  Tidak ada kontur"); return
    cnt = max(cnts, key=cv2.contourArea)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_indices)

    vis = img.copy()
    hull_pts_vis = cv2.convexHull(cnt, returnPoints=True)
    cv2.drawContours(vis, [hull_pts_vis], -1, (0, 255, 0), 2)
    cv2.drawContours(vis, [cnt], -1, (255, 200, 0), 1)

    far_pts = []
    if defects is not None:
        for defect in defects:
            s, e, f, depth = defect[0]
            if depth > 5000:
                far = tuple(cnt[f][0])
                far_pts.append(far)
                cv2.circle(vis, far, 6, (0, 0, 255), -1)

    print(f"  Convexity defects (depth>5000): {len(far_pts)}")
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Convexity Defects\n🔴 = defect point ({len(far_pts)} terdeteksi)"); plt.axis('off')
    plt.tight_layout(); plt.savefig("output_17_convexity_defects.png", dpi=100); plt.show()


if __name__ == "__main__":
    print("=" * 55)
    print("PRAKTIKUM 17: CONVEX HULL DAN SHAPE FITTING")
    print("=" * 55)

    print("\n[1] Convex Hull dari Sekumpulan Titik")
    demo_convex_hull()

    print("\n[2] cv2.fitLine() — Fitting Garis Robust")
    demo_fitline()

    print("\n[3] Shape Fitting: minAreaRect, minEnclosingCircle, fitEllipse")
    demo_shape_fitting()

    print("\n[4] Convexity Defects — Aplikasi Deteksi Jari")
    demo_convexity_defects()

    print("\n[SELESAI] Semua demo convex hull dan shape fitting berhasil dijalankan.")
