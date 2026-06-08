"""
Praktikum 01 - Deteksi Kontur (Contour Detection)
Modul 03: Pemrosesan Citra

Topik: cv2.findContours(), cv2.drawContours(), contour analysis
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


def demo_find_contours(image_path=None):
    """Deteksi kontur dasar dengan findContours."""
    if image_path and cv2.haveImageReader(image_path):
        img = cv2.imread(image_path)
    else:
        # Buat gambar sintetis dengan bentuk-bentuk
        img = np.zeros((400, 500, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(img, (300, 100), 60, (200, 200, 200), -1)
        cv2.ellipse(img, (120, 300), (80, 40), 0, 0, 360, (180, 180, 180), -1)
        cv2.rectangle(img, (300, 250), (450, 370), (150, 150, 150), -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # findContours: RETR_EXTERNAL hanya kontur luar, CHAIN_APPROX_SIMPLE kompres titik
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"[findContours] Jumlah kontur ditemukan: {len(contours)}")

    # Gambar semua kontur
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Analisis setiap kontur
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        print(f"  Kontur {i}: area={area:.1f}, perimeter={perimeter:.1f}")

        # Bounding rect
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(result, f"C{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Gambar Asli"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(binary, cmap='gray'); plt.title("Binary (Threshold)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title("Kontur + Bounding Rect"); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_01_contour_detection.png"), dpi=100); plt.show()
    print("[OK] Output disimpan: output_01_contour_detection.png")
    return contours


def demo_contour_retrieval_modes():
    """Perbandingan mode retrieval: EXTERNAL vs TREE vs LIST."""
    # Buat gambar dengan kontur bersarang
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (370, 270), (255, 255, 255), -1)   # luar
    cv2.rectangle(img, (70, 70), (200, 200), (0, 0, 0), -1)          # lubang dalam
    cv2.circle(img, (130, 130), 30, (255, 255, 255), -1)              # benda dalam lubang
    cv2.circle(img, (300, 150), 60, (200, 200, 200), -1)              # luar lain

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    modes = {
        'RETR_EXTERNAL': cv2.RETR_EXTERNAL,
        'RETR_LIST': cv2.RETR_LIST,
        'RETR_CCOMP': cv2.RETR_CCOMP,
        'RETR_TREE': cv2.RETR_TREE,
    }

    fig, axes = plt.subplots(1, len(modes) + 1, figsize=(20, 4))
    axes[0].imshow(binary, cmap='gray'); axes[0].set_title("Binary"); axes[0].axis('off')
    colors_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]

    for idx, (name, mode) in enumerate(modes.items()):
        cnts, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for j, c in enumerate(cnts):
            cv2.drawContours(vis, [c], -1, colors_list[j % len(colors_list)], 2)
        axes[idx + 1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[idx + 1].set_title(f"{name}\n({len(cnts)} kontur)")
        axes[idx + 1].axis('off')
        print(f"  {name}: {len(cnts)} kontur")

    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_01_retrieval_modes.png"), dpi=100); plt.show()


def demo_contour_approximation():
    """Aproksimasi kontur dengan Ramer-Douglas-Peucker (approxPolyDP)."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    pts = np.array([[100, 50], [200, 30], [280, 100], [260, 200],
                    [180, 260], [100, 230], [40, 150]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (200, 200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    epsilons = [0.01, 0.05, 0.10]
    fig, axes = plt.subplots(1, len(epsilons) + 1, figsize=(18, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original"); axes[0].axis('off')

    for i, eps_frac in enumerate(epsilons):
        epsilon = eps_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vis = img.copy()
        cv2.drawContours(vis, [approx], -1, (0, 255, 0), 2)
        for pt in approx:
            cv2.circle(vis, tuple(pt[0]), 5, (0, 0, 255), -1)
        axes[i + 1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"ε={eps_frac*100:.0f}%\n({len(approx)} titik)")
        axes[i + 1].axis('off')
        print(f"  epsilon={eps_frac}: {len(approx)} titik (dari {len(cnt)} titik asli)")

    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_01_contour_approx.png"), dpi=100); plt.show()


def demo_contour_properties():
    """Properti kontur: moments, centroid, circularity, bounding shapes."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    shapes = [
        ("Lingkaran", lambda i: cv2.circle(i, (100, 100), 70, (255, 255, 255), -1)),
        ("Persegi", lambda i: cv2.rectangle(i, (230, 30), (370, 170), (255, 255, 255), -1)),
        ("Segitiga", lambda i: cv2.fillPoly(i, [np.array([[450, 30], [550, 170], [350, 170]])], (255, 255, 255))),
        ("Elips", lambda i: cv2.ellipse(i, (150, 300), (120, 60), 0, 0, 360, (255, 255, 255), -1)),
    ]
    for _, draw_fn in shapes:
        draw_fn(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    print(f"\n{'Kontur':<8} {'Area':>10} {'Perimeter':>12} {'Circularity':>14} {'Centroid'}")
    print("-" * 58)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-9)

        # Momen untuk centroid
        M = cv2.moments(cnt)
        cx = int(M['m10'] / (M['m00'] + 1e-9))
        cy = int(M['m01'] / (M['m00'] + 1e-9))

        # Min enclosing circle
        (ex, ey), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(result, (int(ex), int(ey)), int(radius), (0, 255, 255), 1)
        # Min area rect
        rect = cv2.minAreaRect(cnt)
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(result, [box], 0, (255, 0, 0), 1)
        cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(result, f"C{i}", (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"C{i:<7} {area:>10.1f} {perimeter:>12.1f} {circularity:>14.3f}  ({cx}, {cy})")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.imshow(binary, cmap='gray'); plt.title("Binary"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title("Properti Kontur"); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_01_contour_properties.png"), dpi=100); plt.show()


if __name__ == "__main__":
    print("=" * 55)
    print("PRAKTIKUM 01: DETEKSI KONTUR (CONTOUR DETECTION)")
    print("=" * 55)

    print("\n[1] Deteksi Kontur Dasar")
    demo_find_contours()

    print("\n[2] Mode Retrieval Kontur")
    demo_contour_retrieval_modes()

    print("\n[3] Aproksimasi Kontur (approxPolyDP)")
    demo_contour_approximation()

    print("\n[4] Properti Kontur (Moments, Circularity, BoundingShapes)")
    demo_contour_properties()

    print("\n[SELESAI] Semua demo contour detection berhasil dijalankan.")
