"""
==========================================================================
 PERCOBAAN 7 — PROYEKSI 3D KE 2D
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memahami proyeksi titik 3D ke bidang gambar 2D.
 Konsep  : Model kamera pinhole: p = K · [R|t] · P
           K = matriks intrinsik, [R|t] = matriks ekstrinsik
           cv2.projectPoints() untuk memproyeksikan titik 3D.
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def buat_kubus_3d():
    """Membuat titik-titik 3D kubus (8 vertex)."""
    pts = np.float32([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # dasar
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],   # atas
    ])
    return pts


def buat_sumbu_3d(panjang=2.0):
    """Membuat titik sumbu X, Y, Z."""
    return np.float32([
        [0, 0, 0],
        [panjang, 0, 0],
        [0, panjang, 0],
        [0, 0, panjang],
    ])


def proyeksi_ke_2d(pts_3d, K, rvec, tvec, dist=None):
    """
    Memproyeksikan titik 3D ke 2D menggunakan model kamera.
    cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
    """
    if dist is None:
        dist = np.zeros(5)
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    pts_2d = pts_2d.reshape(-1, 2)
    return pts_2d


def gambar_kubus_2d(canvas, pts_2d):
    """Menggambar kubus pada canvas dari titik-titik 2D."""
    pts = pts_2d.astype(int)
    # Dasar (biru)
    for i in range(4):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (255, 0, 0), 2)
    # Atas (merah)
    for i in range(4, 8):
        j = 4 + (i - 4 + 1) % 4
        cv2.line(canvas, tuple(pts[i]), tuple(pts[j]), (0, 0, 255), 2)
    # Vertikal (hijau)
    for i in range(4):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 4]), (0, 255, 0), 2)
    return canvas


def gambar_sumbu_2d(canvas, pts_2d):
    """Menggambar sumbu X(merah), Y(hijau), Z(biru)."""
    origin = tuple(pts_2d[0].astype(int))
    warna = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    label = ['X', 'Y', 'Z']
    for i in range(3):
        endpoint = tuple(pts_2d[i + 1].astype(int))
        cv2.arrowedLine(canvas, origin, endpoint, warna[i], 3, tipLength=0.15)
        cv2.putText(canvas, label[i], endpoint, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, warna[i], 2)
    return canvas


def tampilkan_hasil(canvas_kubus, canvas_sumbu, canvas_multi):
    """Visualisasi proyeksi 3D ke 2D."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(cv2.cvtColor(canvas_kubus, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Proyeksi Kubus 3D"); axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(canvas_sumbu, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Proyeksi Sumbu XYZ"); axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(canvas_multi, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Multi Viewpoint"); axes[2].axis("off")

    plt.suptitle("Percobaan 7 — Proyeksi 3D ke 2D", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "07_proyeksi_3d_ke_2d.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 7: PROYEKSI 3D KE 2D")
    print("=" * 60)

    # Parameter kamera simulasi
    K = np.float64([[500, 0, 320],
                     [0, 500, 240],
                     [0,   0,   1]])
    rvec = np.float64([0.3, 0.5, 0.1])
    tvec = np.float64([0, 0, 5])

    kubus = buat_kubus_3d()
    sumbu = buat_sumbu_3d(2.0)

    # 1) Proyeksi kubus
    print("\n[1] Proyeksi kubus 3D:")
    pts_kubus = proyeksi_ke_2d(kubus, K, rvec, tvec)
    canvas1 = np.ones((480, 640, 3), dtype=np.uint8) * 30
    gambar_kubus_2d(canvas1, pts_kubus)
    print(f"  8 vertex diproyeksikan ke 2D")

    # 2) Proyeksi sumbu
    print("\n[2] Proyeksi sumbu XYZ:")
    pts_sumbu = proyeksi_ke_2d(sumbu, K, rvec, tvec)
    canvas2 = np.ones((480, 640, 3), dtype=np.uint8) * 30
    gambar_sumbu_2d(canvas2, pts_sumbu)

    # 3) Multi viewpoint
    print("\n[3] Multi viewpoint (rotasi berbeda):")
    canvas3 = np.ones((480, 640, 3), dtype=np.uint8) * 30
    for i, rv in enumerate([[0.2, 0.3, 0], [0.5, 0.8, 0.2], [0, 0, 0.5]]):
        rv = np.float64(rv)
        pts = proyeksi_ke_2d(kubus, K, rv, tvec)
        color = [(255, 100, 100), (100, 255, 100), (100, 100, 255)][i]
        pts_int = pts.astype(int)
        for j in range(4):
            cv2.line(canvas3, tuple(pts_int[j]), tuple(pts_int[(j + 1) % 4]), color, 1)
            cv2.line(canvas3, tuple(pts_int[j + 4]), tuple(pts_int[4 + (j + 1) % 4]), color, 1)
            cv2.line(canvas3, tuple(pts_int[j]), tuple(pts_int[j + 4]), color, 1)

    tampilkan_hasil(canvas1, canvas2, canvas3)

    print("\nRINGKASAN:")
    print("  Model pinhole: p = K · [R|t] · P")
    print("  K = matriks intrinsik (fx, fy, cx, cy)")
    print("  [R|t] = pose kamera (rotasi + translasi)")
    print("  cv2.projectPoints() untuk proyeksi 3D→2D")


if __name__ == "__main__":
    main()
