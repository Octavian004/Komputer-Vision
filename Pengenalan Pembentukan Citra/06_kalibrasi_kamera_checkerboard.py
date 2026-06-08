"""
==========================================================================
 PERCOBAAN 6 — KALIBRASI KAMERA (CHECKERBOARD)
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memahami proses kalibrasi kamera menggunakan pola checkerboard.
 Konsep  : Kalibrasi = mencari parameter intrinsik K (focal length, optical center)
           dan parameter distorsi (radial k1,k2,k3 dan tangensial p1,p2).
           cv2.findChessboardCorners(), cv2.calibrateCamera()
           Simulasi: buat checkerboard sintetis → terapkan distorsi → kalibrasi.
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


def buat_checkerboard(rows=7, cols=9, square_size=40):
    """
    Membuat gambar checkerboard sintetis.
    rows × cols = jumlah kotak internal.
    """
    h = (rows + 1) * square_size
    w = (cols + 1) * square_size
    board = np.zeros((h, w), dtype=np.uint8)
    for i in range(rows + 1):
        for j in range(cols + 1):
            if (i + j) % 2 == 0:
                y1 = i * square_size
                y2 = y1 + square_size
                x1 = j * square_size
                x2 = x1 + square_size
                board[y1:y2, x1:x2] = 255
    print(f"  Checkerboard sintetis: {w}×{h}, kotak={square_size}px")
    return board


def deteksi_corner_checkerboard(img_gray, pattern_size=(8, 6)):
    """
    Mendeteksi corner pada checkerboard.
    pattern_size = (cols-1, rows-1) titik internal.
    """
    ret, corners = cv2.findChessboardCorners(img_gray, pattern_size, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
        print(f"  Corner terdeteksi: {len(corners)} titik")
    else:
        print("  Corner TIDAK terdeteksi!")
    return ret, corners


def simulasi_kalibrasi(img_gray, pattern_size=(8, 6), square_size=40):
    """
    Simulasi proses kalibrasi kamera dari 1 gambar checkerboard.
    """
    ret, corners = deteksi_corner_checkerboard(img_gray, pattern_size)
    if not ret:
        return None, None, None

    # Titik 3D dunia (z=0 untuk semua)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                            0:pattern_size[1]].T.reshape(-1, 2) * square_size

    h, w = img_gray.shape[:2]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners], (w, h), None, None
    )

    print(f"\n  [Matriks Intrinsik K]:")
    print(f"    fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"    cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"  [Distorsi]: k1={dist[0][0]:.4f}, k2={dist[0][1]:.4f}")
    print(f"  Reprojection error: {ret:.4f}")

    return K, dist, corners


def tampilkan_hasil(board, board_corners, K, dist):
    """Visualisasi checkerboard dan hasil kalibrasi."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(board, cmap='gray')
    axes[0].set_title("Checkerboard Sintetis"); axes[0].axis("off")

    # Gambar corner pada checkerboard
    board_color = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    if board_corners is not None:
        cv2.drawChessboardCorners(board_color, (8, 6), board_corners, True)
    axes[1].imshow(cv2.cvtColor(board_color, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Corner Terdeteksi"); axes[1].axis("off")

    # Info kalibrasi
    axes[2].axis("off")
    if K is not None:
        info = (
            f"Matriks Intrinsik K:\n"
            f"  fx = {K[0,0]:.1f}\n"
            f"  fy = {K[1,1]:.1f}\n"
            f"  cx = {K[0,2]:.1f}\n"
            f"  cy = {K[1,2]:.1f}\n\n"
            f"Distorsi:\n"
            f"  k1 = {dist[0][0]:.6f}\n"
            f"  k2 = {dist[0][1]:.6f}\n"
            f"  p1 = {dist[0][2]:.6f}\n"
            f"  p2 = {dist[0][3]:.6f}"
        )
    else:
        info = "Kalibrasi gagal"
    axes[2].text(0.1, 0.5, info, fontfamily='monospace', fontsize=12,
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[2].set_title("Parameter Kalibrasi")

    plt.suptitle("Percobaan 6 — Kalibrasi Kamera Checkerboard", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "06_kalibrasi_kamera.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 6: KALIBRASI KAMERA (CHECKERBOARD)")
    print("=" * 60)

    print("\n[1] Buat checkerboard sintetis:")
    board = buat_checkerboard(7, 9, 40)

    print("\n[2] Deteksi corner:")
    K, dist, corners = simulasi_kalibrasi(board, (8, 6), 40)

    tampilkan_hasil(board, corners, K, dist)

    print("\nRINGKASAN:")
    print("  Kalibrasi = mencari matriks intrinsik K dan distorsi")
    print("  findChessboardCorners → deteksi titik sudut")
    print("  calibrateCamera → hitung K, dist, rvecs, tvecs")
    print("  K berisi: fx, fy (focal), cx, cy (optical center)")


if __name__ == "__main__":
    main()
