"""
==========================================================================
 PERCOBAAN 11 — KOREKSI DISTORSI LENSA
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Mengoreksi distorsi lensa menggunakan parameter kalibrasi
           kamera yang sudah diketahui.
 Konsep  :
   - cv2.undistort(img, K, distCoeffs)          → koreksi satu langkah.
   - cv2.initUndistortRectifyMap() + cv2.remap() → koreksi via lookup map
     (lebih efisien untuk video karena map dihitung sekali saja).
   - getOptimalNewCameraMatrix() → mengatur apakah tepi gambar dipotong
     atau dipertahankan (alpha=0: tidak ada piksel hitam; alpha=1: semua
     piksel asli tersimpan tetapi ada area hitam di tepi).
   - Reprojection error: ukuran kualitas kalibrasi (idealnya < 0.5 piksel).
 Catatan : Dalam percobaan ini parameter kalibrasi dibuat secara sintetis
           (disimulasikan), bukan dari proses kalibrasi nyata.
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Utilitas ──────────────────────────────────────────────────────────────────
def buat_gambar_grid(h=400, w=500, step=40):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (200, 200, 200), 1)
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (200, 200, 200), 1)
    cx, cy = w // 2, h // 2
    for r in range(step, min(h, w) // 2, step):
        cv2.circle(img, (cx, cy), r, (80, 80, 200), 1)
    return img


def baca_gambar(nama):
    path = os.path.join(IMAGE_DIR, nama)
    img = cv2.imread(path)
    return img if img is not None else buat_gambar_grid()


def buat_kamera_sintetis(w: int, h: int):
    """
    Membuat parameter kamera sintetis yang realistis.
    K  : matriks intrinsik (focal length fx=fy=w, principal point di tengah).
    dist: koefisien distorsi barrel sedang.
    """
    f = float(w)
    K = np.array([[f, 0, w/2.0],
                  [0, f, h/2.0],
                  [0, 0, 1.0 ]], dtype=np.float64)
    # Barrel distortion moderat
    dist = np.array([-0.3, 0.1, 0.001, 0.001, -0.02], dtype=np.float64)
    return K, dist


def distorsi_dengan_remap(img, K, dist):
    """
    Menerapkan distorsi menggunakan initUndistortRectifyMap secara terbalik:
    menghasilkan gambar yang terlihat terdistorsi dari gambar bersih.
    """
    h, w = img.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(K, -dist, None, K, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── Demo 1: undistort vs remap ─────────────────────────────────────────────────
def demo_undistort_vs_remap(img):
    """
    Perbandingan dua cara koreksi distorsi:
    1. cv2.undistort() — mudah, cocok untuk gambar tunggal.
    2. initUndistortRectifyMap + remap — efisien untuk video/banyak frame.
    Keduanya menghasilkan output yang IDENTIK.
    """
    h, w = img.shape[:2]
    K, dist_orig = buat_kamera_sintetis(w, h)

    # Terapkan distorsi ke gambar bersih
    K2, dist_sim = buat_kamera_sintetis(w, h)
    img_dist = distorsi_dengan_remap(img, K2, dist_sim)

    # Metode 1: undistort langsung
    undistorted = cv2.undistort(img_dist, K2, dist_sim)

    # Metode 2: remap (efisien untuk video)
    map1, map2 = cv2.initUndistortRectifyMap(K2, dist_sim, None, K2, (w, h), cv2.CV_32FC1)
    remap_out = cv2.remap(img_dist, map1, map2, cv2.INTER_LINEAR)

    # Identis?
    mse = np.mean((undistorted.astype(float) - remap_out.astype(float))**2)
    print(f"  MSE undistort vs remap: {mse:.6f}  (0 = identis)")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, im, title in zip(axes,
                              [img, img_dist, undistorted, remap_out],
                              ["Asli (bersih)", "Setelah distorsi",
                               "Koreksi undistort()", "Koreksi remap()"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 11a — Koreksi Distorsi: undistort vs remap", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "11_undistort_vs_remap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")
    return img_dist, K2, dist_sim


# ── Demo 2: Pengaruh alpha getOptimalNewCameraMatrix ─────────────────────────
def demo_optimal_camera_matrix(img_dist, K, dist):
    """
    getOptimalNewCameraMatrix() mengontrol trade-off antara:
    - alpha=0 : semua piksel output valid (tidak ada area hitam), tetapi
                beberapa piksel tepi gambar terpotong.
    - alpha=1 : semua piksel asli tersimpan, tetapi ada area hitam di tepi.
    """
    h, w = img_dist.shape[:2]
    alphas = [0.0, 0.3, 0.6, 1.0]

    fig, axes = plt.subplots(1, len(alphas) + 1, figsize=(22, 4))
    axes[0].imshow(cv2.cvtColor(img_dist, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Terdistorsi")
    axes[0].axis("off")

    for ax, alpha in zip(axes[1:], alphas):
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
        result = cv2.undistort(img_dist, K, dist, None, newK)
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(f"alpha={alpha:.1f}")
        ax.axis("off")

    plt.suptitle("Percobaan 11b — getOptimalNewCameraMatrix (alpha 0–1)", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "11_optimal_camera_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Demo 3: Visualisasi vektor distorsi ───────────────────────────────────────
def demo_vektor_distorsi(K, dist, w=500, h=400):
    """
    Visualisasi vektor perpindahan setiap piksel akibat distorsi.
    Membantu memahami pola geometris distorsi (radial = arah dari pusat).
    """
    step = 30
    ys, xs = np.mgrid[0:h:step, 0:w:step].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    pts_dist = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)

    dx = pts_dist[:, 0] - pts.reshape(-1, 2)[:, 0]
    dy = pts_dist[:, 1] - pts.reshape(-1, 2)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(pts.reshape(-1, 2)[:, 0], pts.reshape(-1, 2)[:, 1],
              dx, -dy, scale=5, color="royalblue", width=0.003)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.invert_yaxis()
    ax.set_title("Vektor Distorsi (barrel distortion)", fontsize=12)
    ax.set_xlabel("x (piksel)"); ax.set_ylabel("y (piksel)")
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, alpha=0.3)
    ax.plot(w/2, h/2, "r*", markersize=15, label="Principal Point")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "11_vektor_distorsi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 11: KOREKSI DISTORSI LENSA")
    print("=" * 60)

    grid = buat_gambar_grid()
    foto = baca_gambar("kota.jpg")

    print("\n[1] Koreksi Distorsi: undistort vs remap (pada grid)")
    img_dist, K, dist = demo_undistort_vs_remap(grid)

    print("\n[2] Pengaruh Parameter alpha")
    demo_optimal_camera_matrix(img_dist, K, dist)

    print("\n[3] Visualisasi Vektor Distorsi")
    demo_vektor_distorsi(K, dist)

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
