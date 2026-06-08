"""
==========================================================================
 PERCOBAAN 5 — TRANSFORMASI PERSPEKTIF (HOMOGRAPHY)
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Menerapkan transformasi perspektif menggunakan matriks 3×3.
 Konsep  : cv2.getPerspectiveTransform(pts_src, pts_dst) → matriks H 3×3
           cv2.warpPerspective(src, H, (w, h))
           8 DOF — garis lurus tetap lurus, tapi paralel bisa konvergen.
           Aplikasi: koreksi dokumen miring, bird's eye view.
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


def perspektif_4titik(img, pts_src, pts_dst, ukuran_output=None):
    """
    Transformasi perspektif menggunakan 4 pasang titik korespondensi.
    H = cv2.getPerspectiveTransform(src, dst) → matriks 3×3
    """
    H = cv2.getPerspectiveTransform(pts_src, pts_dst)
    if ukuran_output is None:
        ukuran_output = (img.shape[1], img.shape[0])
    hasil = cv2.warpPerspective(img, H, ukuran_output)
    print(f"  Homography H:\n{np.round(H, 4)}")
    return hasil, H


def koreksi_dokumen(img):
    """
    Simulasi koreksi perspektif dokumen:
    Misalkan dokumen berada di 4 titik miring → di-warp ke persegi panjang.
    """
    h, w = img.shape[:2]
    # Titik sumber (simulasi dokumen miring)
    pts_src = np.float32([
        [w * 0.15, h * 0.10],
        [w * 0.80, h * 0.05],
        [w * 0.95, h * 0.90],
        [w * 0.05, h * 0.85],
    ])
    # Titik tujuan (persegi panjang lurus)
    pts_dst = np.float32([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ])
    hasil, H = perspektif_4titik(img, pts_src, pts_dst, (w, h))
    return hasil, pts_src


def birds_eye_view(img):
    """Transformasi gambar ke tampilan atas (bird's eye view)."""
    h, w = img.shape[:2]
    pts_src = np.float32([
        [w * 0.2, h * 0.3],
        [w * 0.8, h * 0.3],
        [w, h],
        [0, h],
    ])
    pts_dst = np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ])
    hasil, _ = perspektif_4titik(img, pts_src, pts_dst, (w, h))
    return hasil


def tampilkan_hasil(img, koreksi, pts_src, bev):
    """Visualisasi transformasi perspektif."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")

    # Gambar titik sumber pada original
    img_pts = img.copy()
    pts_int = pts_src.astype(int)
    for i in range(4):
        cv2.circle(img_pts, tuple(pts_int[i]), 8, (0, 255, 0), -1)
        cv2.line(img_pts, tuple(pts_int[i]), tuple(pts_int[(i + 1) % 4]),
                 (0, 0, 255), 2)
    axes[0, 1].imshow(cv2.cvtColor(img_pts, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Titik Sumber"); axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(koreksi, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Koreksi Perspektif"); axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Bird's Eye View"); axes[1, 1].axis("off")

    plt.suptitle("Percobaan 5 — Transformasi Perspektif", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "05_transformasi_perspektif.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 5: TRANSFORMASI PERSPEKTIF (HOMOGRAPHY)")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "dokumen.jpg"))
    print(f"\n  Ukuran gambar: {img.shape}")

    print("\n[1] Koreksi perspektif dokumen:")
    koreksi, pts_src = koreksi_dokumen(img)

    print("\n[2] Bird's Eye View:")
    bev = birds_eye_view(img)

    tampilkan_hasil(img, koreksi, pts_src, bev)

    print("\nRINGKASAN:")
    print("  Perspektif: 8 DOF, garis lurus tetap lurus")
    print("  getPerspectiveTransform(4 titik src, 4 titik dst)")
    print("  warpPerspective(img, H, (w,h))")
    print("  Aplikasi: koreksi dokumen, bird's eye view")


if __name__ == "__main__":
    main()
