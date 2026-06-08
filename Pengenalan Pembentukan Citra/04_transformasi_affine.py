"""
==========================================================================
 PERCOBAAN 4 — TRANSFORMASI AFFINE
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Menerapkan transformasi affine yang mempertahankan garis paralel.
 Konsep  : Affine = translasi + rotasi + scaling + shearing.
           cv2.getAffineTransform(pts_src, pts_dst) → matriks 2×3
           cv2.warpAffine(src, M, (w, h))
           6 DOF (degrees of freedom).
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


def transformasi_affine_3titik(img, pts_src, pts_dst):
    """
    Transformasi affine menggunakan 3 pasang titik korespondensi.
    M = cv2.getAffineTransform(pts_src, pts_dst)
    """
    M = cv2.getAffineTransform(pts_src, pts_dst)
    h, w = img.shape[:2]
    hasil = cv2.warpAffine(img, M, (w, h))
    print(f"  Matriks Affine:\n{M}")
    return hasil, M


def transformasi_affine_manual(img, M):
    """Menerapkan matriks affine yang dibuat manual."""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def tampilkan_hasil(img, hasil_list, labels):
    """Visualisasi hasil transformasi affine."""
    n = len(hasil_list)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original"); axes[0].axis("off")

    for i, (im, lbl) in enumerate(zip(hasil_list, labels)):
        axes[i + 1].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(lbl); axes[i + 1].axis("off")

    plt.suptitle("Percobaan 4 — Transformasi Affine", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "04_transformasi_affine.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 4: TRANSFORMASI AFFINE")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
    h, w = img.shape[:2]
    print(f"\n  Ukuran gambar: {w}×{h}")

    hasil_list, labels = [], []

    # 1) Affine dari 3 titik
    print("\n[1] Affine dari 3 pasang titik:")
    pts_src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    pts_dst = np.float32([[w * 0.1, h * 0.15], [w * 0.85, h * 0.1], [w * 0.15, h * 0.9]])
    aff1, M1 = transformasi_affine_3titik(img, pts_src, pts_dst)
    hasil_list.append(aff1)
    labels.append("Affine 3-Titik")

    # 2) Affine manual: shear + translasi
    print("\n[2] Affine manual (shear + translasi):")
    M2 = np.float32([[1, 0.3, 0], [0.1, 1, 0]])
    aff2 = transformasi_affine_manual(img, M2)
    print(f"  M = {M2.tolist()}")
    hasil_list.append(aff2)
    labels.append("Shear + Translasi")

    # 3) Affine: rotasi + scaling
    print("\n[3] Affine: rotasi 25° + scale 0.8:")
    M3 = cv2.getRotationMatrix2D((w // 2, h // 2), 25, 0.8)
    aff3 = transformasi_affine_manual(img, M3)
    hasil_list.append(aff3)
    labels.append("Rotasi 25° + Scale 0.8")

    tampilkan_hasil(img, hasil_list, labels)

    print("\nRINGKASAN:")
    print("  Affine: garis paralel tetap paralel")
    print("  6 DOF: 2 translasi + 1 rotasi + 2 skala + 1 shear")
    print("  getAffineTransform(3 titik src, 3 titik dst)")


if __name__ == "__main__":
    main()
