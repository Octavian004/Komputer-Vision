"""
==========================================================================
 PERCOBAAN 2 — ROTASI SUDUT BEBAS
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Merotasi gambar dengan sudut bebas menggunakan matriks rotasi.
 Konsep  : cv2.getRotationMatrix2D(center, angle, scale)
           cv2.warpAffine(src, M, (w, h))
           Rotasi 2D: R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]
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


def rotasi_sederhana(img, sudut, scale=1.0):
    """Rotasi gambar pada pusat dengan sudut tertentu."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, sudut, scale)
    hasil = cv2.warpAffine(img, M, (w, h))
    print(f"  Rotasi {sudut}° (scale={scale})")
    return hasil


def rotasi_tanpa_crop(img, sudut):
    """Rotasi gambar dengan canvas yang diperbesar agar tidak terpotong."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, sudut, 1.0)
    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])
    w_baru = int(h * sin_val + w * cos_val)
    h_baru = int(h * cos_val + w * sin_val)
    M[0, 2] += (w_baru - w) / 2
    M[1, 2] += (h_baru - h) / 2
    hasil = cv2.warpAffine(img, M, (w_baru, h_baru))
    print(f"  Rotasi {sudut}° tanpa crop: {w}×{h} → {w_baru}×{h_baru}")
    return hasil


def rotasi_multi_sudut(img, sudut_list):
    """Rotasi gambar ke beberapa sudut sekaligus."""
    return [(s, rotasi_sederhana(img, s)) for s in sudut_list]


def tampilkan_hasil(img, rotasi_list, rotasi_nocrop):
    """Visualisasi rotasi berbagai sudut."""
    n = len(rotasi_list)
    fig, axes = plt.subplots(2, max(n, 3), figsize=(4 * n, 8))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    for i, (s, im) in enumerate(rotasi_list):
        if i + 1 < axes.shape[1]:
            axes[0, i + 1].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            axes[0, i + 1].set_title(f"Rotasi {s}°")
            axes[0, i + 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Original"); axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(rotasi_nocrop, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("30° Tanpa Crop"); axes[1, 1].axis("off")
    for j in range(2, axes.shape[1]):
        axes[1, j].axis("off")

    plt.suptitle("Percobaan 2 — Rotasi Sudut Bebas", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "02_rotasi_sudut_bebas.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 2: ROTASI SUDUT BEBAS")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "baboon.jpg"))
    print(f"\n  Ukuran gambar: {img.shape}")

    print("\n[1] Rotasi berbagai sudut:")
    sudut_list = [45, 90, 135, 180]
    rotasi_list = rotasi_multi_sudut(img, sudut_list)

    print("\n[2] Rotasi tanpa crop:")
    r_nocrop = rotasi_tanpa_crop(img, 30)

    tampilkan_hasil(img, rotasi_list, r_nocrop)

    print("\nRINGKASAN:")
    print("  getRotationMatrix2D(center, angle, scale) → matriks 2×3")
    print("  warpAffine(img, M, (w,h)) → terapkan rotasi")
    print("  Rotasi tanpa crop: perbesar canvas sesuai sin/cos sudut")


if __name__ == "__main__":
    main()
