"""
==========================================================================
 PERCOBAAN 1 — TRANSLASI GAMBAR
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Menggeser posisi gambar secara horizontal dan vertikal
           menggunakan matriks translasi dan cv2.warpAffine.
 Konsep  : Translasi = perpindahan posisi piksel
           M = [[1, 0, tx], [0, 1, ty]]
           tx = geser horizontal (+kanan, -kiri)
           ty = geser vertikal (+bawah, -atas)
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


def translasi(img, tx, ty):
    """
    Menggeser gambar sebesar (tx, ty) piksel.
    Matriks translasi 2×3:
      M = | 1  0  tx |
          | 0  1  ty |
    """
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    hasil = cv2.warpAffine(img, M, (w, h))
    print(f"  Translasi tx={tx}, ty={ty}")
    return hasil


def translasi_multi_arah(img):
    """Menggeser gambar ke 4 arah berbeda."""
    h, w = img.shape[:2]
    offset = min(h, w) // 5
    arah = {
        "Kanan": (offset, 0),
        "Bawah": (0, offset),
        "Kiri-Atas": (-offset, -offset),
        "Kanan-Bawah": (offset, offset),
    }
    hasil = {}
    for nama, (tx, ty) in arah.items():
        hasil[nama] = translasi(img, tx, ty)
    return hasil


def tampilkan_hasil(img, hasil_multi):
    """Visualisasi translasi ke berbagai arah."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")

    for i, (nama, im) in enumerate(hasil_multi.items()):
        r, c = divmod(i + 1, 3)
        axes[r, c].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[r, c].set_title(f"Translasi: {nama}")
        axes[r, c].axis("off")

    axes[1, 2].axis("off")

    plt.suptitle("Percobaan 1 — Translasi Gambar", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "01_translasi_gambar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 1: TRANSLASI GAMBAR")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
    print(f"\n  Ukuran gambar: {img.shape}")

    print("\n[1] Translasi ke berbagai arah:")
    hasil = translasi_multi_arah(img)

    tampilkan_hasil(img, hasil)

    print("\nRINGKASAN:")
    print("  Translasi = geser posisi gambar")
    print("  M = [[1,0,tx],[0,1,ty]]")
    print("  cv2.warpAffine(img, M, (w,h))")


if __name__ == "__main__":
    main()
