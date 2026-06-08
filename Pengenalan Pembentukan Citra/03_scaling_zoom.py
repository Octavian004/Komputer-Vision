"""
==========================================================================
 PERCOBAAN 3 — SCALING DAN ZOOM
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memperbesar/memperkecil gambar dengan faktor skala
           dan membandingkan metode interpolasi.
 Konsep  : cv2.resize(img, dsize, fx, fy, interpolation)
           Scaling = perubahan ukuran, bagian dari transformasi similarity.
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


def scaling_faktor(img, fx, fy, metode=cv2.INTER_LINEAR):
    """Memperbesar/memperkecil gambar dengan faktor skala fx dan fy."""
    hasil = cv2.resize(img, None, fx=fx, fy=fy, interpolation=metode)
    print(f"  Scale fx={fx}, fy={fy}: {img.shape[:2]} → {hasil.shape[:2]}")
    return hasil


def zoom_crop(img, faktor):
    """
    Zoom ke pusat gambar: perbesar lalu crop ke ukuran asli.
    Simulasi efek zoom kamera.
    """
    h, w = img.shape[:2]
    besar = cv2.resize(img, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_CUBIC)
    hb, wb = besar.shape[:2]
    y_start = (hb - h) // 2
    x_start = (wb - w) // 2
    cropped = besar[y_start:y_start + h, x_start:x_start + w]
    print(f"  Zoom {faktor}×: crop dari {wb}×{hb} ke {w}×{h}")
    return cropped


def bandingkan_interpolasi(img, faktor=3.0):
    """Membandingkan 5 metode interpolasi saat memperbesar gambar."""
    metode = {
        "NEAREST":  cv2.INTER_NEAREST,
        "LINEAR":   cv2.INTER_LINEAR,
        "CUBIC":    cv2.INTER_CUBIC,
        "AREA":     cv2.INTER_AREA,
        "LANCZOS4": cv2.INTER_LANCZOS4,
    }
    hasil = {}
    for nama, m in metode.items():
        h, w = img.shape[:2]
        crop = img[h//4:h//4+50, w//4:w//4+50]
        up = cv2.resize(crop, None, fx=faktor, fy=faktor, interpolation=m)
        hasil[nama] = up
        print(f"  Interpolasi {nama}: 50×50 → {up.shape[1]}×{up.shape[0]}")
    return hasil


def tampilkan_hasil(img, scales, zooms, interp):
    """Visualisasi scaling, zoom, dan interpolasi."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Baris 1: Scaling
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    for i, (lbl, im) in enumerate(scales[:2]):
        axes[0, i + 1].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[0, i + 1].set_title(lbl); axes[0, i + 1].axis("off")

    # Baris 2: Zoom
    axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Original"); axes[1, 0].axis("off")
    for i, (lbl, im) in enumerate(zooms[:2]):
        axes[1, i + 1].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[1, i + 1].set_title(lbl); axes[1, i + 1].axis("off")

    # Baris 3: Interpolasi (3 dari 5)
    interp_items = list(interp.items())
    for i in range(min(3, len(interp_items))):
        nama, im = interp_items[i]
        axes[2, i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[2, i].set_title(f"Interp: {nama}"); axes[2, i].axis("off")

    plt.suptitle("Percobaan 3 — Scaling dan Zoom", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "03_scaling_zoom.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 3: SCALING DAN ZOOM")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
    print(f"\n  Ukuran gambar: {img.shape}")

    print("\n[1] Scaling dengan faktor:")
    s_kecil = scaling_faktor(img, 0.5, 0.5)
    s_besar = scaling_faktor(img, 2.0, 2.0)
    scales = [("0.5× (Kecil)", s_kecil), ("2.0× (Besar)", s_besar)]

    print("\n[2] Zoom ke pusat:")
    z2 = zoom_crop(img, 2.0)
    z3 = zoom_crop(img, 3.0)
    zooms = [("Zoom 2×", z2), ("Zoom 3×", z3)]

    print("\n[3] Perbandingan interpolasi (zoom 3×):")
    interp = bandingkan_interpolasi(img, 3.0)

    tampilkan_hasil(img, scales, zooms, interp)

    print("\nRINGKASAN:")
    print("  Scaling = resize dengan faktor fx, fy")
    print("  Zoom = perbesar + crop ke ukuran asli")
    print("  NEAREST=cepat/blocky, CUBIC=halus, LANCZOS4=terbaik upscale")


if __name__ == "__main__":
    main()
