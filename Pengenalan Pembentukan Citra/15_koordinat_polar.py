"""
==========================================================================
 PERCOBAAN 15 — KOORDINAT POLAR DAN LOG-POLAR
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Mengonversi gambar dari koordinat Cartesian ke Polar dan
           Log-Polar untuk aplikasi seperti iris recognition dan
           rotation-invariant matching.
 Konsep  :
   - Koordinat POLAR: setiap piksel dinyatakan dengan radius r dan
     sudut θ dari titik pusat.
     Cartesian (x,y) → Polar (r, θ): r=√(x²+y²), θ=arctan(y/x).
   - Koordinat LOG-POLAR: radius dikonversi ke skala logaritmik.
     Transformasi ini membuat ROTASI menjadi translasi vertikal dan
     SCALING menjadi translasi horizontal di ruang log-polar!
   - cv2.linearPolar(img, center, maxRadius, flags) → Cartesian ke Polar.
   - cv2.logPolar(img, center, M, flags) → Cartesian ke Log-Polar.
   - cv2.WARP_POLAR_LINEAR/LOG + WARP_FILL_OUTLIERS + WARP_INVERSE_MAP
 Catatan : Transformasi log-polar adalah fondasi dari algoritma
           Fourier-Mellin transform untuk rotation/scale invariant matching.
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


def baca_gambar(nama: str) -> np.ndarray:
    path = os.path.join(IMAGE_DIR, nama)
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(img, (200, 200), 120, (200, 80, 80), -1)
        cv2.rectangle(img, (80, 80), (320, 320), (80, 200, 80), 3)
        cv2.circle(img, (200, 200), 60, (255, 200, 0), -1)
        cv2.putText(img, "POLAR", (120, 215), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 0), 3)
    return img


# ── Demo 1: Linear Polar ──────────────────────────────────────────────────────
def demo_linear_polar(img):
    """
    Konversi Cartesian → Linear Polar dan kembali.
    Dalam ruang polar: sumbu X = radius (0..maxR), sumbu Y = sudut (0..360°).
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    max_r  = min(w, h) / 2.0

    flags_fwd = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
    flags_inv = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP

    polar     = cv2.warpPolar(img, (w, h), center, max_r, flags_fwd)
    recovered = cv2.warpPolar(polar, (w, h), center, max_r, flags_inv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, im, title in zip(axes,
                              [img, polar, recovered],
                              ["Asli (Cartesian)", "Linear Polar\n(X=radius, Y=sudut)",
                               "Rekonstruksi"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 15a — Transformasi Linear Polar", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "15_linear_polar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Demo 2: Log-Polar ─────────────────────────────────────────────────────────
def demo_log_polar(img):
    """
    Konversi ke Log-Polar.
    Keunggulan: rotasi gambar → translasi vertikal; scaling → translasi horizontal.
    Ini membuat matching berbasis Fourier rotation/scale invariant.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    max_r  = min(w, h) / 2.0
    M      = w / np.log(max_r)  # parameter scale log-polar

    flags_fwd = cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS
    flags_inv = cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP

    log_pol   = cv2.warpPolar(img, (w, h), center, max_r, flags_fwd)
    recovered = cv2.warpPolar(log_pol, (w, h), center, max_r, flags_inv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, im, title in zip(axes,
                              [img, log_pol, recovered],
                              ["Asli (Cartesian)", "Log-Polar\n(X=log(r), Y=sudut)",
                               "Rekonstruksi"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 15b — Transformasi Log-Polar", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "15_log_polar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Demo 3: Rotasi = translasi vertikal di Log-Polar ─────────────────────────
def demo_rotasi_jadi_translasi(img):
    """
    BUKTI bahwa rotasi di Cartesian → translasi di Log-Polar:
    img dan img_rotasi 30° → dalam ruang log-polar bergeser secara vertikal.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    max_r  = min(w, h) / 2.0

    M_rot = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_rotasi = cv2.warpAffine(img, M_rot, (w, h))

    flags = cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS
    lp_asli = cv2.warpPolar(img, (w, h), center, max_r, flags)
    lp_rota  = cv2.warpPolar(img_rotasi, (w, h), center, max_r, flags)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, im, title in zip(axes.flat,
                              [img, img_rotasi, lp_asli, lp_rota],
                              ["Asli (Cartesian)", "Rotasi 45°\n(Cartesian)",
                               "Asli (Log-Polar)", "Rotasi 45°\n(Log-Polar → shift vertikal)"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 15c — Rotasi Cartesian → Translasi di Log-Polar",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "15_rotasi_translasi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Demo 4: Aplikasi Iris — unwrap lingkaran ──────────────────────────────────
def demo_unwrap_iris():
    """
    Mensimulasikan unwrapping iris dari bentuk annular (donut) ke persegi
    menggunakan transformasi polar. Teknik ini digunakan dalam pengenalan iris.
    """
    h, w = 400, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    center = (w // 2, h // 2)
    # Gambar iris sintetis
    cv2.circle(img, center, 160, (80, 80, 80), -1)   # iris
    cv2.circle(img, center,  60, (10, 10, 10), -1)   # pupil
    # Pola garis radial sebagai tekstur iris
    for angle in range(0, 360, 12):
        rad = np.radians(angle)
        x2 = int(center[0] + 160 * np.cos(rad))
        y2 = int(center[1] + 160 * np.sin(rad))
        x1 = int(center[0] +  60 * np.cos(rad))
        y1 = int(center[1] +  60 * np.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), (0, int(200*np.abs(np.cos(rad))), 200), 2)

    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
    unwrapped = cv2.warpPolar(img, (360, 100), tuple(map(float, center)), 160.0, flags)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Iris (annular)")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(unwrapped, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Iris Unwrapped (polar strip)")
    axes[1].axis("off")

    plt.suptitle("Percobaan 15d — Unwrap Iris dengan Polar Transform",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "15_unwrap_iris.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 15: KOORDINAT POLAR DAN LOG-POLAR")
    print("=" * 60)

    img = baca_gambar("foto_kucing.jpg")

    print("\n[1] Linear Polar Transform")
    demo_linear_polar(img)

    print("\n[2] Log-Polar Transform")
    demo_log_polar(img)

    print("\n[3] Rotasi = Translasi di Log-Polar")
    demo_rotasi_jadi_translasi(img)

    print("\n[4] Aplikasi: Unwrap Iris")
    demo_unwrap_iris()

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
