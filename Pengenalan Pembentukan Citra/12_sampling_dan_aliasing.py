"""
==========================================================================
 PERCOBAAN 12 — SAMPLING DAN ALIASING
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memahami fenomena aliasing saat gambar di-downsample tanpa
           filter anti-aliasing yang tepat.
 Konsep  :
   - Sampling : proses diskritisasi sinyal kontinu menjadi piksel.
   - Teorema Nyquist-Shannon: frekuensi sampling ≥ 2× frekuensi max sinyal.
     Jika dilanggar → ALIASING (artefak frekuensi palsu).
   - Alias tampak sebagai pola moiré, garis bergerigi (jagged), atau
     artefak pada tepi objek bertekstur halus.
   - Anti-aliasing: low-pass filter (Gaussian blur) sebelum downsampling
     menghilangkan frekuensi tinggi yang menyebabkan aliasing.
   - INTER_AREA di OpenCV sudah menerapkan anti-aliasing internal.
 Catatan : Konsep ini sama dengan masalah nyata pada kamera beresolusi
           rendah yang memotret layar monitor atau tekstur halus.
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


# ── Gambar uji frekuensi tinggi ───────────────────────────────────────────────
def buat_zone_plate(h=400, w=400) -> np.ndarray:
    """
    Zone plate: pola lingkaran konsentris frekuensi meningkat dari tengah.
    Sangat sensitif terhadap aliasing — ideal sebagai gambar uji.
    """
    y, x = np.mgrid[-h//2:h//2, -w//2:w//2].astype(np.float32)
    r2 = x**2 + y**2
    img = (np.sin(r2 / 300.0) * 127.5 + 127.5).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def buat_garis_diagonal(h=400, w=400) -> np.ndarray:
    """
    Gambar garis diagonal tipis — menguji aliasing pada edge miring.
    Tanpa anti-aliasing → tepi jagged (bergerigi).
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, 10):
        cv2.line(img, (0, y), (w, y + 50), (255, 255, 255), 1)
    cv2.line(img, (0, 0), (w, h), (0, 200, 255), 1)
    cv2.circle(img, (w//2, h//2), 150, (0, 255, 0), 1)
    return img


# ── Demo 1: Aliasing saat downsampling ────────────────────────────────────────
def demo_aliasing_downsample(img):
    """
    Downsampling TANPA blur dulu → aliasing.
    Downsampling DENGAN Gaussian blur dulu → bersih.
    Perbandingan faktor 4× dan 8× downscale.
    """
    h, w = img.shape[:2]

    for factor in [4, 8]:
        nh, nw = h // factor, w // factor
        # Tanpa anti-aliasing
        alias_nn  = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
        alias_lin = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # Dengan anti-aliasing: blur dulu
        sigma = factor / 2.0
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        anti_alias = cv2.resize(blurred, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # INTER_AREA sudah anti-alias
        inter_area = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # Resize kembali untuk tampilan perbandingan
        disp = [cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST)
                for x in [alias_nn, alias_lin, anti_alias, inter_area]]

        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Asli"); axes[0].axis("off")
        for ax, im, title in zip(axes[1:], disp,
                                  ["NEAREST\n(aliasing)", "LINEAR\n(aliasing)",
                                   "Gaussian+LINEAR\n(anti-alias)", "INTER_AREA\n(anti-alias)"]):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.suptitle(f"Percobaan 12 — Downsample {factor}× : Aliasing vs Anti-aliasing",
                     fontweight="bold")
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"12_aliasing_{factor}x.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[SIMPAN] {out}")


# ── Demo 2: Moiré pattern ─────────────────────────────────────────────────────
def demo_moire():
    """
    Demonstrasi pola moiré: hasil interferensi dua pola garis.
    Pola moiré adalah bentuk visual aliasing yang umum di dunia nyata
    (foto layar monitor, kain bermotif rapat, atap genteng dari udara).
    """
    h, w = 400, 500
    freq1 = 15  # frekuensi pola pertama (garis/periode)
    freq2 = 16  # frekuensi pola kedua (sedikit berbeda → moire)

    x = np.linspace(0, w, w)
    pat1 = (np.sin(2 * np.pi * x / freq1) > 0).astype(np.uint8) * 255
    pat2 = (np.sin(2 * np.pi * x / freq2) > 0).astype(np.uint8) * 255

    img1 = np.tile(pat1, (h, 1))
    img2 = np.tile(pat2, (h, 1))
    moire = cv2.bitwise_and(img1, img2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, im, title in zip(axes, [img1, img2, moire],
                              [f"Pola 1 (f={freq1} px)", f"Pola 2 (f={freq2} px)", "Moiré = AND keduanya"]):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 12c — Pola Moiré (visual aliasing)", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "12_moire.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Demo 3: Pengaruh sigma Gaussian ──────────────────────────────────────────
def demo_pengaruh_sigma(img):
    """Membandingkan kualitas downsampling dengan berbagai nilai sigma blur."""
    h, w = img.shape[:2]
    factor = 6
    nh, nw = h // factor, w // factor

    sigmas = [0, 0.5, 1.0, 2.0, 4.0]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(20, 4))
    for ax, s in zip(axes, sigmas):
        if s == 0:
            down = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
            title = "σ=0\n(no blur)"
        else:
            blur = cv2.GaussianBlur(img, (0, 0), s)
            down = cv2.resize(blur, (nw, nh), interpolation=cv2.INTER_LINEAR)
            title = f"σ={s}"
        ax.imshow(cv2.cvtColor(down, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"Percobaan 12d — Pengaruh Sigma Gaussian sebelum Downsample {factor}×",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "12_sigma_gaussian.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 12: SAMPLING DAN ALIASING")
    print("=" * 60)

    zone = buat_zone_plate()
    garis = buat_garis_diagonal()

    print("\n[1] Aliasing pada Zone Plate (4× dan 8× downscale)")
    demo_aliasing_downsample(zone)

    print("\n[2] Aliasing pada Garis Diagonal")
    demo_aliasing_downsample(garis)

    print("\n[3] Pola Moiré")
    demo_moire()

    print("\n[4] Pengaruh Sigma Gaussian")
    demo_pengaruh_sigma(zone)

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
