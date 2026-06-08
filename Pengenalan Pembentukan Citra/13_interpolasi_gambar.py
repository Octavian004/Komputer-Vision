"""
==========================================================================
 PERCOBAAN 13 — INTERPOLASI GAMBAR
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memahami dan membandingkan 5 metode interpolasi yang digunakan
           saat gambar di-resize, dirotasi, atau ditransformasi.
 Konsep  :
   - Interpolasi diperlukan karena posisi piksel output tidak selalu tepat
     pada piksel input → perlu estimasi nilai dari tetangga.
   - INTER_NEAREST  : ambil piksel terdekat → cepat, blocky/pixelated.
   - INTER_LINEAR   : rata-rata 4 piksel terdekat (bilinear) → cepat, halus.
   - INTER_CUBIC    : konvolusi 4×4 piksel (bicubic) → lebih halus.
   - INTER_LANCZOS4 : konvolusi 8×8 piksel (Lanczos) → terbaik untuk upscale.
   - INTER_AREA     : rata-rata area piksel → terbaik untuk downscale, alias-free.
 Catatan : Pilihan interpolasi mempengaruhi kualitas dan kecepatan.
           NEAREST untuk sketsa/pixel art, AREA untuk thumbnail,
           LANCZOS4 untuk mencetak resolusi tinggi.
==========================================================================
"""

import cv2
import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METODE = {
    "NEAREST":  cv2.INTER_NEAREST,
    "LINEAR":   cv2.INTER_LINEAR,
    "CUBIC":    cv2.INTER_CUBIC,
    "LANCZOS4": cv2.INTER_LANCZOS4,
    "AREA":     cv2.INTER_AREA,
}


def baca_gambar(nama: str) -> np.ndarray:
    path = os.path.join(IMAGE_DIR, nama)
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        for y in range(0, 300, 30):
            cv2.line(img, (0, y), (400, y), (180, 180, 180), 1)
        for x in range(0, 400, 30):
            cv2.line(img, (x, 0), (x, 300), (180, 180, 180), 1)
        cv2.circle(img, (200, 150), 100, (0, 120, 200), -1)
        cv2.putText(img, "TEST", (130, 165), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 4)
    return img


# ── Demo 1: Perbandingan Upscale ──────────────────────────────────────────────
def demo_upscale(img, faktor: float = 4.0):
    """
    Perbandingan kualitas 5 metode saat gambar diperbesar 4×.
    Perbedaan paling terlihat pada tepi objek dan area gradien halus.
    """
    h, w = img.shape[:2]
    nh, nw = int(h * faktor), int(w * faktor)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Asli ({w}×{h})")
    axes[0, 0].axis("off")

    for ax, (nama, flag) in zip(axes.flat[1:], METODE.items()):
        result = cv2.resize(img, (nw, nh), interpolation=flag)
        # Ambil patch tengah untuk perbandingan detail
        crop = result[nh//2-60:nh//2+60, nw//2-80:nw//2+80]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{nama}\n(patch {faktor}×)")
        ax.axis("off")

    plt.suptitle(f"Percobaan 13a — Upscale {faktor}×: Perbandingan Interpolasi",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"13_upscale_{int(faktor)}x.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Demo 2: Perbandingan Downscale ────────────────────────────────────────────
def demo_downscale(img, faktor: float = 4.0):
    """
    Perbandingan 5 metode saat gambar diperkecil 4×.
    INTER_AREA biasanya memberikan hasil terbaik untuk downscale.
    """
    h, w = img.shape[:2]
    nh, nw = int(h / faktor), int(w / faktor)

    hasil = {}
    waktu = {}
    for nama, flag in METODE.items():
        t0 = time.perf_counter()
        r = cv2.resize(img, (nw, nh), interpolation=flag)
        waktu[nama] = (time.perf_counter() - t0) * 1000  # ms
        hasil[nama] = r

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Asli ({w}×{h})")
    axes[0].axis("off")

    for ax, (nama, im) in zip(axes[1:], hasil.items()):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{nama}\n{waktu[nama]:.2f} ms")
        ax.axis("off")

    plt.suptitle(f"Percobaan 13b — Downscale {faktor}×: Kualitas & Kecepatan",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"13_downscale_{int(faktor)}x.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")

    # Tabel kecepatan
    print("\n  Kecepatan interpolasi:")
    for nama, t in waktu.items():
        print(f"  {nama:10s}: {t:.3f} ms")


# ── Demo 3: Rotasi dengan berbagai interpolasi ────────────────────────────────
def demo_rotasi_interpolasi(img):
    """
    Membandingkan kualitas 5 metode interpolasi saat rotasi 30°.
    Rotasi paling baik menunjukkan perbedaan interpolasi karena banyak
    piksel output yang berada di antara piksel input.
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 30, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Asli"); axes[0, 0].axis("off")

    for ax, (nama, flag) in zip(axes.flat[1:], METODE.items()):
        result = cv2.warpAffine(img, M, (w, h), flags=flag)
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Rotasi 30°\n{nama}")
        ax.axis("off")

    plt.suptitle("Percobaan 13c — Rotasi 30°: Pengaruh Metode Interpolasi",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "13_rotasi_interpolasi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Demo 4: Analisis PSNR ─────────────────────────────────────────────────────
def demo_psnr(img):
    """
    Mengukur kualitas interpolasi secara kuantitatif menggunakan PSNR
    (Peak Signal-to-Noise Ratio): downscale lalu upscale kembali,
    bandingkan hasilnya dengan gambar asli.
    PSNR lebih tinggi = lebih mirip aslinya.
    """
    h, w = img.shape[:2]
    faktor = 4

    print("\n  PSNR setelah downscale(4×) → upscale(4×):")
    print(f"  {'Metode':<12} {'PSNR (dB)':>10}")
    print("  " + "-" * 25)

    for nama, flag in METODE.items():
        small = cv2.resize(img, (w // faktor, h // faktor), interpolation=flag
                           if flag != cv2.INTER_AREA else cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        psnr = cv2.PSNR(img, restored)
        print(f"  {nama:<12} {psnr:>10.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 13: INTERPOLASI GAMBAR")
    print("=" * 60)

    img = baca_gambar("foto_kucing.jpg")
    print(f"Gambar: {img.shape[1]}×{img.shape[0]}")

    print("\n[1] Upscale 4×")
    demo_upscale(img, faktor=4.0)

    print("\n[2] Downscale 4×")
    demo_downscale(img, faktor=4.0)

    print("\n[3] Rotasi 30° dengan berbagai interpolasi")
    demo_rotasi_interpolasi(img)

    print("\n[4] Analisis PSNR")
    demo_psnr(img)

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
