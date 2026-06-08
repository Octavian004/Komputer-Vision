"""
==========================================================================
 PERCOBAAN 19 — CITRA SINTETIS UNTUK PENGUJIAN ALGORITMA
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Membuat pola-pola gambar standar yang biasa digunakan sebagai
           target uji (test target) untuk mengevaluasi kualitas algoritma
           pengolahan citra tanpa bergantung pada foto nyata.
 Konsep  :
   - TEST TARGET memungkinkan pengujian yang terkontrol: kita tahu persis
     nilai kebenaran (ground truth) sehingga mudah mengukur error.
   - CHECKERBOARD: digunakan untuk kalibrasi kamera & uji aliasing.
   - SIEMENS STAR: digunakan untuk mengukur resolusi dan sharpness lensa.
   - ZONE PLATE: pola frekuensi radial yang meningkat ke arah luar; berguna
     untuk analisis response frekuensi dan uji aliasing.
   - RAMPA GRADIEN: ideal untuk uji linearitas sensor dan gamma.
   - POLA NOISE: Gaussian, salt-and-pepper, Poisson — uji denoiser.
   - GARIS & TEPI: digunakan untuk mengukur MTF (Modulation Transfer Function).
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Pembuat pola sintetis ─────────────────────────────────────────────────────
def buat_checkerboard(h=300, w=400, ukuran_kotak=30,
                       warna1=0, warna2=255) -> np.ndarray:
    """Checkerboard hitam-putih standar."""
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((y // ukuran_kotak) + (x // ukuran_kotak)) % 2 == 0:
                img[y, x] = warna1
            else:
                img[y, x] = warna2
    return img


def buat_siemens_star(h=400, w=400, n_jari=36,
                       radius=None) -> np.ndarray:
    """Siemens star: target resolusi dengan jari-jari radial."""
    if radius is None:
        radius = min(h, w) // 2 - 10
    cx, cy = w // 2, h // 2
    img = np.zeros((h, w), dtype=np.uint8)
    Y, X = np.mgrid[:h, :w]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    theta = np.arctan2(Y - cy, X - cx)

    # Setiap jari: alternasi hitam-putih
    sektor = (np.floor(theta / (np.pi / n_jari * 2) * 2) % 2).astype(bool)
    img[r <= radius] = np.where(sektor[r <= radius], 255, 0)

    # Lingkaran putih di tengah
    cv2.circle(img, (cx, cy), 10, 128, -1)
    return img


def buat_zone_plate(h=400, w=400,
                     frekuensi_max=0.5) -> np.ndarray:
    """Zone plate: frekuensi radial meningkat dari pusat ke tepi."""
    cx, cy = w // 2, h // 2
    Y, X = np.mgrid[:h, :w]
    r2 = ((X - cx)**2 + (Y - cy)**2).astype(np.float32)
    r2_max = cx**2 + cy**2
    fase = frekuensi_max * np.pi * r2 / r2_max
    img = (np.cos(fase) * 127.5 + 127.5).astype(np.uint8)
    return img


def buat_gradien(h=300, w=400) -> np.ndarray:
    """Rampa gradien linear: kiri=0, kanan=255."""
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))


def buat_gradien_2d(h=300, w=400) -> np.ndarray:
    """Gradien 2D: X=kecerahan, Y=saturasi imajiner."""
    X = np.linspace(0, 255, w, dtype=np.float32)
    Y = np.linspace(0, 255, h, dtype=np.float32)
    gx, gy = np.meshgrid(X, Y)
    return ((gx + gy) / 2).clip(0, 255).astype(np.uint8)


def tambah_noise_gaussian(img: np.ndarray, sigma=25) -> np.ndarray:
    """Tambahkan noise Gaussian (model noise sensor)."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def tambah_salt_pepper(img: np.ndarray, density=0.02) -> np.ndarray:
    """Tambahkan noise salt-and-pepper."""
    out = img.copy()
    n   = int(density * img.size)
    # Salt (putih)
    ys  = np.random.randint(0, img.shape[0], n)
    xs  = np.random.randint(0, img.shape[1], n)
    out[ys, xs] = 255
    # Pepper (hitam)
    ys  = np.random.randint(0, img.shape[0], n)
    xs  = np.random.randint(0, img.shape[1], n)
    out[ys, xs] = 0
    return out


def buat_tepi_slanted(h=300, w=400, sudut_deg=5.0) -> np.ndarray:
    """Slanted edge target — digunakan untuk mengukur MTF."""
    angle_rad = np.deg2rad(sudut_deg)
    Y, X = np.mgrid[:h, :w]
    batas = h // 2 + np.tan(angle_rad) * (X - w // 2)
    img = np.where(Y < batas, 255, 0).astype(np.uint8)
    return img


def buat_warna_macbeth() -> np.ndarray:
    """Tiruan color checker Macbeth 4×6 dengan 24 patch warna standar."""
    warna_bgr = [
        (68, 82, 115),  (130, 150, 194), (157, 122, 98),  (87, 108, 67),
        (133, 128, 177),(114, 189, 103), (72, 149, 214),  (214, 126, 44),
        (80, 91, 193),  (90, 60, 94),    (163, 204, 40),  (18, 163, 227),
        (173, 70, 27),  (67, 149, 70),   (50, 54, 175),   (22, 200, 238),
        (149, 86, 187), (164, 166, 0),   (240, 242, 243), (200, 200, 200),
        (160, 160, 160),(120, 120, 120), (80, 80, 80),    (40, 40, 40),
    ]
    patch_h, patch_w = 60, 60
    img = np.zeros((patch_h * 4, patch_w * 6, 3), dtype=np.uint8)
    for i, c in enumerate(warna_bgr):
        row, col = i // 6, i % 6
        y0, x0 = row * patch_h, col * patch_w
        img[y0:y0+patch_h, x0:x0+patch_w] = c
    return img


# ── Demo ──────────────────────────────────────────────────────────────────────
def demo_semua_pola():
    """Tampilkan dan simpan semua pola sintetis."""
    pola = {
        "Checkerboard":    buat_checkerboard(),
        "Siemens Star":    buat_siemens_star(),
        "Zone Plate":      buat_zone_plate(),
        "Gradien Linear":  buat_gradien(),
        "Gradien 2D":      buat_gradien_2d(),
        "Tepi Miring":     buat_tepi_slanted(),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (judul, img) in zip(axes.flat, pola.items()):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(judul, fontsize=12)
        ax.axis("off")
        # Simpan individual
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"19_{judul.lower().replace(' ','_')}.png"), img)

    plt.suptitle("Percobaan 19 — Pola Citra Sintetis", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "19_semua_pola.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


def demo_noise(img: np.ndarray = None):
    """Bandingkan jenis-jenis noise."""
    if img is None:
        img = buat_checkerboard()

    gambar = {
        "Asli":              img,
        "Noise Gaussian σ=15": tambah_noise_gaussian(img, 15),
        "Noise Gaussian σ=40": tambah_noise_gaussian(img, 40),
        "Salt & Pepper 1%":  tambah_salt_pepper(img, 0.01),
        "Salt & Pepper 5%":  tambah_salt_pepper(img, 0.05),
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, (judul, im) in zip(axes.flat, gambar.items()):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(judul, fontsize=9)
        ax.axis("off")

    plt.suptitle("Percobaan 19 — Jenis-jenis Noise", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "19_noise_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


def demo_color_checker():
    """Tampilkan color checker sintetis Macbeth."""
    img = buat_warna_macbeth()
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Color Checker Macbeth Sintetis (24 patch)")
    plt.axis("off")
    out = os.path.join(OUTPUT_DIR, "19_color_checker.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 19: CITRA SINTETIS UNTUK PENGUJIAN ALGORITMA")
    print("=" * 60)

    print("\n[1] Semua Pola Sintetis")
    demo_semua_pola()

    print("\n[2] Jenis-jenis Noise")
    demo_noise()

    print("\n[3] Color Checker Sintetis")
    demo_color_checker()

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
