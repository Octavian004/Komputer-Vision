"""
==========================================================================
 PERCOBAAN 18 — TRANSFORMASI INTENSITAS (POINT OPERATIONS)
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Mempelajari transformasi intensitas piksel-per-piksel yang umum
           digunakan untuk meningkatkan kontras dan visibilitas fitur gambar.
 Konsep  :
   - POINT OPERATION: I_out[y,x] = T(I_in[y,x]) — output hanya bergantung
     pada nilai satu piksel, tidak pada tetangga (berbeda dari filter).
   - LOG TRANSFORM: S = c * log(1 + r). Memperluas rentang nilai gelap,
     mengompresi rentang nilai terang → baik untuk Fourier spectrum.
   - POWER-LAW (Gamma): S = c * r^γ — lihat percobaan 17.
   - PIECEWISE LINEAR: fungsi segmen garis — bisa dirancang bebas, misalnya
     untuk meregangkan rentang kontras tertentu (contrast stretching).
   - HISTOGRAM EQUALIZATION: redistribusi intensitas agar histogram rata →
     kontras global meningkat; CLAHE = ekualisasi lokal (adaptif).
   - NEGATIVE: S = 255 - r — membalik intensitas, berguna untuk X-ray.
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


def baca_gambar(nama: str) -> np.ndarray:
    path = os.path.join(IMAGE_DIR, nama)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Gambar sintetis: gradien diagonal + noise
        xx, yy = np.meshgrid(np.arange(400), np.arange(300))
        img = ((xx + yy) / (399 + 299) * 255).astype(np.uint8)
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ── Fungsi transformasi ───────────────────────────────────────────────────────
def transform_negatif(img: np.ndarray) -> np.ndarray:
    """Transformasi negatif: membalik intensitas."""
    return 255 - img


def transform_log(img: np.ndarray, c: float = None) -> np.ndarray:
    """Log transform: S = c * log(1 + r). Memperjelas detail gelap."""
    if c is None:
        c = 255 / np.log(1 + 255)
    r = img.astype(np.float32)
    s = c * np.log(1 + r)
    return np.clip(s, 0, 255).astype(np.uint8)


def transform_power_law(img: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """Power-law: S = c * r^gamma."""
    r = img.astype(np.float32) / 255.0
    s = c * np.power(r, gamma) * 255.0
    return np.clip(s, 0, 255).astype(np.uint8)


def transform_piecewise(img: np.ndarray,
                         r1=80, s1=30,
                         r2=180, s2=230) -> np.ndarray:
    """Piecewise linear contrast stretching antara (r1,s1) dan (r2,s2)."""
    lut = np.zeros(256, dtype=np.uint8)
    for r in range(256):
        if r <= r1:
            lut[r] = int(s1 / max(r1, 1) * r)
        elif r <= r2:
            lut[r] = int(s1 + (s2 - s1) / max(r2 - r1, 1) * (r - r1))
        else:
            lut[r] = int(s2 + (255 - s2) / max(255 - r2, 1) * (r - r2))
    return cv2.LUT(img, lut)


def transform_histogram_eq(img: np.ndarray) -> np.ndarray:
    """Histogram equalization global."""
    return cv2.equalizeHist(img)


def transform_clahe(img: np.ndarray,
                    clip_limit=2.0, tile_size=(8, 8)) -> np.ndarray:
    """CLAHE — ekualisasi histogram adaptif dengan batasan klip."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img)


# ── Demo ──────────────────────────────────────────────────────────────────────
def demo_perbandingan(img):
    """Bandingkan semua transformasi intensitas dalam satu tampilan."""
    gambar = {
        "Asli":             img,
        "Negatif":          transform_negatif(img),
        "Log Transform":    transform_log(img),
        "Power γ=0.4":      transform_power_law(img, 0.4),
        "Power γ=2.5":      transform_power_law(img, 2.5),
        "Piecewise":        transform_piecewise(img),
        "Hist. Eq.":        transform_histogram_eq(img),
        "CLAHE":            transform_clahe(img),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (judul, im) in zip(axes.flat, gambar.items()):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(judul, fontsize=11)
        ax.axis("off")

    plt.suptitle("Percobaan 18 — Perbandingan Transformasi Intensitas", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "18_perbandingan_transformasi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


def demo_histogram_perbandingan(img):
    """Histogram sebelum/sesudah ekualisasi global vs CLAHE."""
    eq   = transform_histogram_eq(img)
    clahe = transform_clahe(img)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (judul, g) in enumerate([("Asli", img),
                                       ("Hist. Eq.", eq),
                                       ("CLAHE", clahe)]):
        axes[0, col].imshow(g, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(judul); axes[0, col].axis("off")
        axes[1, col].hist(g.ravel(), bins=64, color="#e74c3c", alpha=0.8)
        axes[1, col].set_xlabel("Intensitas"); axes[1, col].set_xlim(0, 255)

    plt.suptitle("Percobaan 18 — Histogram Sebelum & Sesudah Ekualisasi", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "18_histogram_ekualisasi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


def plot_kurva_transformasi():
    """Plot kurva T(r) untuk setiap transformasi."""
    r = np.arange(256, dtype=np.float32)

    def to_curve(fn):
        dummy = r.astype(np.uint8).reshape(1, 256)
        return fn(dummy).reshape(256).astype(float)

    kurva = {
        "Negatif":       255 - r,
        "Log":           255 / np.log(256) * np.log(1 + r),
        "Power γ=0.4":   (r / 255)**0.4 * 255,
        "Power γ=2.5":   (r / 255)**2.5 * 255,
        "Piecewise":     to_curve(transform_piecewise),
    }
    warna = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r, r, "k--", lw=1, label="Identitas")
    for (nama, y), c in zip(kurva.items(), warna):
        ax.plot(r, np.clip(y, 0, 255), color=c, lw=2, label=nama)
    ax.set_xlabel("Intensitas Input r"); ax.set_ylabel("Intensitas Output s")
    ax.set_title("Kurva Transformasi Intensitas")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, "18_kurva_transformasi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 18: TRANSFORMASI INTENSITAS (POINT OPERATIONS)")
    print("=" * 60)

    img = baca_gambar("foto.jpg")
    print(f"Gambar: {img.shape[1]}×{img.shape[0]}")

    print("\n[1] Kurva Transformasi")
    plot_kurva_transformasi()

    print("\n[2] Perbandingan Semua Transformasi")
    demo_perbandingan(img)

    print("\n[3] Histogram Global vs CLAHE")
    demo_histogram_perbandingan(img)

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
