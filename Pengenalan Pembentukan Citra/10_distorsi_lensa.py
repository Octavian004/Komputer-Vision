"""
==========================================================================
 PERCOBAAN 10 — DISTORSI LENSA
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memahami dan mensimulasikan distorsi lensa pada kamera nyata.
 Konsep  :
   - Distorsi RADIAL: garis lurus tampak melengkung karena cacat lensa.
       Barrel  (k1 < 0): gambar "menggelembung" ke luar → kamera wide-angle.
       Pincushion (k1 > 0): gambar "menciut" ke dalam → kamera telephoto.
   - Distorsi TANGENSIAL (p1, p2): lensa tidak sejajar bidang sensor.
   - Model OpenCV: x' = x(1+k1r²+k2r⁴+k3r⁶) + 2p1xy + p2(r²+2x²)
                   y' = y(1+k1r²+k2r⁴+k3r⁶) + p1(r²+2y²) + 2p2xy
   - Koreksi: cv2.undistort(img, K, distCoeffs) → gambar bebas distorsi.
 Catatan : Pada modul ini kita simulasikan distorsi; untuk koreksi nyata
           diperlukan kalibrasi kamera terlebih dahulu.
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


# ── Gambar grid sebagai referensi visualisasi distorsi ───────────────────────
def buat_gambar_grid(h: int = 400, w: int = 500, step: int = 40) -> np.ndarray:
    """
    Membuat gambar grid garis lurus berwarna putih di atas hitam.
    Grid mudah digunakan untuk memvisualisasikan distorsi karena kita
    tahu bahwa seharusnya semua garis adalah lurus.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (200, 200, 200), 1)
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (200, 200, 200), 1)
    # Tambah lingkaran konsentris sebagai referensi radial
    cx, cy = w // 2, h // 2
    for r in range(step, min(h, w) // 2, step):
        cv2.circle(img, (cx, cy), r, (80, 80, 200), 1)
    return img


def baca_gambar(nama: str) -> np.ndarray:
    path = os.path.join(IMAGE_DIR, nama)
    img  = cv2.imread(path)
    return img if img is not None else buat_gambar_grid()


# ── Simulasi distorsi menggunakan cv2.remap ───────────────────────────────────
def simulasi_distorsi(img: np.ndarray,
                      k1: float = 0.0, k2: float = 0.0,
                      p1: float = 0.0, p2: float = 0.0,
                      k3: float = 0.0) -> np.ndarray:
    """
    Mensimulasikan distorsi lensa menggunakan model OpenCV.
    Untuk SIMULASI, kita membuat matriks K sederhana lalu memanggil
    initUndistortRectifyMap dengan koefisien TERBALIK (undistort the ideal
    → distorted), kemudian remap.
    Parameter:
      k1, k2, k3 : koefisien distorsi radial
      p1, p2     : koefisien distorsi tangensial
    """
    h, w = img.shape[:2]
    # Matriks kamera sederhana (kamera ideal, focal length = w)
    f  = w
    K  = np.array([[f, 0, w/2],
                   [0, f, h/2],
                   [0, 0,  1 ]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # Buat peta remap untuk MENERAPKAN distorsi (kebalikan dari undistort)
    map1 = np.zeros((h, w), dtype=np.float32)
    map2 = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            # Normalisasi ke koordinat kamera
            xn = (x - w/2) / f
            yn = (y - h/2) / f
            r2 = xn**2 + yn**2
            r4 = r2**2
            r6 = r2**3
            radial = 1 + k1*r2 + k2*r4 + k3*r6
            xd = xn*radial + 2*p1*xn*yn + p2*(r2 + 2*xn**2)
            yd = yn*radial + p1*(r2 + 2*yn**2) + 2*p2*xn*yn
            # Kembali ke piksel
            map1[y, x] = xd * f + w/2
            map2[y, x] = yd * f + h/2

    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


# Versi cepat menggunakan vectorized NumPy
def simulasi_distorsi_cepat(img: np.ndarray,
                             k1: float = 0.0, k2: float = 0.0,
                             p1: float = 0.0, p2: float = 0.0,
                             k3: float = 0.0) -> np.ndarray:
    """Versi vectorized (lebih cepat) dari simulasi_distorsi."""
    h, w = img.shape[:2]
    f = float(w)
    xs = (np.arange(w, dtype=np.float32) - w/2) / f
    ys = (np.arange(h, dtype=np.float32) - h/2) / f
    xn, yn = np.meshgrid(xs, ys)
    r2 = xn**2 + yn**2
    r4 = r2**2
    r6 = r2**3
    radial = 1 + k1*r2 + k2*r4 + k3*r6
    xd = xn*radial + 2*p1*xn*yn + p2*(r2 + 2*xn**2)
    yd = yn*radial + p1*(r2 + 2*yn**2) + 2*p2*xn*yn
    map1 = (xd * f + w/2).astype(np.float32)
    map2 = (yd * f + h/2).astype(np.float32)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


# ── Demo 1: Barrel vs Pincushion ──────────────────────────────────────────────
def demo_barrel_pincushion(img):
    """Perbandingan barrel distortion (k1 < 0) dan pincushion (k1 > 0)."""
    configs = [
        ("Asli",          0.00),
        ("Barrel k1=−0.3", -0.30),
        ("Barrel k1=−0.6", -0.60),
        ("Pincushion k1=+0.3", 0.30),
        ("Pincushion k1=+0.6", 0.60),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, (title, k1) in zip(axes, configs):
        result = simulasi_distorsi_cepat(img, k1=k1)
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("Percobaan 10a — Barrel vs Pincushion Distortion", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "10_barrel_pincushion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Demo 2: Distorsi tangensial ───────────────────────────────────────────────
def demo_tangensial(img):
    """Pengaruh koefisien distorsi tangensial (p1, p2)."""
    configs = [
        ("Asli",             0.0,  0.0),
        ("p1=+0.2",          0.2,  0.0),
        ("p2=+0.2",          0.0,  0.2),
        ("p1=−0.2, p2=−0.2", -0.2, -0.2),
        ("k1+p1+p2",         -0.3, 0.1),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, (title, p1, p2) in zip(axes, configs):
        result = simulasi_distorsi_cepat(img, k1=0.0 if "k1" not in title else -0.3,
                                         p1=p1, p2=p2)
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("Percobaan 10b — Distorsi Tangensial", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "10_tangensial.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Demo 3: Pada gambar nyata ─────────────────────────────────────────────────
def demo_pada_gambar_nyata():
    """Menerapkan simulasi distorsi pada gambar foto nyata."""
    img_foto = baca_gambar("kota.jpg")
    if img_foto is None:
        img_foto = baca_gambar("foto_kucing.jpg")

    barrel    = simulasi_distorsi_cepat(img_foto, k1=-0.4)
    pincushion = simulasi_distorsi_cepat(img_foto, k1=0.3)
    combined  = simulasi_distorsi_cepat(img_foto, k1=-0.3, p1=0.1, p2=-0.05)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, im, title in zip(axes,
                              [img_foto, barrel, pincushion, combined],
                              ["Foto Asli", "Barrel\n(k1=−0.4)", "Pincushion\n(k1=+0.3)",
                               "Kombinasi\n(k1+p1+p2)"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 10c — Distorsi Lensa pada Foto Nyata", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "10_distorsi_foto_nyata.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 10: DISTORSI LENSA")
    print("=" * 60)

    grid = buat_gambar_grid()
    print(f"Gambar grid: {grid.shape[1]}×{grid.shape[0]}")

    print("\n[1] Barrel vs Pincushion (pada grid)")
    demo_barrel_pincushion(grid)

    print("\n[2] Distorsi Tangensial (pada grid)")
    demo_tangensial(grid)

    print("\n[3] Distorsi pada Foto Nyata")
    demo_pada_gambar_nyata()

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
