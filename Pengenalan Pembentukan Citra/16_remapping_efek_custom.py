"""
==========================================================================
 PERCOBAAN 16 — REMAPPING DAN EFEK CUSTOM
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Membuat transformasi gambar fleksibel menggunakan cv2.remap()
           dengan custom lookup map untuk efek artistik dan koreksi optik.
 Konsep  :
   - cv2.remap(src, map_x, map_y, interpolation) : untuk setiap piksel
     output (x, y), nilai diambil dari src di posisi (map_x[y,x], map_y[y,x]).
   - map_x dan map_y adalah float32 array berukuran sama dengan output.
   - Dengan mendefinisikan map sendiri, kita bisa membuat TRANSFORMASI
     ARBITRARI: gelombang, pusaran, fisheye, kaca pembesar, dll.
   - Seluruh transformasi geometric normal (affine, perspektif) pun bisa
     diekspresikan sebagai remap; remap lebih fleksibel karena mendukung
     transformasi non-linear per-piksel.
 Catatan : Remap paling efisien ketika map dihitung sekali lalu dipakai
           berulang (efisien untuk video real-time).
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
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        for y in range(0, 300, 20):
            cv2.line(img, (0, y), (400, y), (180, 180, 180), 1)
        for x in range(0, 400, 20):
            cv2.line(img, (x, 0), (x, 300), (180, 180, 180), 1)
        cv2.circle(img, (200, 150), 100, (0, 140, 220), -1)
        cv2.putText(img, "REMAP", (110, 165), cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (255, 255, 255), 4)
    return img


def buat_mesh(img: np.ndarray):
    """Membuat mesh koordinat dasar (map identitas)."""
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    return map_x, map_y, h, w


# ── Efek-efek remap ───────────────────────────────────────────────────────────
def efek_gelombang_horizontal(img: np.ndarray, amplitudo=20, frekuensi=0.03):
    """Efek gelombang: setiap baris bergeser sinusoidal."""
    map_x, map_y, h, w = buat_mesh(img)
    map_x = map_x + amplitudo * np.sin(map_y * frekuensi)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def efek_gelombang_vertikal(img: np.ndarray, amplitudo=15, frekuensi=0.04):
    """Efek gelombang vertikal."""
    map_x, map_y, h, w = buat_mesh(img)
    map_y = map_y + amplitudo * np.sin(map_x * frekuensi)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def efek_pusaran(img: np.ndarray, kekuatan=0.5, radius_max=None):
    """Efek pusaran (swirl): piksel berputar sebanding dengan kedekatan ke pusat."""
    map_x, map_y, h, w = buat_mesh(img)
    if radius_max is None:
        radius_max = min(h, w) / 2.0
    cx, cy = w / 2.0, h / 2.0
    dx = map_x - cx
    dy = map_y - cy
    r  = np.sqrt(dx**2 + dy**2)
    sudut = np.arctan2(dy, dx) + kekuatan * (1 - r / radius_max).clip(0, 1)
    map_x_new = (cx + r * np.cos(sudut)).astype(np.float32)
    map_y_new = (cy + r * np.sin(sudut)).astype(np.float32)
    return cv2.remap(img, map_x_new, map_y_new, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def efek_kaca_pembesar(img: np.ndarray, kekuatan=2.0, radius=None):
    """Efek kaca pembesar di tengah gambar."""
    map_x, map_y, h, w = buat_mesh(img)
    if radius is None:
        radius = min(h, w) / 3.0
    cx, cy = w / 2.0, h / 2.0
    dx = map_x - cx
    dy = map_y - cy
    r  = np.sqrt(dx**2 + dy**2)
    # Di dalam radius: perkecil pertambahan (→ zoom in)
    mask = r < radius
    faktor = np.where(mask, r / (radius * kekuatan), 1.0)
    map_x_new = (cx + dx * faktor).astype(np.float32)
    map_y_new = (cy + dy * faktor).astype(np.float32)
    return cv2.remap(img, map_x_new, map_y_new, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def efek_fisheye(img: np.ndarray, kekuatan=0.5):
    """Efek fisheye/barrel: memperluas piksel di tepi."""
    map_x, map_y, h, w = buat_mesh(img)
    cx, cy = w / 2.0, h / 2.0
    xn  = (map_x - cx) / cx
    yn  = (map_y - cy) / cy
    r   = np.sqrt(xn**2 + yn**2)
    # Distorsi barrel
    r_dist = r * (1 + kekuatan * r**2)
    map_x_new = (cx + xn * r_dist / (r + 1e-9) * cx).astype(np.float32)
    map_y_new = (cy + yn * r_dist / (r + 1e-9) * cy).astype(np.float32)
    return cv2.remap(img, map_x_new, map_y_new, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def efek_cermin_kaleidoskop(img: np.ndarray):
    """Efek kaleidoskop: cermin 4 kuadran."""
    h, w = img.shape[:2]
    tl = img[:h//2, :w//2]
    tr = cv2.flip(tl, 1)
    top = np.hstack([tl, tr])
    bot = cv2.flip(top, 0)
    return np.vstack([top, bot])


# ── Demo: Semua efek ─────────────────────────────────────────────────────────
def demo_semua_efek(img):
    """Visualisasi semua efek remap dalam satu grid."""
    efek = [
        ("Asli",          img),
        ("Gelombang H",   efek_gelombang_horizontal(img)),
        ("Gelombang V",   efek_gelombang_vertikal(img)),
        ("Pusaran",       efek_pusaran(img, kekuatan=1.5)),
        ("Kaca Pembesar", efek_kaca_pembesar(img, kekuatan=2.5)),
        ("Fisheye",       efek_fisheye(img, kekuatan=0.4)),
        ("Kaleidoskop",   efek_cermin_kaleidoskop(img)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (title, im) in zip(axes.flat, efek):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    axes.flat[-1].axis("off")

    plt.suptitle("Percobaan 16 — Efek Custom menggunakan cv2.remap()", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "16_remapping_efek.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")

    # Simpan masing-masing efek
    for title, im in efek[1:]:
        fname = f"16_{title.lower().replace(' ', '_')}.png"
        cv2_out = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(cv2_out, im)
    print(f"  Efek individual disimpan di {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 16: REMAPPING DAN EFEK CUSTOM")
    print("=" * 60)

    img = baca_gambar("foto_kucing.jpg")
    print(f"Gambar: {img.shape[1]}×{img.shape[0]}")

    print("\n[1] Demo Semua Efek Remap")
    demo_semua_efek(img)

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
