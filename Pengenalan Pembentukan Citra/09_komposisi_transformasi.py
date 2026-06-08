"""
==========================================================================
 PERCOBAAN 9 — KOMPOSISI TRANSFORMASI
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memahami bagaimana transformasi geometri dapat digabungkan
           (dikomposis) melalui perkalian matriks homogen 3×3.
 Konsep  :
   - Setiap transformasi affine direpresentasikan sebagai matriks 3×3 homogen.
   - Komposisi: M_total = M_n · M_(n-1) · ... · M_1  (kanan ke kiri).
   - Urutan SANGAT penting: rotasi lalu translasi ≠ translasi lalu rotasi.
   - cv2.warpAffine(img, M[:2], dsize) menerapkan matriks 2×3 ke gambar.
 Catatan : Operasi matriks 3×3 homogen memudahkan komposisi transformasi
           yang rumit sekalipun hanya dengan perkalian matriks biasa.
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


# ── Utilitas matriks homogen 3×3 ─────────────────────────────────────────────
def mat_translasi(tx: float, ty: float) -> np.ndarray:
    """Matriks homogen 3×3 untuk translasi (tx, ty)."""
    return np.float64([[1, 0, tx],
                        [0, 1, ty],
                        [0, 0,  1]])


def mat_rotasi(sudut_deg: float, cx: float = 0, cy: float = 0) -> np.ndarray:
    """
    Matriks homogen 3×3 untuk rotasi di sekitar titik (cx, cy).
    Positif = berlawanan jarum jam (konvensi matematika).
    """
    rad = np.radians(sudut_deg)
    cos, sin = np.cos(rad), np.sin(rad)
    # Rotasi di sekitar origin:
    R = np.float64([[cos, -sin, 0],
                     [sin,  cos, 0],
                     [  0,    0, 1]])
    # Komposisi: translasi ke origin → putar → translasi kembali
    T1 = mat_translasi(-cx, -cy)
    T2 = mat_translasi( cx,  cy)
    return T2 @ R @ T1


def mat_scaling(sx: float, sy: float, cx: float = 0, cy: float = 0) -> np.ndarray:
    """Matriks homogen 3×3 untuk scaling di sekitar titik (cx, cy)."""
    S = np.float64([[sx,  0, 0],
                     [ 0, sy, 0],
                     [ 0,  0, 1]])
    T1 = mat_translasi(-cx, -cy)
    T2 = mat_translasi( cx,  cy)
    return T2 @ S @ T1


def mat_shear(sh: float) -> np.ndarray:
    """Matriks homogen 3×3 untuk shearing horizontal."""
    return np.float64([[1, sh, 0],
                        [0,  1, 0],
                        [0,  0, 1]])


def apply_transform(img: np.ndarray, M3x3: np.ndarray) -> np.ndarray:
    """Menerapkan matriks homogen 3×3 ke gambar menggunakan warpAffine."""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M3x3[:2].astype(np.float32), (w, h))


# ── Baca gambar ───────────────────────────────────────────────────────────────
def baca_gambar(nama: str) -> np.ndarray:
    path = os.path.join(IMAGE_DIR, nama)
    img  = cv2.imread(path)
    if img is None:
        img = np.ones((300, 400, 3), dtype=np.uint8) * 200
        cv2.rectangle(img, (100, 80), (300, 220), (80, 120, 200), -1)
        cv2.putText(img, "M02-P9", (110, 165), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), 3)
    return img


# ── Demo 1: Urutan transformasi ───────────────────────────────────────────────
def demo_urutan_transformasi(img):
    """
    Membuktikan bahwa rotasi → translasi ≠ translasi → rotasi.
    Ini adalah sifat non-komutatif perkalian matriks.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    R = mat_rotasi(30, cx, cy)
    T = mat_translasi(80, 50)

    RT = apply_transform(img, R @ T)   # (rotasi di posisi baru) lalu rotasi lagi → salah baca: T dulu lalu R
    TR = apply_transform(img, T @ R)   # (R dulu lalu T): translasi setelah rotasi
    R_only = apply_transform(img, R)
    T_only = apply_transform(img, T)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, im, title in zip(axes,
                              [R_only, T_only, RT, TR],
                              ["Rotasi 30°", "Translasi (+80,+50)", "T → R", "R → T"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 9a — Urutan Transformasi (non-komutatif)", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "09_urutan_transformasi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  R@T == T@R ? {np.allclose(R @ T, T @ R)}  ← seharusnya False")
    print(f"[SIMPAN] {out}")


# ── Demo 2: Komposisi kompleks ────────────────────────────────────────────────
def demo_komposisi_kompleks(img):
    """
    Membuat satu matriks tunggal dari rangkaian: scale → rotasi → translasi.
    Hasil ekuivalen dengan menerapkan tiap transformasi satu per satu.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    S  = mat_scaling(0.7, 0.7, cx, cy)
    R  = mat_rotasi(45, cx, cy)
    T  = mat_translasi(50, -30)

    # Komposisi: S dulu, lalu R, lalu T  →  M = T @ R @ S
    M_gabung = T @ R @ S

    # Terapkan satu per satu (referensi)
    step1 = apply_transform(img, S)
    step2 = apply_transform(step1, R)
    step3 = apply_transform(step2, T)

    # Terapkan sekaligus
    sekaligus = apply_transform(img, M_gabung)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, im, title in zip(axes,
                              [img, step1, step2, step3, sekaligus],
                              ["Original", "Scale 0.7×", "+Rotasi 45°",
                               "+Translasi", "Semua sekaligus"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 9b — Komposisi S → R → T", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "09_komposisi_kompleks.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  step3 ≈ sekaligus ? {np.allclose(step3.astype(float), sekaligus.astype(float), atol=2)}")
    print(f"[SIMPAN] {out}")


# ── Demo 3: Animasi komposisi ─────────────────────────────────────────────────
def demo_komposisi_bertahap(img):
    """
    Visualisasi komposisi bertahap: transformasi dasar, lalu gabungan semua.
    Memperlihatkan bagaimana pipeline transformasi dibangun langkah demi langkah.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    langkah = [
        ("Original",   np.eye(3, dtype=np.float64)),
        ("Rotasi 20°", mat_rotasi(20, cx, cy)),
        ("+ Shear 0.2", mat_shear(0.2) @ mat_rotasi(20, cx, cy)),
        ("+ Scale 0.8", mat_scaling(0.8, 0.8, cx, cy) @ mat_shear(0.2) @ mat_rotasi(20, cx, cy)),
        ("+ Geser",    mat_translasi(60, 30) @ mat_scaling(0.8, 0.8, cx, cy) @ mat_shear(0.2) @ mat_rotasi(20, cx, cy)),
    ]

    fig, axes = plt.subplots(1, len(langkah), figsize=(20, 4))
    for ax, (title, M) in zip(axes, langkah):
        result = apply_transform(img, M)
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 9c — Komposisi Bertahap", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "09_komposisi_bertahap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 9: KOMPOSISI TRANSFORMASI")
    print("=" * 60)

    img = baca_gambar("foto_kucing.jpg")

    print("\n[1] Urutan Transformasi (non-komutatif)")
    demo_urutan_transformasi(img)

    print("\n[2] Komposisi Skala → Rotasi → Translasi")
    demo_komposisi_kompleks(img)

    print("\n[3] Komposisi Bertahap")
    demo_komposisi_bertahap(img)

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
