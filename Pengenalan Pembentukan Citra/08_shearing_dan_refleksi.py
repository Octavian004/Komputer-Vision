"""
==========================================================================
 PERCOBAAN 8 — SHEARING DAN REFLEKSI
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memahami shearing (geser miring) dan refleksi (cermin) sebagai
           transformasi geometri khusus dalam keluarga affine.
 Konsep  :
   - Shearing horizontal: M = [[1, sh, 0], [0, 1, 0]]  → piksel bergeser
     horizontal sebanding dengan posisi vertikalnya (atau sebaliknya).
   - Refleksi horizontal: cv2.flip(img, 1)   → cermin kiri-kanan.
   - Refleksi vertikal  : cv2.flip(img, 0)   → cermin atas-bawah.
   - Refleksi keduanya  : cv2.flip(img, -1)  → rotasi 180°.
   - Semua transformasi affine dapat dikomposisikan menjadi satu matriks.
 Catatan : Shearing tidak mempertahankan sudut antar garis (bukan rigid),
           tetapi mempertahankan area dan paralelisme.
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


# ── Utilitas ──────────────────────────────────────────────────────────────────
def baca_gambar(nama_file: str):
    """Membaca gambar dari IMAGE_DIR; jika tidak ada, buat sintetis."""
    path = os.path.join(IMAGE_DIR, nama_file)
    img = cv2.imread(path)
    if img is not None:
        return img
    # Gambar sintetis: kotak + garis diagonal sebagai referensi visual
    img = np.ones((300, 400, 3), dtype=np.uint8) * 220
    cv2.rectangle(img, (80, 60), (320, 240), (60, 100, 200), -1)
    cv2.line(img, (0, 0), (400, 300), (0, 180, 0), 3)
    cv2.putText(img, "M02", (140, 170), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (255, 255, 255), 5)
    return img


# ── Shearing ──────────────────────────────────────────────────────────────────
def shearing_horizontal(img, sh: float) -> np.ndarray:
    """
    Shearing horizontal: kolom bergeser sebanding dengan baris.
    Matriks affine 2×3:
      M = | 1  sh  0 |
          | 0   1  0 |
    sh > 0 → kanan; sh < 0 → kiri.
    """
    h, w = img.shape[:2]
    M = np.float32([[1, sh, 0], [0, 1, 0]])
    # Lebar output diperluas agar tidak ada bagian terpotong
    new_w = int(w + abs(sh) * h)
    # Tambah translasi jika sh negatif supaya gambar tidak keluar kiri
    if sh < 0:
        M[0, 2] = abs(sh) * h
    return cv2.warpAffine(img, M, (new_w, h))


def shearing_vertikal(img, sv: float) -> np.ndarray:
    """
    Shearing vertikal: baris bergeser sebanding dengan kolom.
    M = | 1  0  0 |
        | sv 1  0 |
    """
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [sv, 1, 0]])
    new_h = int(h + abs(sv) * w)
    if sv < 0:
        M[1, 2] = abs(sv) * w
    return cv2.warpAffine(img, M, (w, new_h))


def demo_shearing(img):
    """Perbandingan berbagai nilai shearing horizontal."""
    nilai_sh = [-0.4, -0.2, 0.0, 0.2, 0.4]
    hasil = [shearing_horizontal(img, sh) for sh in nilai_sh]

    fig, axes = plt.subplots(1, len(nilai_sh), figsize=(18, 4))
    for ax, im, sh in zip(axes, hasil, nilai_sh):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(f"sh = {sh:+.1f}")
        ax.axis("off")

    plt.suptitle("Percobaan 8a — Shearing Horizontal", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "08_shearing_horizontal.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")

    # Shearing vertikal
    fig2, axes2 = plt.subplots(1, len(nilai_sh), figsize=(18, 5))
    for ax, sv in zip(axes2, nilai_sh):
        hasil_v = shearing_vertikal(img, sv)
        ax.imshow(cv2.cvtColor(hasil_v, cv2.COLOR_BGR2RGB))
        ax.set_title(f"sv = {sv:+.1f}")
        ax.axis("off")

    plt.suptitle("Percobaan 8b — Shearing Vertikal", fontweight="bold")
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "08_shearing_vertikal.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out2}")


# ── Refleksi ──────────────────────────────────────────────────────────────────
def demo_refleksi(img):
    """
    Refleksi menggunakan cv2.flip():
      flipCode=1  → horizontal (cermin kiri-kanan).
      flipCode=0  → vertikal (cermin atas-bawah).
      flipCode=-1 → keduanya (rotasi 180°).
    Juga demonstrasi refleksi terhadap garis diagonal menggunakan warpAffine.
    """
    ref_h  = cv2.flip(img, 1)
    ref_v  = cv2.flip(img, 0)
    ref_hv = cv2.flip(img, -1)

    # Refleksi terhadap diagonal utama (transpose): x↔y
    h, w = img.shape[:2]
    M_diag = np.float32([[0, 1, 0], [1, 0, 0]])
    ref_diag = cv2.warpAffine(img, M_diag, (h, w))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, im, title in zip(axes,
                              [img, ref_h, ref_v, ref_hv, ref_diag],
                              ["Original", "Flip H\n(kiri-kanan)",
                               "Flip V\n(atas-bawah)", "Flip H+V\n(180°)",
                               "Diagonal"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 8c — Refleksi Gambar", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "08_refleksi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


# ── Aplikasi: efek artistik shear ────────────────────────────────────────────
def demo_efek_artistik(img):
    """Kombinasi shearing + refleksi untuk efek artistik sederhana."""
    ref_h = cv2.flip(img, 1)

    sh = 0.3
    h, w = img.shape[:2]
    M = np.float32([[1, sh, 0], [0, 1, 0]])
    kiri = cv2.warpAffine(img, M, (int(w + sh * h), h))

    M2 = np.float32([[1, -sh, sh * h], [0, 1, 0]])
    kanan = cv2.warpAffine(ref_h, M2, (int(w + sh * h), h))

    # Susun berdampingan
    gabung = np.hstack([
        cv2.resize(kiri,  (w, h)),
        cv2.resize(kanan, (w, h)),
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, im, title in zip(axes,
                              [img, ref_h, gabung],
                              ["Original", "Refleksi H", "Shear + Gabung"]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Percobaan 8d — Efek Artistik Shear & Refleksi", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "08_efek_artistik.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 8: SHEARING DAN REFLEKSI")
    print("=" * 60)

    img = baca_gambar("foto_kucing.jpg")
    h, w = img.shape[:2]
    print(f"Ukuran gambar: {w}×{h}")

    print("\n[1] Demo Shearing Horizontal & Vertikal")
    demo_shearing(img)

    print("\n[2] Demo Refleksi (Flip)")
    demo_refleksi(img)

    print("\n[3] Efek Artistik Kombinasi")
    demo_efek_artistik(img)

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
