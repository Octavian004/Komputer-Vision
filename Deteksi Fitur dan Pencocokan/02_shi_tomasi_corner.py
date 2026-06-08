"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
Topik   : Shi-Tomasi Corner Detection
=============================================================================
Shi-Tomasi (1994) merupakan penyempurnaan Harris Corner.

Kriteria seleksi:
  Harris   : R = det(M) - k*trace(M)^2   -- corner jika R > thr
  Shi-Tomasi: R_ST = min(lambda1, lambda2) -- corner jika min-eigenvalue > thr

Mengapa min-eigenvalue lebih baik?
  Karena fitur yang stabil untuk pelacakan (optical flow) harus memiliki
  perubahan intensitas yang CUKUP BESAR di SEMUA arah gerak, dan hal itu
  dijamin ketika KEDUA eigenvalue (terutama yang terkecil) bernilai besar.

Implementasi di OpenCV: cv2.goodFeaturesToTrack()
  - gray       : gambar grayscale uint8
  - maxCorners : jumlah corner maksimum yang dikembalikan
  - qualityLevel: threshold relatif (0-1) terhadap nilai R_ST terbesar
  - minDistance : jarak minimal antar corner (piksel)

Untuk presisi sub-piksel: cv2.cornerSubPix()
  - corners_rough -> corners_refined dengan presisi floating-point
=============================================================================
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Direktori kerja
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fungsi utilitas
# ---------------------------------------------------------------------------

def muat_gambar(filename="kota.jpg"):
    """
    Memuat gambar dari IMAGE_DIR. Jika tidak ada, gambar sintetis dibuat
    dengan banyak sudut (kotak, segitiga, checker) sebagai fallback.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(filepath):
        img = cv2.imread(filepath)
        if img is not None:
            print(f"[INFO] Gambar dimuat: {filepath}")
            return img

    print(f"[INFO] '{filename}' tidak ditemukan -- membuat gambar sintetis.")
    h, w = 480, 640
    kanvas = np.ones((h, w, 3), dtype=np.uint8) * 40

    # Banyak kotak kecil -- menghasilkan grid corner
    for r in range(4):
        for c in range(5):
            x0 = 60 + c * 110
            y0 = 60 + r * 90
            cv2.rectangle(kanvas, (x0, y0), (x0+70, y0+55), (200, 200, 200), 2)

    # Segitiga besar
    pts = np.array([[320, 30], [200, 220], [440, 220]], np.int32)
    cv2.polylines(kanvas, [pts], True, (220, 220, 220), 2)

    # Diagonal
    cv2.line(kanvas, (0, h), (w, 0), (150, 150, 150), 1)
    noise = np.random.randint(0, 12, kanvas.shape, dtype=np.uint8)
    kanvas = cv2.add(kanvas, noise)
    return kanvas


def bgr_ke_rgb(img):
    """Konversi BGR -> RGB untuk Matplotlib."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gambar_corners(img_bgr, corners, warna=(0, 255, 0), radius=3):
    """
    Gambar titik corner pada salinan gambar.
    corners : array (N, 1, 2) float32 dari goodFeaturesToTrack
    """
    vis = img_bgr.copy()
    if corners is not None:
        for pt in corners:
            x, y = pt.ravel().astype(int)
            cv2.circle(vis, (x, y), radius, warna, -1)
    return vis


# ---------------------------------------------------------------------------
# Demo 1 -- Shi-Tomasi dasar
# ---------------------------------------------------------------------------

def demo_shi_tomasi_dasar(img_bgr):
    """
    Demo dasar cv2.goodFeaturesToTrack() dengan parameter default.
    Menampilkan gambar asli, grayscale, dan hasil deteksi corner.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Deteksi Shi-Tomasi -- parameter standar
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners   = 200,     # maksimum 200 corner dikembalikan
        qualityLevel = 0.01,    # min kualitas = 1% dari corner terbaik
        minDistance  = 10       # antar corner minimal 10 piksel
    )

    vis = gambar_corners(img_bgr, corners, warna=(0, 255, 0))
    n   = len(corners) if corners is not None else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Demo 1 -- Shi-Tomasi Corner Detection Dasar",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(bgr_ke_rgb(img_bgr)); axes[0].set_title("Asli"); axes[0].axis("off")
    axes[1].imshow(gray, cmap="gray");   axes[1].set_title("Grayscale"); axes[1].axis("off")
    axes[2].imshow(bgr_ke_rgb(vis))
    axes[2].set_title(
        f"Shi-Tomasi Corners\nmaxCorners=200, qLevel=0.01\n{n} corner terdeteksi",
        fontsize=9)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_shi_tomasi_dasar.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 1] 02_shi_tomasi_dasar.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 2 -- Pengaruh maxCorners
# ---------------------------------------------------------------------------

def demo_pengaruh_max_corners(img_bgr):
    """
    Pengaruh parameter maxCorners terhadap jumlah dan distribusi corner.

    goodFeaturesToTrack pertama menghitung R_ST untuk semua piksel, lalu
    memilih puncak-puncak terbesar sebanyak maxCorners yang masih memenuhi
    qualityLevel dan minDistance.

    Corner yang dipilih selalu yang memiliki nilai R_ST terbesar --
    sehingga menambah maxCorners berarti corner dengan kualitas lebih rendah
    mulai ditambahkan.
    """
    gray         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    daftar_max   = [10, 50, 200, 500]     # variasi maxCorners
    warna_titik  = (0, 255, 0)

    fig, axes = plt.subplots(1, len(daftar_max), figsize=(16, 4))
    fig.suptitle("Demo 2 -- Pengaruh maxCorners",
                 fontsize=12, fontweight="bold")

    for ax, mc in zip(axes, daftar_max):
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=mc,
                                          qualityLevel=0.01, minDistance=5)
        vis = gambar_corners(img_bgr, corners, warna=warna_titik)
        n   = len(corners) if corners is not None else 0
        ax.imshow(bgr_ke_rgb(vis))
        ax.set_title(f"maxCorners={mc}\n{n} terdeteksi", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_shi_tomasi_max_corners.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 2] 02_shi_tomasi_max_corners.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 3 -- Pengaruh qualityLevel
# ---------------------------------------------------------------------------

def demo_pengaruh_quality_level(img_bgr):
    """
    Pengaruh qualityLevel terhadap seleksi corner.

    qualityLevel adalah threshold RELATIF terhadap nilai R_ST maksimum:
      threshold_abs = qualityLevel * max(R_ST)
    Corner dengan R_ST < threshold_abs dibuang.

    qualityLevel kecil (0.01) -> ambang rendah -> banyak corner (juga noise)
    qualityLevel besar (0.30) -> ambang tinggi -> hanya corner tajam
    """
    gray         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    daftar_ql    = [0.01, 0.05, 0.10, 0.30]
    warna_titik  = (255, 100, 0)           # oranye

    fig, axes = plt.subplots(1, len(daftar_ql), figsize=(16, 4))
    fig.suptitle("Demo 3 -- Pengaruh qualityLevel",
                 fontsize=12, fontweight="bold")

    for ax, ql in zip(axes, daftar_ql):
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=500,
                                          qualityLevel=ql, minDistance=5)
        vis = gambar_corners(img_bgr, corners, warna=warna_titik)
        n   = len(corners) if corners is not None else 0
        ax.imshow(bgr_ke_rgb(vis))
        ax.set_title(f"qualityLevel={ql}\n{n} corner", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_shi_tomasi_quality_level.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 3] 02_shi_tomasi_quality_level.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 4 -- Sub-piksel refinement dengan cornerSubPix
# ---------------------------------------------------------------------------

def demo_subpiksel_refinement(img_bgr):
    """
    cv2.cornerSubPix() menyempurnakan lokasi corner ke presisi sub-piksel.

    Algoritma bekerja dengan mengiterasi posisi corner menggunakan kriteria
    ortogonalitas: vektor dari corner ke piksel sekitar harus tegak lurus
    dengan gradien gambar di piksel tersebut.

    Visualisasi: perbandingan posisi kasar (integer) vs posisi halus (float).
    Kita tampilkan zoom pada salah satu area corner untuk melihat perbedaan.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Deteksi corner kasar (integer pixels)
    corners_kasar = cv2.goodFeaturesToTrack(
        gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    if corners_kasar is None or len(corners_kasar) == 0:
        print("[Demo 4] Tidak ada corner terdeteksi -- skip.")
        return

    # Kriteria penghentian iterasi: akurasi 0.001 piksel atau 40 iterasi
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    corners_halus = corners_kasar.copy()
    # winSize: setengah ukuran window pencarian (11 = 23x23 total)
    cv2.cornerSubPix(gray, corners_halus,
                     winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)

    # Hitung perpindahan posisi setelah refinement
    delta = np.linalg.norm(corners_halus - corners_kasar, axis=2).flatten()

    # Visualisasi: gambar kasar (merah) dan halus (hijau) berdampingan
    vis_k = img_bgr.copy()
    vis_h = img_bgr.copy()
    for pt in corners_kasar:
        x, y = pt.ravel().astype(int)
        cv2.circle(vis_k, (x, y), 4, (0, 0, 255), -1)
    for pt in corners_halus:
        x, y = int(round(pt[0][0])), int(round(pt[0][1]))
        cv2.circle(vis_h, (x, y), 4, (0, 255, 0), -1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Demo 4 -- Sub-Piksel Refinement cornerSubPix",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(bgr_ke_rgb(vis_k))
    axes[0].set_title(f"Posisi Kasar (integer)\n{len(corners_kasar)} corner, merah",
                      fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(bgr_ke_rgb(vis_h))
    axes[1].set_title("Posisi Halus (sub-piksel)\nhijau", fontsize=9)
    axes[1].axis("off")

    # Histogram perpindahan
    axes[2].hist(delta, bins=25, color="steelblue", edgecolor="black")
    axes[2].set_xlabel("Perpindahan (piksel)")
    axes[2].set_ylabel("Jumlah corner")
    axes[2].set_title(
        f"Distribusi Pergeseran\nrata-rata={delta.mean():.4f} piksel", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_shi_tomasi_subpixel.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 4] 02_shi_tomasi_subpixel.png disimpan.")
    print(f"         Rata-rata pergeseran sub-piksel: {delta.mean():.4f} px")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Modul 04 -- Shi-Tomasi Corner Detection")
    print("=" * 60)
    img = muat_gambar("kota.jpg")
    demo_shi_tomasi_dasar(img)
    demo_pengaruh_max_corners(img)
    demo_pengaruh_quality_level(img)
    demo_subpiksel_refinement(img)
    print("\n[SELESAI] Semua demo Shi-Tomasi telah dijalankan.")
    print(f"Output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
