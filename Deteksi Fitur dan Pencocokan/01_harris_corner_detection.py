"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
Topik   : Harris Corner Detection
=============================================================================
Harris Corner Detection adalah algoritma klasik untuk mendeteksi sudut (corner)
dalam citra. Ide dasarnya: sebuah titik disebut SUDUT jika pergeseran jendela
kecil ke segala arah menghasilkan perubahan intensitas yang besar.

Matriks struktur M (second-moment matrix):
    M = sum w(x,y) * [ Ix^2   Ix*Iy ]
                     [ Ix*Iy  Iy^2  ]
di mana Ix, Iy adalah gradien gambar.

Eigenvalue M:
  - lambda1 ~ 0, lambda2 ~ 0  -> daerah datar (flat region)
  - lambda1 >> 0, lambda2 ~ 0  -> tepi (edge)
  - lambda1 >> 0, lambda2 >> 0 -> SUDUT (corner) -- yang ingin terdeteksi

Fungsi respons Harris:
    R = det(M) - k * trace(M)^2
    R = lambda1*lambda2 - k*(lambda1+lambda2)^2
  - R > threshold besar -> sudut
  - R < 0               -> tepi
  - |R| kecil           -> daerah datar
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
    Memuat gambar dari IMAGE_DIR.  Jika file tidak ditemukan, gambar sintetis
    (kotak, segitiga, diagonal) dibuat sebagai fallback.
    Mengembalikan gambar BGR uint8.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(filepath):
        img = cv2.imread(filepath)
        if img is not None:
            print(f"[INFO] Gambar dimuat: {filepath}")
            return img

    # Fallback: gambar sintetis dengan banyak sudut tajam
    print(f"[INFO] '{filename}' tidak ditemukan -- membuat gambar sintetis.")
    h, w = 480, 640
    kanvas = np.ones((h, w, 3), dtype=np.uint8) * 30   # latar gelap

    # Kotak -- menghasilkan empat sudut 90 derajat
    cv2.rectangle(kanvas, (80, 80),   (240, 200), (200, 200, 200), 2)
    cv2.rectangle(kanvas, (300, 120), (500, 320), (180, 180, 180), 2)
    # Segitiga -- sudut miring
    pts = np.array([[160, 300], [80, 420], [250, 420]], np.int32)
    cv2.polylines(kanvas, [pts], True, (220, 220, 220), 2)
    # Garis diagonal
    cv2.line(kanvas, (0, 0), (w, h), (160, 160, 160), 1)
    cv2.line(kanvas, (w, 0), (0, h), (160, 160, 160), 1)
    # Lingkaran (tidak membentuk sudut tajam -- sebagai kontrol)
    cv2.circle(kanvas, (520, 380), 60, (200, 200, 200), 2)
    # Tambahkan noise ringan agar lebih realistis
    noise = np.random.randint(0, 15, kanvas.shape, dtype=np.uint8)
    kanvas = cv2.add(kanvas, noise)
    return kanvas


def bgr_ke_rgb(img):
    """Konversi BGR (OpenCV) ke RGB (Matplotlib)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tambah_judul(ax, judul):
    """Judul subplot seragam."""
    ax.set_title(judul, fontsize=9, pad=4)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Demo 1 -- Harris Corner Detection dasar
# ---------------------------------------------------------------------------

def demo_harris_dasar(img_bgr):
    """
    Langkah-langkah dasar Harris Corner Detection:
    1. Konversi ke grayscale
    2. cv2.cornerHarris(gray_f32, blockSize, ksize, k)
       - blockSize : ukuran neighbourhood untuk menghitung M
       - ksize     : ukuran kernel Sobel (harus ganjil)
       - k         : konstanta sensitivity (biasanya 0.04-0.06)
    3. Dilasi response map agar corner lebih terlihat
    4. Threshold: piksel dengan R > thr*max(R) ditandai merah
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f  = np.float32(gray)            # cornerHarris perlu float32

    block_size = 2      # ukuran jendela sekitar piksel
    ksize      = 3      # ukuran kernel Sobel untuk gradien
    k          = 0.04   # nilai umum: 0.04 -- 0.06
    threshold  = 0.01   # 1% dari max respons

    # Hitung fungsi respons Harris R
    R = cv2.cornerHarris(gray_f, block_size, ksize, k)
    # Dilasi supaya lokasi corner lebih tebal / terlihat
    R_dil = cv2.dilate(R, None)

    # Tandai corner dengan warna merah
    vis = img_bgr.copy()
    vis[R_dil > threshold * R_dil.max()] = [0, 0, 255]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Demo 1 -- Harris Corner Detection Dasar",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(bgr_ke_rgb(img_bgr))
    tambah_judul(axes[0], "Gambar Asli")

    axes[1].imshow(gray, cmap="gray")
    tambah_judul(axes[1], "Grayscale")

    im = axes[2].imshow(R, cmap="hot")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    tambah_judul(axes[2], "Respons R (raw)\nmerah = sudut besar")

    n_corner = int((R_dil > threshold * R_dil.max()).sum())
    axes[3].imshow(bgr_ke_rgb(vis))
    tambah_judul(axes[3],
        f"Sudut Terdeteksi (merah)\nthresh={threshold}, k={k}\n~{n_corner} piksel")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_harris_dasar.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 1] 01_harris_dasar.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 2 -- Pengaruh parameter k
# ---------------------------------------------------------------------------

def demo_pengaruh_k(img_bgr):
    """
    Pengaruh konstanta k terhadap jumlah corner terdeteksi.

    k kecil (0.01) -> lebih sensitif, edge pun terdeteksi sebagai corner
    k besar (0.25) -> sangat selektif, hanya sudut sangat tajam yang lolos

    Secara matematis: R = det(M) - k*trace(M)^2
    Semakin besar k, semakin besar penalti terhadap trace (energi total),
    sehingga hanya titik dengan lambda1 dan lambda2 sama-sama besar
    (sudut sejati) yang menghasilkan R positif besar.
    """
    gray_f    = np.float32(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    daftar_k  = [0.01, 0.04, 0.10, 0.25]   # range nilai k yang diuji
    threshold = 0.01

    fig, axes = plt.subplots(1, len(daftar_k), figsize=(16, 4))
    fig.suptitle("Demo 2 -- Pengaruh Parameter k pada Harris",
                 fontsize=12, fontweight="bold")

    for ax, k_val in zip(axes, daftar_k):
        R     = cv2.cornerHarris(gray_f, 2, 3, k_val)
        R_dil = cv2.dilate(R, None)
        vis   = img_bgr.copy()
        mask  = R_dil > threshold * R_dil.max()
        vis[mask] = [0, 0, 255]             # tandai dengan merah
        n     = int(mask.sum())
        ax.imshow(bgr_ke_rgb(vis))
        ax.set_title(f"k = {k_val}\n{n} piksel corner", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_harris_pengaruh_k.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 2] 01_harris_pengaruh_k.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 3 -- Heatmap Fungsi Respons R
# ---------------------------------------------------------------------------

def demo_heatmap_respons(img_bgr):
    """
    Visualisasi komponen-komponen penyusun fungsi respons Harris R:
      R = det(M) - k * trace(M)^2
      det(M)   = Ix2*Iy2 - IxIy^2     (produk eigenvalue: lambda1*lambda2)
      trace(M) = Ix2 + Iy2             (jumlah eigenvalue: lambda1+lambda2)

    Kita tampilkan: gradien Ix, Iy, det(M), trace(M), dan R akhir.
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)

    # Hitung gradien dengan Sobel
    Ix  = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    Iy  = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    # Haluskan elemen-elemen M dengan Gaussian agar lebih stabil
    sigma = 2
    ks    = 2 * int(3 * sigma) + 1
    Ix2   = cv2.GaussianBlur(Ix * Ix, (ks, ks), sigma)
    Iy2   = cv2.GaussianBlur(Iy * Iy, (ks, ks), sigma)
    IxIy  = cv2.GaussianBlur(Ix * Iy, (ks, ks), sigma)

    k       = 0.04
    det_M   = Ix2 * Iy2 - IxIy ** 2        # lambda1 * lambda2
    trace_M = Ix2 + Iy2                     # lambda1 + lambda2
    R       = det_M - k * (trace_M ** 2)    # fungsi respons Harris

    items = [
        (gray,   "gray",   "Grayscale"),
        (Ix,     "RdBu",   "Gradien Ix (horizontal)"),
        (Iy,     "RdBu",   "Gradien Iy (vertikal)"),
        (det_M,  "hot",    "det(M) = lambda1*lambda2"),
        (R,      "RdBu_r", "Respons R\nbiru=tepi, merah=sudut"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Demo 3 -- Heatmap Komponen Fungsi Respons R",
                 fontsize=12, fontweight="bold")

    for ax, (data, cmap, judul) in zip(axes, items):
        im = ax.imshow(data, cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(judul, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_harris_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 3] 01_harris_heatmap.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 4 -- Perbandingan Harris vs Shi-Tomasi
# ---------------------------------------------------------------------------

def demo_harris_vs_shi_tomasi(img_bgr):
    """
    Perbandingan dua kriteria deteksi corner:

    Harris   : R = det(M) - k*trace(M)^2
               Corner jika R > threshold (ambang relatif terhadap max(R))

    Shi-Tomasi (1994): R_ST = min(lambda1, lambda2)
               Corner jika min-eigenvalue > threshold
               Lebih andal untuk optical flow karena corner yang dipilih
               memiliki stabilitas di SEMUA arah (kedua eigenvalue besar).

    cv2.goodFeaturesToTrack() mengimplementasikan Shi-Tomasi.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f  = np.float32(gray)

    # Harris -- di-threshold 1% dari max respons
    R     = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    R_dil = cv2.dilate(R, None)
    vis_h = img_bgr.copy()
    mask  = R_dil > 0.01 * R_dil.max()
    vis_h[mask] = [0, 0, 255]              # merah
    n_harris = int(mask.sum())

    # Shi-Tomasi via goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=300,
                                      qualityLevel=0.01, minDistance=5)
    vis_st = img_bgr.copy()
    if corners is not None:
        for pt in corners:
            x, y = pt.ravel().astype(int)
            cv2.circle(vis_st, (x, y), 3, (0, 255, 0), -1)  # hijau
    n_shi = len(corners) if corners is not None else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Demo 4 -- Harris vs Shi-Tomasi Corner",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(bgr_ke_rgb(img_bgr))
    tambah_judul(axes[0], "Gambar Asli")

    axes[1].imshow(bgr_ke_rgb(vis_h))
    axes[1].set_title(
        f"Harris (merah)\ncriterion: R = det-k*trace^2\n~{n_harris} piksel", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(bgr_ke_rgb(vis_st))
    axes[2].set_title(
        f"Shi-Tomasi (hijau)\ncriterion: min(lambda1,lambda2)\n{n_shi} corner", fontsize=9)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_harris_vs_shi_tomasi.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 4] 01_harris_vs_shi_tomasi.png disimpan.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Modul 04 -- Harris Corner Detection")
    print("=" * 60)
    img = muat_gambar("kota.jpg")
    demo_harris_dasar(img)
    demo_pengaruh_k(img)
    demo_heatmap_respons(img)
    demo_harris_vs_shi_tomasi(img)
    print("\n[SELESAI] Semua demo Harris Corner Detection telah dijalankan.")
    print(f"Output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
