"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
Topik   : AKAZE dan FAST Feature Detectors
=============================================================================
AKAZE (Accelerated-KAZE) -- Alcantarilla et al., 2013.
  - Membangun nonlinear scale-space menggunakan Fast Explicit Diffusion (FED)
    bukan Gaussian blur (berbeda dengan SIFT/SURF yang pakai Gaussian)
  - Nonlinear diffusion mempertahankan batas (edge/boundary) lebih baik
  - Deskriptor M-LDB (Modified Local Difference Binary): biner, rotation &
    scale invariant
  - Lebih baik dari KAZE dalam kecepatan, komparabel dengan ORB

FAST (Features from Accelerated Segment Test) -- Rosten & Drummond, 2006.
  - Algoritma deteksi corner tercepat yang ada
  - Periksa 16 piksel pada lingkaran Bresenham (radius 3)
  - Corner jika N (biasanya 9) piksel berturutan semua lebih terang atau
    semua lebih gelap dari piksel pusat + threshold intensitas
  - Tidak membangun scale-space -> tidak scale-invariant by default
  - Biasanya digabungkan dengan deskriptor lain (mis. BRIEF, SIFT, ORB)

Non-Maximum Suppression (NMS) di FAST:
  - Jika dua corner yang berdekatan terdeteksi, pertahankan hanya yang
    memiliki response lebih besar (versi FAST-9,16)

Perbandingan final: Harris / ORB / AKAZE / FAST
=============================================================================
"""

import os
import time
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
    Memuat gambar dari IMAGE_DIR. Jika tidak ada, gambar sintetis kaya
    tekstur dibuat sebagai fallback.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(filepath):
        img = cv2.imread(filepath)
        if img is not None:
            print(f"[INFO] Gambar dimuat: {filepath}")
            return img

    print(f"[INFO] '{filename}' tidak ditemukan -- membuat gambar sintetis.")
    np.random.seed(99)
    h, w = 480, 640
    kanvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background
    for i in range(h):
        kanvas[i] = [int(30 + i * 0.3), int(30 + i * 0.2), int(80 - i * 0.1)]

    # Bangunan / gedung (kotak dengan banyak jendela)
    for bx, by, bw, bh, col in [
        (50,  80,  150, 300, (160, 140, 120)),
        (250, 120, 180, 260, (140, 160, 140)),
        (480, 60,  120, 340, (120, 140, 160)),
    ]:
        cv2.rectangle(kanvas, (bx, by), (bx+bw, by+bh), col, -1)
        cv2.rectangle(kanvas, (bx, by), (bx+bw, by+bh), (200, 200, 200), 1)
        # Jendela-jendela kecil
        for wr in range(4):
            for wc in range(3):
                wx = bx + 15 + wc * (bw // 3)
                wy = by + 20 + wr * 60
                cv2.rectangle(kanvas, (wx, wy), (wx+25, wy+35), (80, 80, 80), -1)
                cv2.rectangle(kanvas, (wx, wy), (wx+25, wy+35), (200, 200, 200), 1)

    # Jalan dan objek lain
    cv2.rectangle(kanvas, (0, 400), (w, h), (60, 60, 60), -1)
    cv2.line(kanvas, (w//2, 400), (w//2, h), (200, 200, 100), 2)
    cv2.circle(kanvas, (80, 370), 30, (50, 200, 50), -1)   # pohon
    cv2.circle(kanvas, (560, 380), 25, (50, 200, 50), -1)

    noise = np.random.randint(0, 12, kanvas.shape, dtype=np.uint8)
    kanvas = cv2.add(kanvas, noise)
    return kanvas


def bgr_ke_rgb(img):
    """Konversi BGR -> RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gambar_keypoints(img_bgr, kps, warna=(0, 255, 0), rich=False):
    """Gambar keypoint pada salinan gambar."""
    flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else cv2.DrawMatchesFlags_DEFAULT
    return cv2.drawKeypoints(img_bgr, kps, None, color=warna, flags=flag)


# ---------------------------------------------------------------------------
# Demo 1 -- AKAZE: Nonlinear Scale-Space
# ---------------------------------------------------------------------------

def demo_akaze_dasar(img_bgr):
    """
    AKAZE menggunakan Fast Explicit Diffusion (FED) untuk membangun
    nonlinear scale-space. Berbeda dengan Gaussian yang memudarkan semua
    tepi, diffusi nonlinier mempertahankan kontur objek lebih baik.

    cv2.AKAZE_create() -- parameter utama:
      - descriptor_type   : AKAZE.DESCRIPTOR_MLDB (biner, default)
      - descriptor_size   : 0 = ukuran penuh
      - descriptor_channels: 3 (default)
      - threshold         : respons minimum keypoint (default 0.001)
      - nOctaves, nOctaveLayers: kontrol scale-space
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()

    kps, descs = akaze.detectAndCompute(gray, None)

    vis_rich = gambar_keypoints(img_bgr, kps, rich=True)

    # Distribusi distribusi level oktaf
    sizes = [kp.size for kp in kps]
    resps = [kp.response for kp in kps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Demo 1 -- AKAZE: Nonlinear Scale-Space",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(bgr_ke_rgb(img_bgr))
    axes[0].set_title("Gambar Asli"); axes[0].axis("off")

    axes[1].imshow(bgr_ke_rgb(vis_rich))
    axes[1].set_title(
        f"AKAZE Keypoints (Rich)\n{len(kps)} kp\n"
        f"desc: {descs.shape if descs is not None else 'None'}", fontsize=9)
    axes[1].axis("off")

    axes[2].scatter(sizes, resps, alpha=0.6, s=20, c="teal")
    axes[2].set_xlabel("Size (piksel)")
    axes[2].set_ylabel("Response")
    axes[2].set_title("Response vs Size\nAKAZE Keypoints", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_akaze_dasar.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Demo 1] {len(kps)} kp AKAZE. 05_akaze_dasar.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 2 -- FAST: Circle-Based Corner Detection
# ---------------------------------------------------------------------------

def demo_fast_dasar(img_bgr):
    """
    FAST (Features from Accelerated Segment Test):
    Sebuah piksel p dianggap corner jika terdapat N piksel berturutan pada
    lingkaran radius 3 (16 piksel total) yang semuanya memenuhi:
        I_p > I(lingkaran) + threshold  (brighter arc)
        atau
        I_p < I(lingkaran) - threshold  (darker arc)

    N biasanya 9 (FAST-9) atau 12 (FAST-12).

    cv2.FastFeatureDetector_create(threshold, nonmaxSuppression, type)
      - threshold       : perbedaan intensitas minimum untuk dianggap corner
      - nonmaxSuppression: hapus corner lemah yang berdekatan (default True)
      - type            : FAST_FEATURE_DETECTOR_TYPE_9_16 (default)

    Pengaruh threshold:
      kecil -> lebih sensitif, banyak corner tapi juga noise
      besar -> lebih selektif, hanya corner sangat tajam
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    daftar_thr = [5, 15, 30, 60]

    fig, axes = plt.subplots(1, len(daftar_thr), figsize=(16, 4))
    fig.suptitle("Demo 2 -- FAST: Pengaruh Threshold",
                 fontsize=12, fontweight="bold")

    for ax, thr in zip(axes, daftar_thr):
        fast = cv2.FastFeatureDetector_create(threshold=thr, nonmaxSuppression=True)
        kps  = fast.detect(gray, None)
        vis  = gambar_keypoints(img_bgr, kps, warna=(255, 100, 0))
        ax.imshow(bgr_ke_rgb(vis))
        ax.set_title(f"threshold={thr}\n{len(kps)} corner", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_fast_threshold.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 2] 05_fast_threshold.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 3 -- Non-Maximum Suppression pada FAST
# ---------------------------------------------------------------------------

def demo_fast_nms(img_bgr):
    """
    Non-Maximum Suppression (NMS) di FAST menghapus corner yang berdekatan
    dan memiliki respons lebih lemah, sehingga hanya satu corner perwakilan
    yang dipilih per kelompok.

    Tanpa NMS: banyak corner terkumpul, memenuhi tepi bangunan dll.
    Dengan NMS: distribusi lebih menyebar, lebih representatif.

    Visualisasi juga menampilkan heatmap distribusi spasial keypoint
    menggunakan kernel density estimation sederhana.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    fast_tanpa = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=False)
    fast_dengan = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)

    kps_tanpa  = fast_tanpa.detect(gray, None)
    kps_dengan = fast_dengan.detect(gray, None)

    vis_tanpa  = gambar_keypoints(img_bgr, kps_tanpa,  warna=(0, 0, 255))
    vis_dengan = gambar_keypoints(img_bgr, kps_dengan, warna=(0, 255, 0))

    # Buat density map sederhana
    def buat_density(kps, shape):
        dm = np.zeros(shape, dtype=np.float32)
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                dm[y, x] += 1.0
        # Blur untuk tampilan lebih halus
        return cv2.GaussianBlur(dm, (51, 51), 15)

    dens_tanpa  = buat_density(kps_tanpa,  (h, w))
    dens_dengan = buat_density(kps_dengan, (h, w))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Demo 3 -- FAST: Non-Maximum Suppression (NMS)",
                 fontsize=12, fontweight="bold")

    axes[0, 0].imshow(bgr_ke_rgb(vis_tanpa))
    axes[0, 0].set_title(f"Tanpa NMS: {len(kps_tanpa)} corner (merah)", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(bgr_ke_rgb(vis_dengan))
    axes[0, 1].set_title(f"Dengan NMS: {len(kps_dengan)} corner (hijau)", fontsize=9)
    axes[0, 1].axis("off")

    im0 = axes[1, 0].imshow(dens_tanpa, cmap="hot")
    plt.colorbar(im0, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].set_title("Density Map -- Tanpa NMS", fontsize=9)
    axes[1, 0].axis("off")

    im1 = axes[1, 1].imshow(dens_dengan, cmap="hot")
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
    axes[1, 1].set_title("Density Map -- Dengan NMS", fontsize=9)
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_fast_nms.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Demo 3] Tanpa NMS: {len(kps_tanpa)} | Dengan NMS: {len(kps_dengan)}. "
          "05_fast_nms.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 4 -- Perbandingan 4 Detektor: Harris / ORB / AKAZE / FAST
# ---------------------------------------------------------------------------

def demo_perbandingan_empat_detektor(img_bgr):
    """
    Perbandingan komprehensif 4 detektor fitur:
      Harris  : corner berbasis eigenvalue, tidak scale-invariant
      ORB     : FAST + piramid multi-skala + rBRIEF, real-time
      AKAZE   : nonlinear scale-space, deskriptor biner M-LDB
      FAST    : tercepat, lingkaran 16-piksel, tidak scale-invariant

    Metrik yang dibandingkan:
      1. Jumlah keypoint terdeteksi
      2. Distribusi spasial (heatmap)
      3. Waktu eksekusi (ms)
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f  = np.float32(gray)
    h, w    = gray.shape
    n_ulang = 5

    # --- Harris ---
    def deteksi_harris():
        R = cv2.cornerHarris(gray_f, 2, 3, 0.04)
        R_dil = cv2.dilate(R, None)
        coords = np.argwhere(R_dil > 0.01 * R_dil.max())
        return [cv2.KeyPoint(float(c), float(r), 5) for r, c in coords]

    # --- ORB ---
    orb   = cv2.ORB_create(nfeatures=500)
    # --- AKAZE ---
    akaze = cv2.AKAZE_create()
    # --- FAST ---
    fast  = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)

    detektor = {
        "Harris": deteksi_harris,
        "ORB":    lambda: orb.detect(gray, None),
        "AKAZE":  lambda: akaze.detect(gray, None),
        "FAST":   lambda: fast.detect(gray, None),
    }
    warna_map = {
        "Harris": (255,  50,  50),
        "ORB":    ( 50, 255,  50),
        "AKAZE":  ( 50,  50, 255),
        "FAST":   (255, 200,   0),
    }

    hasil   = {}
    waktu   = {}
    kps_map = {}

    for nama, fn in detektor.items():
        t0 = time.perf_counter()
        for _ in range(n_ulang):
            kps = fn()
        elapsed = (time.perf_counter() - t0) * 1000 / n_ulang
        hasil[nama]   = len(kps)
        waktu[nama]   = elapsed
        kps_map[nama] = kps

    # --- Plot 4 panel gambar keypoint ---
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Demo 4 -- Perbandingan Harris / ORB / AKAZE / FAST",
                 fontsize=13, fontweight="bold")

    names = list(detektor.keys())
    for i, nama in enumerate(names, 1):
        ax = fig.add_subplot(2, 4, i)
        vis = gambar_keypoints(img_bgr, kps_map[nama], warna=warna_map[nama])
        ax.imshow(bgr_ke_rgb(vis))
        ax.set_title(
            f"{nama}\n{hasil[nama]} kp | {waktu[nama]:.1f} ms", fontsize=9)
        ax.axis("off")

    # Density map untuk setiap detektor
    def density(kps):
        dm = np.zeros((h, w), np.float32)
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < w and 0 <= y < h:
                dm[y, x] += 1.0
        return cv2.GaussianBlur(dm, (31, 31), 10)

    for i, nama in enumerate(names, 5):
        ax = fig.add_subplot(2, 4, i)
        im = ax.imshow(density(kps_map[nama]), cmap="inferno")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"Density Map -- {nama}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_perbandingan_4_detektor.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # --- Ringkasan ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle("Ringkasan: Jumlah Keypoint & Waktu Eksekusi", fontsize=11)

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    axes2[0].bar(names, [hasil[n] for n in names], color=colors)
    axes2[0].set_ylabel("Jumlah Keypoint")
    axes2[0].set_title("Jumlah Keypoint")
    for ax_bar, val in zip(axes2[0].patches, [hasil[n] for n in names]):
        axes2[0].text(ax_bar.get_x() + ax_bar.get_width()/2,
                      ax_bar.get_height() + 5, str(val), ha="center", fontsize=9)

    axes2[1].bar(names, [waktu[n] for n in names], color=colors)
    axes2[1].set_ylabel("Waktu (ms)")
    axes2[1].set_title("Waktu Eksekusi Rata-rata")
    for ax_bar, val in zip(axes2[1].patches, [waktu[n] for n in names]):
        axes2[1].text(ax_bar.get_x() + ax_bar.get_width()/2,
                      ax_bar.get_height() * 1.02,
                      f"{val:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_ringkasan_4_detektor.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    print("[Demo 4] Hasil perbandingan:")
    for nama in names:
        print(f"  {nama:8s}: {hasil[nama]:5d} kp | {waktu[nama]:6.1f} ms")
    print("05_perbandingan_4_detektor.png & 05_ringkasan_4_detektor.png disimpan.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Modul 04 -- AKAZE dan FAST Feature Detectors")
    print("=" * 60)
    img = muat_gambar("kota.jpg")
    demo_akaze_dasar(img)
    demo_fast_dasar(img)
    demo_fast_nms(img)
    demo_perbandingan_empat_detektor(img)
    print("\n[SELESAI] Semua demo AKAZE & FAST telah dijalankan.")
    print(f"Output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
