"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
Topik   : ORB Feature Detection (FAST + rBRIEF)
=============================================================================
ORB (Oriented FAST and Rotated BRIEF) -- Rublee et al., 2011.
Open-source, bebas paten, dan jauh lebih cepat dari SIFT/SURF.

Komponen ORB:
  1. Detektor Keypoint: FAST (Features from Accelerated Segment Test)
     - Periksa 16 piksel pada lingkaran (radius 3) di sekitar kandidat pusat
     - Jika N piksel berturutan lebih terang/gelap dari pusat + threshold ->
       kandidat adalah corner
     - ORB menggunakan piramid multi-skala untuk invariansi skala

  2. Orientasi Keypoint: Intensity Centroid
     - Hitung momen intensitas di patch sekitar keypoint
     - Arah dari pusat ke centroid intensitas = orientasi keypoint

  3. Deskriptor: rBRIEF (Rotation-aware BRIEF)
     - BRIEF: 256 perbandingan biner piksel acak di patch keypoint
     - rBRIEF: rotasikan pola sampling sesuai orientasi keypoint
     - Hasil: vektor biner 256-bit (32 byte) -- sangat efisien

Cocokkan dengan Hamming distance (operasi bitwise XOR + popcount).
=============================================================================
"""

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    dengan banyak corner untuk ORB (FAST sangat sensitif terhadap sudut).
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(filepath):
        img = cv2.imread(filepath)
        if img is not None:
            print(f"[INFO] Gambar dimuat: {filepath}")
            return img

    print(f"[INFO] '{filename}' tidak ditemukan -- membuat gambar sintetis.")
    np.random.seed(7)
    h, w = 480, 640
    kanvas = np.ones((h, w, 3), dtype=np.uint8) * 50

    # Grid kotak -- menghasilkan banyak corner tajam
    for r in range(5):
        for c in range(6):
            x0 = 30 + c * 100
            y0 = 30 + r * 85
            col = (int(100 + c * 20), int(100 + r * 20), 180)
            cv2.rectangle(kanvas, (x0, y0), (x0+70, y0+60), col, 2)

    # Tambah beberapa lingkaran dan ellipse
    cv2.ellipse(kanvas, (320, 240), (120, 80), 30, 0, 360, (200, 180, 100), 2)
    cv2.circle(kanvas, (540, 400), 50, (180, 200, 180), 2)

    # Tambah tekstur checker board kecil
    for i in range(0, w, 20):
        for j in range(0, h, 20):
            if (i // 20 + j // 20) % 2 == 0:
                cv2.rectangle(kanvas, (i, j), (i+10, j+10), (150, 150, 80), -1)

    noise = np.random.randint(0, 10, kanvas.shape, dtype=np.uint8)
    kanvas = cv2.add(kanvas, noise)
    return kanvas


def bgr_ke_rgb(img):
    """Konversi BGR -> RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def buat_orb(nfeatures=500, scale_factor=1.2, nlevels=8):
    """Membuat detektor ORB dengan parameter yang dapat dikonfigurasi."""
    return cv2.ORB_create(nfeatures=nfeatures,
                          scaleFactor=scale_factor,
                          nlevels=nlevels)


# ---------------------------------------------------------------------------
# Demo 1 -- ORB dasar
# ---------------------------------------------------------------------------

def demo_orb_dasar(img_bgr):
    """
    Demo dasar ORB: detectAndCompute, lalu visualisasi keypoint.
    Tampilkan juga distribusi ukuran dan respon keypoint.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb  = buat_orb(nfeatures=500)

    kps, descs = orb.detectAndCompute(gray, None)

    # Rich keypoints: lingkaran proporsional ke size + garis orientasi
    vis = cv2.drawKeypoints(img_bgr, kps, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    sizes     = [kp.size     for kp in kps]
    responses = [kp.response for kp in kps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Demo 1 -- ORB Feature Detection Dasar",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(bgr_ke_rgb(img_bgr))
    axes[0].set_title("Gambar Asli"); axes[0].axis("off")

    axes[1].imshow(bgr_ke_rgb(vis))
    axes[1].set_title(
        f"ORB Keypoints (Rich)\n{len(kps)} kp | desc shape: {descs.shape if descs is not None else 'None'}",
        fontsize=9)
    axes[1].axis("off")

    axes[2].scatter(sizes, responses, alpha=0.6, s=20, c="darkorange")
    axes[2].set_xlabel("Size (piksel)")
    axes[2].set_ylabel("Response")
    axes[2].set_title(f"Response vs Size\n{len(kps)} keypoint ORB", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_orb_dasar.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Demo 1] {len(kps)} keypoint ORB. 04_orb_dasar.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 2 -- Pengaruh nfeatures
# ---------------------------------------------------------------------------

def demo_pengaruh_nfeatures(img_bgr):
    """
    Pengaruh jumlah fitur yang diminta terhadap cakupan dan distribusi keypoint.

    ORB memilih keypoint FAST berdasarkan respons (kekuatan corner), lalu
    membatasi jumlah hingga nfeatures. Semakin banyak nfeatures, semakin
    banyak keypoint dengan kualitas lebih rendah yang ikut dimasukkan.
    """
    gray         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    daftar_nfeat = [100, 500, 1000, 5000]

    fig, axes = plt.subplots(1, len(daftar_nfeat), figsize=(16, 4))
    fig.suptitle("Demo 2 -- Pengaruh nfeatures pada ORB",
                 fontsize=12, fontweight="bold")

    for ax, nf in zip(axes, daftar_nfeat):
        orb     = buat_orb(nfeatures=nf)
        kps, _  = orb.detectAndCompute(gray, None)
        vis     = cv2.drawKeypoints(img_bgr, kps, None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(bgr_ke_rgb(vis))
        ax.set_title(f"nfeatures={nf}\n{len(kps)} terdeteksi", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_orb_nfeatures.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 2] 04_orb_nfeatures.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 3 -- Visualisasi binary descriptor 256-bit
# ---------------------------------------------------------------------------

def demo_visualisasi_descriptor_biner(img_bgr):
    """
    Descriptor ORB adalah vektor biner 256-bit (disimpan sebagai 32 byte uint8).
    Visualisasikan pola bit untuk beberapa keypoint dan distribusi Hamming distance
    antar descriptor.

    Hamming distance = jumlah bit yang berbeda antara dua descriptor biner.
    Matching: cari pasangan dengan Hamming distance minimum.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb  = buat_orb(nfeatures=200)
    kps, descs = orb.detectAndCompute(gray, None)

    if descs is None or len(descs) < 5:
        print("[Demo 3] Terlalu sedikit descriptor -- skip.")
        return

    # Konversi setiap baris (32 byte) ke array 256 bit
    def bytes_to_bits(desc_row):
        bits = []
        for byte in desc_row:
            for bit_pos in range(7, -1, -1):
                bits.append((byte >> bit_pos) & 1)
        return np.array(bits)

    n_show = min(20, len(descs))     # tampilkan 20 descriptor

    # Matriks bit (n_show x 256)
    bit_matrix = np.array([bytes_to_bits(descs[i]) for i in range(n_show)])

    # Hitung Hamming distance antara semua pasangan
    n = len(descs)
    ham_dists = []
    for i in range(min(n, 50)):
        for j in range(i + 1, min(n, 50)):
            dist = int(np.sum(bytes_to_bits(descs[i]) != bytes_to_bits(descs[j])))
            ham_dists.append(dist)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Demo 3 -- Visualisasi Binary Descriptor ORB 256-bit",
                 fontsize=12, fontweight="bold")

    # Heatmap pola bit
    axes[0].imshow(bit_matrix, aspect="auto", cmap="binary", interpolation="nearest")
    axes[0].set_xlabel("Bit (0-255)")
    axes[0].set_ylabel("Keypoint index")
    axes[0].set_title(f"Pola Bit Descriptor\n({n_show} descriptor x 256 bit)", fontsize=9)

    # Bar chart jumlah '1' per dimensi bit
    sum_bits = bit_matrix.sum(axis=0)
    axes[1].bar(range(256), sum_bits, color="steelblue", width=1.0)
    axes[1].set_xlabel("Bit index (0-255)")
    axes[1].set_ylabel(f"Jumlah '1' (dari {n_show})")
    axes[1].set_title("Distribusi Bit Aktif per Dimensi\n(idealnya ~50% untuk diskriminasi baik)",
                      fontsize=9)

    # Histogram Hamming distance
    axes[2].hist(ham_dists, bins=30, color="darkorange", edgecolor="black")
    axes[2].set_xlabel("Hamming Distance")
    axes[2].set_ylabel("Frekuensi")
    axes[2].set_title(
        f"Distribusi Hamming Distance\n"
        f"rata-rata={np.mean(ham_dists):.1f} / 256 (~{100*np.mean(ham_dists)/256:.0f}%)",
        fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_orb_descriptor_biner.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Demo 3] {len(descs)} descriptor 256-bit. 04_orb_descriptor_biner.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 4 -- Perbandingan kecepatan SIFT vs ORB
# ---------------------------------------------------------------------------

def demo_perbandingan_sift_orb(img_bgr):
    """
    Perbandingan SIFT vs ORB dari sisi:
      - Waktu eksekusi (ms) per frame
      - Jumlah keypoint yang terdeteksi
      - Dimensi dan tipe descriptor

    ORB dirancang untuk kasus real-time di mana kecepatan adalah prioritas.
    SIFT lebih akurat dan robust terhadap perubahan skala/rotasi, tetapi lambat.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    n_ulang = 10       # rata-rata dari 10 percobaan untuk stabilitas timing

    # --- Benchmark SIFT ---
    sift = cv2.SIFT_create(nfeatures=500)
    t0 = time.perf_counter()
    for _ in range(n_ulang):
        kps_sift, descs_sift = sift.detectAndCompute(gray, None)
    waktu_sift = (time.perf_counter() - t0) * 1000 / n_ulang   # ms

    # --- Benchmark ORB ---
    orb = buat_orb(nfeatures=500)
    t0 = time.perf_counter()
    for _ in range(n_ulang):
        kps_orb, descs_orb = orb.detectAndCompute(gray, None)
    waktu_orb = (time.perf_counter() - t0) * 1000 / n_ulang    # ms

    # Visualisasi keypoint masing-masing
    vis_sift = cv2.drawKeypoints(img_bgr, kps_sift, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_orb  = cv2.drawKeypoints(img_bgr, kps_orb, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Demo 4 -- Perbandingan SIFT vs ORB",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(bgr_ke_rgb(vis_sift))
    axes[0].set_title(
        f"SIFT\n{len(kps_sift)} kp | {waktu_sift:.1f} ms\ndesc: {descs_sift.shape[1]}-D float32",
        fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(bgr_ke_rgb(vis_orb))
    axes[1].set_title(
        f"ORB\n{len(kps_orb)} kp | {waktu_orb:.1f} ms\ndesc: {descs_orb.shape[1] * 8 if descs_orb is not None else '?'}-bit biner",
        fontsize=9)
    axes[1].axis("off")

    # Bar chart perbandingan
    metrik   = ["Waktu (ms)", "Jumlah Keypoint"]
    val_sift = [waktu_sift,    len(kps_sift)]
    val_orb  = [waktu_orb,     len(kps_orb)]
    x        = np.arange(len(metrik))
    w        = 0.35
    axes[2].bar(x - w/2, val_sift, w, label="SIFT", color="steelblue")
    axes[2].bar(x + w/2, val_orb,  w, label="ORB",  color="darkorange")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrik, fontsize=9)
    axes[2].set_title("SIFT vs ORB: Waktu & Jumlah Keypoint\n(nfeatures=500)",
                      fontsize=9)
    axes[2].legend()
    # Anotasi nilai
    for rect, val in zip(axes[2].patches, val_sift + val_orb):
        axes[2].text(rect.get_x() + rect.get_width()/2,
                     rect.get_height() * 1.01,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_orb_vs_sift.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Demo 4] SIFT: {waktu_sift:.1f} ms | ORB: {waktu_orb:.1f} ms. "
          "04_orb_vs_sift.png disimpan.")
    print(f"         Speedup ORB vs SIFT: {waktu_sift/waktu_orb:.1f}x lebih cepat")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Modul 04 -- ORB Feature Detection (FAST + rBRIEF)")
    print("=" * 60)
    img = muat_gambar("kota.jpg")
    demo_orb_dasar(img)
    demo_pengaruh_nfeatures(img)
    demo_visualisasi_descriptor_biner(img)
    demo_perbandingan_sift_orb(img)
    print("\n[SELESAI] Semua demo ORB telah dijalankan.")
    print(f"Output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
