"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
Topik   : SIFT Feature Detection & Description
=============================================================================
SIFT (Scale-Invariant Feature Transform) -- Lowe, 2004.
Invariant terhadap: rotasi, skala, perubahan pencahayaan, perubahan sudut pandang
  (sebagian).

Tahapan SIFT:
  1. Scale-Space Extrema Detection
     - Bangun Gaussian pyramid (D oktaf, masing-masing S+3 level)
     - Hitung DoG (Difference of Gaussian) untuk setiap pasangan level
     - Cari local extrema dalam DoG (kandidat keypoint)

  2. Keypoint Localization
     - Hapus keypoint dengan response rendah (|DoG| < thr)
     - Hapus keypoint di tepi (edge) dengan rasio kelengkungan

  3. Orientation Assignment
     - Hitung histogram gradien sekitar keypoint (36 bin, 360 deg)
     - Arah puncak histogram = orientasi dominan keypoint

  4. Keypoint Descriptor
     - Bagi region 16x16 di sekitar keypoint menjadi 4x4 sub-region
     - Hitung histogram gradien 8-bin di setiap sub-region
     - Gabungkan: 4*4*8 = 128 dimensi, lalu normalisasi

Keypoint attributes: pt (x,y), size (sigma*scale), angle (0-360),
                     response (strength), octave (scale level)
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
    Memuat gambar dari IMAGE_DIR. Fallback: gambar sintetis dengan tekstur
    yang cukup kaya untuk menghasilkan keypoint SIFT.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(filepath):
        img = cv2.imread(filepath)
        if img is not None:
            print(f"[INFO] Gambar dimuat: {filepath}")
            return img

    print(f"[INFO] '{filename}' tidak ditemukan -- membuat gambar sintetis.")
    np.random.seed(42)
    h, w = 480, 640
    # Base gradient
    kanvas = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        kanvas[:, :, c] = np.tile(
            np.linspace(30, 180, w, dtype=np.uint8), (h, 1))

    # Tambahkan berbagai bentuk geometris untuk fitur beragam
    cv2.rectangle(kanvas, (50, 50),   (200, 150), (220, 180, 100), -1)
    cv2.rectangle(kanvas, (250, 200), (450, 380), (100, 180, 220), -1)
    cv2.circle(kanvas,    (520, 120), 70,         (200, 100, 180), -1)
    pts = np.array([[100, 350], [50, 460], [200, 460]], np.int32)
    cv2.fillPoly(kanvas, [pts], (180, 200, 100))
    # Tambahkan kotak kecil sebagai tekstur
    for i in range(0, w, 40):
        for j in range(0, h, 40):
            if (i // 40 + j // 40) % 2 == 0:
                cv2.rectangle(kanvas, (i, j), (i+20, j+20), (150, 150, 150), 1)
    # Noise Gaussian
    noise = np.random.normal(0, 8, kanvas.shape).astype(np.int16)
    kanvas = np.clip(kanvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return kanvas


def bgr_ke_rgb(img):
    """Konversi BGR -> RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def buat_sift():
    """Membuat detektor SIFT (kompatibel OpenCV 4.x)."""
    return cv2.SIFT_create()


# ---------------------------------------------------------------------------
# Demo 1 -- SIFT dasar: keypoint dengan ukuran dan orientasi
# ---------------------------------------------------------------------------

def demo_sift_dasar(img_bgr):
    """
    Mendeteksi dan menampilkan keypoint SIFT beserta informasi:
      - Posisi  (pt.x, pt.y)
      - Ukuran  (pt.size) -- merepresentasikan sigma skala di mana featur terdeteksi
      - Orientasi (pt.angle -- 0-360 derajat)

    DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS menampilkan:
      - Lingkaran dengan radius proporsional terhadap pt.size
      - Garis dari pusat menunjukkan orientasi
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = buat_sift()

    # detectAndCompute: deteksi keypoint + hitung descriptor sekaligus
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Gambar keypoint "kaya" (ukuran + orientasi)
    vis_rich = cv2.drawKeypoints(
        img_bgr, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Gambar keypoint sederhana (hanya titik)
    vis_simple = cv2.drawKeypoints(img_bgr, keypoints, None,
                                   color=(0, 255, 0))

    # Statistik keypoint
    sizes     = [kp.size     for kp in keypoints]
    responses = [kp.response for kp in keypoints]
    octaves   = [kp.octave & 0xFF for kp in keypoints]  # bits 0-7 = oktaf

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Demo 1 -- SIFT Keypoints (Ukuran + Orientasi)",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(bgr_ke_rgb(img_bgr))
    axes[0].set_title("Gambar Asli"); axes[0].axis("off")

    axes[1].imshow(bgr_ke_rgb(vis_rich))
    axes[1].set_title(
        f"Rich Keypoints\n{len(keypoints)} kp | "
        f"size avg={np.mean(sizes):.1f}", fontsize=9)
    axes[1].axis("off")

    # Scatter plot: response vs size per oktaf
    sc = axes[2].scatter(sizes, responses, c=octaves,
                         cmap="tab10", alpha=0.7, s=15)
    plt.colorbar(sc, ax=axes[2], label="Oktaf")
    axes[2].set_xlabel("Size (sigma)")
    axes[2].set_ylabel("Response")
    axes[2].set_title("Response vs Size per Oktaf", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_sift_dasar.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Demo 1] {len(keypoints)} keypoint SIFT. 03_sift_dasar.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 2 -- Pengaruh nfeatures
# ---------------------------------------------------------------------------

def demo_pengaruh_nfeatures(img_bgr):
    """
    nfeatures = jumlah keypoint terbaik yang dikembalikan.
    nfeatures=0 berarti semua keypoint yang ditemukan dikembalikan.

    SIFT mengurutkan keypoint berdasarkan response (kualitas), lalu
    memotong hasilnya ke nfeatures teratas.
    """
    gray           = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    daftar_nfeat   = [0, 50, 200, 500]   # 0 = semua

    fig, axes = plt.subplots(1, len(daftar_nfeat), figsize=(16, 4))
    fig.suptitle("Demo 2 -- Pengaruh nfeatures pada SIFT",
                 fontsize=12, fontweight="bold")

    for ax, nf in zip(axes, daftar_nfeat):
        sift = cv2.SIFT_create(nfeatures=nf)
        kps, _  = sift.detectAndCompute(gray, None)
        vis = cv2.drawKeypoints(img_bgr, kps, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        label = "semua" if nf == 0 else str(nf)
        ax.imshow(bgr_ke_rgb(vis))
        ax.set_title(f"nfeatures={label}\n{len(kps)} kp", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_sift_nfeatures.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[Demo 2] 03_sift_nfeatures.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 3 -- Visualisasi Descriptor 128-D sebagai Heatmap
# ---------------------------------------------------------------------------

def demo_visualisasi_descriptor(img_bgr):
    """
    Deskriptor SIFT = vektor 128 elemen float yang merepresentasikan
    distribusi gradien di sekitar keypoint (4x4 grid x 8 histogram bins).

    Visualisasi:
      - Heatmap matriks deskriptor (N keypoint x 128 dimensi)
      - Distribusi nilai setiap dimensi (mean & std)
      - Contoh beberapa deskriptor individual sebagai bar chart
    """
    gray           = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift           = cv2.SIFT_create(nfeatures=100)
    kps, descs     = sift.detectAndCompute(gray, None)

    if descs is None or len(descs) == 0:
        print("[Demo 3] Tidak ada descriptor -- skip.")
        return

    # Normalisasi ke [0,1] untuk visualisasi
    descs_norm = descs / (descs.max(axis=1, keepdims=True) + 1e-8)
    n_show     = min(len(descs), 80)      # batasi jumlah baris heatmap

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Demo 3 -- Visualisasi Descriptor SIFT 128-D",
                 fontsize=12, fontweight="bold")

    # Heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    im  = ax1.imshow(descs_norm[:n_show], aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax1, fraction=0.046)
    ax1.set_xlabel("Dimensi (0-127)")
    ax1.set_ylabel("Keypoint")
    ax1.set_title(f"Heatmap Descriptor\n({n_show} dari {len(descs)} kp)", fontsize=9)

    # Mean dan std per dimensi
    ax2 = fig.add_subplot(1, 3, 2)
    mean_per_dim = descs.mean(axis=0)
    std_per_dim  = descs.std(axis=0)
    ax2.fill_between(range(128),
                     mean_per_dim - std_per_dim,
                     mean_per_dim + std_per_dim,
                     alpha=0.3, color="steelblue", label="±1 std")
    ax2.plot(mean_per_dim, color="steelblue", lw=1.5, label="Mean")
    ax2.set_xlabel("Dimensi")
    ax2.set_ylabel("Nilai descriptor")
    ax2.set_title("Mean ± Std per Dimensi", fontsize=9)
    ax2.legend(fontsize=8)

    # Bar chart 3 contoh deskriptor
    ax3 = fig.add_subplot(1, 3, 3)
    for i, color in zip([0, len(descs)//2, -1], ["red", "green", "blue"]):
        ax3.plot(descs_norm[i], color=color, alpha=0.7, lw=0.8,
                 label=f"kp #{i if i >= 0 else len(descs)+i}")
    ax3.set_xlabel("Dimensi (0-127)")
    ax3.set_ylabel("Nilai (ternormalisasi)")
    ax3.set_title("3 Contoh Deskriptor", fontsize=9)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_sift_descriptor.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Demo 3] {len(descs)} descriptor ({descs.shape[1]}-D). "
          "03_sift_descriptor.png disimpan.")


# ---------------------------------------------------------------------------
# Demo 4 -- SIFT Scale-Space: keypoint di berbagai level sigma
# ---------------------------------------------------------------------------

def demo_scale_space(img_bgr):
    """
    SIFT membangun scale-space menggunakan Gaussian pyramid dengan berbagai sigma.
    Keypoint di level sigma lebih tinggi merepresentasikan struktur lebih besar.

    Di sini kita visualisasikan keypoint yang terdeteksi di setiap oktaf
    (level piramida) secara terpisah, untuk melihat:
    - Oktaf 0: fitur skala kecil (detail halus)
    - Oktaf 2+: fitur skala besar (struktur kasar)

    kp.octave (int32) menyimpan oktaf di bits 0-7, level di bits 8-15.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=0)
    kps, _ = sift.detectAndCompute(gray, None)

    # Kelompokkan keypoint berdasarkan oktaf (bits 0-7)
    kelompok = {}
    for kp in kps:
        oct_val = kp.octave & 0xFF       # ambil byte pertama
        if oct_val > 127:                # nilai negatif di-encode
            oct_val -= 256
        kelompok.setdefault(oct_val, []).append(kp)

    daftar_oktaf = sorted(kelompok.keys())[:4]   # tampilkan maks 4 oktaf

    fig, axes = plt.subplots(1, max(len(daftar_oktaf), 1), figsize=(16, 4))
    fig.suptitle("Demo 4 -- SIFT Scale-Space: Keypoint per Oktaf",
                 fontsize=12, fontweight="bold")

    if len(daftar_oktaf) == 1:
        axes = [axes]

    for ax, oct_val in zip(axes, daftar_oktaf):
        kps_oct = kelompok[oct_val]
        vis = cv2.drawKeypoints(img_bgr, kps_oct, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        sizes_oct = [kp.size for kp in kps_oct]
        ax.imshow(bgr_ke_rgb(vis))
        ax.set_title(
            f"Oktaf {oct_val}\n{len(kps_oct)} kp\nsize avg={np.mean(sizes_oct):.1f}",
            fontsize=9)
        ax.axis("off")

    # Isi sisa subplot kosong jika daftar_oktaf lebih sedikit dari 4
    for ax in axes[len(daftar_oktaf):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_sift_scale_space.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Demo 4] {len(kps)} kp total, {len(daftar_oktaf)} oktaf. "
          "03_sift_scale_space.png disimpan.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Modul 04 -- SIFT Feature Detection & Description")
    print("=" * 60)
    img = muat_gambar("kota.jpg")
    demo_sift_dasar(img)
    demo_pengaruh_nfeatures(img)
    demo_visualisasi_descriptor(img)
    demo_scale_space(img)
    print("\n[SELESAI] Semua demo SIFT telah dijalankan.")
    print(f"Output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
