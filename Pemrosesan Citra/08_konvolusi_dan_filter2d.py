"""
Modul 03 - Pemrosesan Citra (Image Processing)
Topik  : Konvolusi dan Filter2D
Tujuan : Memahami operasi konvolusi 2D secara manual maupun menggunakan
         cv2.filter2D(), serta mempelajari berbagai jenis kernel standar
         dan pengaruh ukuran kernel terhadap waktu proses dan hasil.
Fungsi : cv2.filter2D(src, -1, kernel), np.ones()/(9), kernel normalisasi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ── Path direktori ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Fungsi bantu ────────────────────────────────────────────────────────────
def muat_atau_buat_gambar():
    """Memuat gambar dari IMAGE_DIR/kota.jpg; jika tidak ada, buat gambar sintetis."""
    jalur = os.path.join(IMAGE_DIR, "kota.jpg")
    if os.path.exists(jalur):
        # Baca gambar asli dan konversi ke RGB untuk matplotlib
        img = cv2.imread(jalur)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Gambar sintetis: gradien latar + persegi panjang + lingkaran
    kanvas = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(300):
        kanvas[i, :] = [i // 2, 80, 180 - i // 3]
    cv2.rectangle(kanvas, (50, 50), (150, 150), (255, 200, 0), -1)
    cv2.circle(kanvas, (280, 150), 80, (0, 220, 180), -1)
    cv2.putText(kanvas, "SINTETIS", (90, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return kanvas


# ── Demo 1: Konvolusi manual 3×3 vs cv2.filter2D ────────────────────────────
def demo_konvolusi_manual_vs_filter2d(img_rgb):
    """Membandingkan konvolusi manual piksel-per-piksel dengan cv2.filter2D()."""
    # Gunakan grayscale agar iterasi lebih ringkas
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Kernel rata-rata (box blur) 3×3 – jumlah elemen = 1
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0

    # Konvolusi manual dengan padding tepi 'edge'
    h, w = abu.shape
    hasil_manual = np.zeros_like(abu)
    abu_pad = np.pad(abu, 1, mode='edge')
    for y in range(h):
        for x in range(w):
            # Ekstrak patch 3×3, kalikan elemen-per-elemen, jumlahkan
            patch = abu_pad[y:y + 3, x:x + 3]
            hasil_manual[y, x] = np.sum(patch * kernel)

    # Bandingkan dengan implementasi OpenCV yang dioptimasi
    hasil_cv2 = cv2.filter2D(abu, -1, kernel)

    # Tampilkan ketiganya berdampingan
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].imshow(abu, cmap='gray');        axs[0].set_title("Asli (Grayscale)")
    axs[1].imshow(hasil_manual, cmap='gray'); axs[1].set_title("Konvolusi Manual 3×3")
    axs[2].imshow(hasil_cv2,   cmap='gray'); axs[2].set_title("cv2.filter2D (Box 3×3)")
    for ax in axs: ax.axis('off')
    plt.suptitle("Demo 1 – Konvolusi Manual vs cv2.filter2D", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_konvolusi_manual_vs_filter2d.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 2: Berbagai kernel standar ─────────────────────────────────────────
def demo_berbagai_kernel(img_rgb):
    """Menerapkan kernel identitas, box blur, sharpen, emboss, dan edge detect."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Kumpulan kernel standar yang umum digunakan
    kernels = {
        "Identity":    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32),
        "Box Blur":    np.ones((3, 3), np.float32) / 9.0,
        "Sharpen":     np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
        "Emboss":      np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32),
        "Edge Detect": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32),
    }

    fig, axs = plt.subplots(1, len(kernels) + 1, figsize=(16, 4))
    axs[0].imshow(abu, cmap='gray'); axs[0].set_title("Asli"); axs[0].axis('off')

    # Terapkan setiap kernel dan tampilkan hasilnya secara berurutan
    for idx, (nama, k) in enumerate(kernels.items(), start=1):
        hasil = cv2.filter2D(abu.astype(np.float32), -1, k)
        hasil = np.clip(hasil, 0, 255).astype(np.uint8)
        axs[idx].imshow(hasil, cmap='gray')
        axs[idx].set_title(nama); axs[idx].axis('off')

    plt.suptitle("Demo 2 – Berbagai Kernel Standar", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_berbagai_kernel.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 3: Custom kernel dari nol ──────────────────────────────────────────
def demo_custom_kernel(img_rgb):
    """Membuat kernel kustom: diagonal, dan Laplacian 5×5."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Kernel tepi arah horizontal
    k_horiz = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    # Kernel tepi diagonal (45°)
    k_diag  = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], np.float32)
    # Kernel Laplacian 5×5 – mendeteksi perubahan orde kedua
    k_lap5  = -np.ones((5, 5), np.float32)
    k_lap5[2, 2] = 24.0

    h1 = np.clip(cv2.filter2D(abu.astype(np.float32), -1, k_horiz), 0, 255).astype(np.uint8)
    h2 = np.clip(cv2.filter2D(abu.astype(np.float32), -1, k_diag),  0, 255).astype(np.uint8)
    h3 = np.clip(cv2.filter2D(abu.astype(np.float32), -1, k_lap5) + 128, 0, 255).astype(np.uint8)

    # Baris atas: visualisasi kernel; baris bawah: hasil filter
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, judul, data in zip(axs[0],
            ["Kernel Horizontal", "Kernel Diagonal", "Kernel Laplacian 5×5"],
            [k_horiz, k_diag, k_lap5]):
        im = ax.imshow(data, cmap='RdBu_r', interpolation='nearest')
        ax.set_title(judul); plt.colorbar(im, ax=ax)
    for ax, judul, data in zip(axs[1],
            ["Hasil Horizontal", "Hasil Diagonal", "Hasil Laplacian 5×5"],
            [h1, h2, h3]):
        ax.imshow(data, cmap='gray'); ax.set_title(judul); ax.axis('off')

    plt.suptitle("Demo 3 – Custom Kernel dari Nol", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_custom_kernel.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Demo 4: Pengaruh ukuran kernel terhadap waktu dan hasil ─────────────────
def demo_pengaruh_ukuran_kernel(img_rgb):
    """Mengukur waktu filter2D untuk kernel box dengan berbagai ukuran."""
    abu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ukuran_kernel = [3, 7, 15, 31, 61]
    waktu_ms, hasil_list = [], []

    for k in ukuran_kernel:
        kernel = np.ones((k, k), np.float32) / (k * k)
        t0 = time.perf_counter()
        # Ulangi 20× untuk stabilitas pengukuran waktu
        for _ in range(20):
            h = cv2.filter2D(abu, -1, kernel)
        t1 = time.perf_counter()
        waktu_ms.append((t1 - t0) / 20 * 1000)
        hasil_list.append(h)

    # Baris atas: hasil filter; baris bawah: grafik waktu proses
    fig, axs = plt.subplots(2, len(ukuran_kernel), figsize=(16, 6),
                            gridspec_kw={'height_ratios': [3, 1]})
    for i, (k, h) in enumerate(zip(ukuran_kernel, hasil_list)):
        axs[0, i].imshow(h, cmap='gray')
        axs[0, i].set_title(f"k={k}\n{waktu_ms[i]:.2f} ms")
        axs[0, i].axis('off')

    # Grafik batang waktu vs ukuran kernel
    axs[1, 0].bar(range(len(ukuran_kernel)), waktu_ms, color='steelblue')
    axs[1, 0].set_xticks(range(len(ukuran_kernel)))
    axs[1, 0].set_xticklabels([str(k) for k in ukuran_kernel])
    axs[1, 0].set_xlabel("Ukuran Kernel"); axs[1, 0].set_ylabel("Waktu (ms)")
    axs[1, 0].set_title("Waktu vs Ukuran Kernel")
    for ax in axs[1, 1:]: ax.axis('off')

    plt.suptitle("Demo 4 – Pengaruh Ukuran Kernel terhadap Waktu & Hasil", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_pengaruh_ukuran_kernel.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Fungsi utama ─────────────────────────────────────────────────────────────
def main():
    print("=== Modul 03 | Konvolusi dan Filter2D ===")
    img = muat_atau_buat_gambar()
    print(f"Ukuran gambar: {img.shape}")

    demo_konvolusi_manual_vs_filter2d(img)
    demo_berbagai_kernel(img)
    demo_custom_kernel(img)
    demo_pengaruh_ukuran_kernel(img)

    print(f"Output disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
