"""
==========================================================================
 PERCOBAAN 17 — KOREKSI GAMMA DAN TRANSFORMASI POWER-LAW
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memahami dan menerapkan koreksi gamma untuk mengatur kecerahan
           gambar sesuai karakteristik sensor/layar dan persepsi manusia.
 Konsep  :
   - GAMMA CORRECTION: intensitas_output = intensitas_input ^ (1/gamma)
     atau secara umum: I_out = c * I_in ^ gamma
   - Gamma < 1 → gambar lebih terang (mencerahkan bayangan)
   - Gamma > 1 → gambar lebih gelap (meningkatkan kontras highlight)
   - LUT (Lookup Table): cara efisien menerapkan gamma; hitung 256 nilai
     sekali, lalu gunakan cv2.LUT() untuk pemetaan O(1) per piksel.
   - Kamera digital menyimpan gambar dalam ruang sRGB (gamma ~2.2).
     Untuk komputasi linear, perlu linearisasi: I_lin = I_srgb ^ 2.2
   - Manusia melihat intensitas secara logaritmik (Weber-Fechner), bukan
     linear — inilah sebab koreksi gamma penting untuk tampilan natural.
 Catatan : OpenCV menyimpan dalam uint8; selalu konversi ke float [0,1]
           sebelum operasi gamma, lalu konversi kembali ke uint8.
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def baca_gambar(nama: str) -> np.ndarray:
    path = os.path.join(IMAGE_DIR, nama)
    img = cv2.imread(path)
    if img is None:
        # Gambar sintetis: gradien + lingkaran
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        for i in range(400):
            img[:, i] = int(i / 399 * 255)
        cv2.circle(img, (200, 150), 80, (200, 120, 50), -1)
        cv2.putText(img, "GAMMA", (120, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3)
    return img


def buat_lut_gamma(gamma: float) -> np.ndarray:
    """Membuat LUT 256 elemen untuk koreksi gamma."""
    tabel = np.arange(256, dtype=np.float32) / 255.0
    tabel = np.power(tabel, gamma) * 255.0
    return np.clip(tabel, 0, 255).astype(np.uint8)


def koreksi_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Menerapkan koreksi gamma via LUT (cepat untuk uint8)."""
    lut = buat_lut_gamma(gamma)
    return cv2.LUT(img, lut)


def demo_variasi_gamma(img):
    """Bandingkan hasil gamma 0.25 – 3.0 secara visual."""
    nilai_gamma = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, g in zip(axes.flat, nilai_gamma):
        hasil = koreksi_gamma(img_gray, g)
        ax.imshow(hasil, cmap="gray", vmin=0, vmax=255)
        label = "terang" if g < 1 else ("normal" if g == 1.0 else "gelap")
        ax.set_title(f"γ = {g} ({label})", fontsize=11)
        ax.axis("off")

    plt.suptitle("Percobaan 17 — Variasi Koreksi Gamma", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "17_variasi_gamma.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


def plot_kurva_gamma():
    """Plot kurva transfer gamma I_out = I_in^γ untuk berbagai γ."""
    x = np.linspace(0, 1, 256)
    nilai_gamma = [0.25, 0.5, 1.0, 2.0, 3.0]
    warna = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for g, c in zip(nilai_gamma, warna):
        y = np.power(x, g)
        ax.plot(x, y, color=c, lw=2, label=f"γ = {g}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="linear (γ=1)")
    ax.set_xlabel("Intensitas Input (ternormalisasi)")
    ax.set_ylabel("Intensitas Output (ternormalisasi)")
    ax.set_title("Kurva Transfer Gamma")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    out = os.path.join(OUTPUT_DIR, "17_kurva_gamma.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


def demo_foto_gelap(img):
    """Mencerahkan foto yang under-exposed dengan gamma < 1."""
    # Simulasi foto gelap: kurangi brightness
    img_gelap = np.clip(img.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)

    gamma_values = [0.3, 0.5, 0.7]
    hasil_list = [koreksi_gamma(img_gelap, g) for g in gamma_values]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(cv2.cvtColor(img_gelap, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Foto Gelap (under-exposed)")
    axes[0].axis("off")
    for ax, g, h in zip(axes[1:], gamma_values, hasil_list):
        ax.imshow(cv2.cvtColor(h, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Dikoreksi γ = {g}")
        ax.axis("off")

    plt.suptitle("Percobaan 17 — Perbaikan Foto Gelap dengan Gamma", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "17_koreksi_foto_gelap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


def demo_histogram_gamma(img):
    """Tampilkan histogram sebelum dan sesudah koreksi gamma."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    terang = koreksi_gamma(gray, 0.45)   # γ < 1 → cerahkan
    gelap  = koreksi_gamma(gray, 2.20)   # γ > 1 → gelapkan

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    gambar = [("Asli", gray), ("γ=0.45 (cerah)", terang), ("γ=2.20 (gelap)", gelap)]
    for col, (judul, g) in enumerate(gambar):
        axes[0, col].imshow(g, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(judul); axes[0, col].axis("off")
        axes[1, col].hist(g.ravel(), bins=64, color="#3498db", alpha=0.8)
        axes[1, col].set_xlabel("Intensitas"); axes[1, col].set_ylabel("Frekuensi")
        axes[1, col].set_xlim(0, 255)

    plt.suptitle("Percobaan 17 — Histogram Sebelum & Sesudah Gamma", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "17_histogram_gamma.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 17: KOREKSI GAMMA DAN TRANSFORMASI POWER-LAW")
    print("=" * 60)

    img = baca_gambar("foto.jpg")
    print(f"Gambar: {img.shape[1]}×{img.shape[0]}")

    print("\n[1] Kurva Transfer Gamma")
    plot_kurva_gamma()

    print("\n[2] Variasi Gamma (0.25 – 3.0)")
    demo_variasi_gamma(img)

    print("\n[3] Mencerahkan Foto Gelap")
    demo_foto_gelap(img)

    print("\n[4] Histogram Sebelum & Sesudah Gamma")
    demo_histogram_gamma(img)

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
