"""
==========================================================================
PERCOBAAN 8: MENULIS TEKS PADA GAMBAR
==========================================================================
Program ini mempelajari cara menambahkan teks dan anotasi pada gambar.
Teks sering digunakan untuk label, caption, watermark, dan debug info.

Fungsi utama:
- cv2.putText()           : Menulis teks pada gambar
- cv2.getTextSize()       : Mendapatkan ukuran teks (untuk positioning)
- cv2.FONT_HERSHEY_*     : Berbagai font bawaan OpenCV

Parameter putText:
- img     : gambar target
- text    : string teks
- org     : posisi kiri-bawah teks (x, y)
- fontFace: jenis font
- fontScale: skala ukuran font
- color   : warna BGR
- thickness: ketebalan garis teks
- lineType : LINE_AA untuk anti-aliasing
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 8: MENULIS TEKS PADA GAMBAR")
print("=" * 60)

# ============================================================
# 1. Berbagai jenis font OpenCV
# ============================================================

# Membuat canvas hitam untuk menampilkan berbagai font
canvas = np.zeros((600, 800, 3), dtype=np.uint8)

# Daftar semua font bawaan OpenCV
fonts = [
    (cv2.FONT_HERSHEY_SIMPLEX, "SIMPLEX"),
    (cv2.FONT_HERSHEY_PLAIN, "PLAIN"),
    (cv2.FONT_HERSHEY_DUPLEX, "DUPLEX"),
    (cv2.FONT_HERSHEY_COMPLEX, "COMPLEX"),
    (cv2.FONT_HERSHEY_TRIPLEX, "TRIPLEX"),
    (cv2.FONT_HERSHEY_COMPLEX_SMALL, "COMPLEX_SMALL"),
    (cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, "SCRIPT_SIMPLEX"),
    (cv2.FONT_HERSHEY_SCRIPT_COMPLEX, "SCRIPT_COMPLEX"),
]

# Menulis teks dengan masing-masing font
for i, (font, nama) in enumerate(fonts):
    # Menghitung posisi y untuk setiap baris teks
    y = 50 + i * 60

    # Menulis nama font di sisi kiri
    cv2.putText(canvas, nama, (10, y), font, 0.6, (150, 150, 150), 1)

    # Menulis contoh teks "Hello OpenCV!" dengan font tersebut
    cv2.putText(canvas, "Hello OpenCV!", (300, y), font, 1.0,
                (0, 255, 0), 2, cv2.LINE_AA)

print("[INFO] 8 jenis font OpenCV ditampilkan")

# Font italic (tambahkan flag FONT_ITALIC)
y_italic = 530
cv2.putText(canvas, "ITALIC MODE", (10, y_italic),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
# Menggabungkan font dengan flag italic menggunakan operator bitwise OR (|)
cv2.putText(canvas, "Hello OpenCV!", (300, y_italic),
            cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC, 1.0,
            (0, 255, 255), 2, cv2.LINE_AA)

# ============================================================
# 2. Ukuran dan ketebalan teks
# ============================================================

canvas2 = np.zeros((400, 800, 3), dtype=np.uint8)

# Variasi fontScale (ukuran teks)
scales = [0.5, 1.0, 1.5, 2.0]
for i, scale in enumerate(scales):
    y = 50 + i * 80
    # cv2.putText(img, text, org, font, fontScale, color, thickness, lineType)
    cv2.putText(canvas2, f"Scale={scale}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 200, 255), 2, cv2.LINE_AA)

# Variasi thickness (ketebalan garis)
thicknesses = [1, 2, 3, 5]
for i, thick in enumerate(thicknesses):
    y = 50 + i * 80
    cv2.putText(canvas2, f"Thick={thick}", (500, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), thick, cv2.LINE_AA)

print("[INFO] Variasi ukuran dan ketebalan teks ditampilkan")

# ============================================================
# 3. Mendapatkan ukuran teks (untuk positioning)
# ============================================================
print("\n--- 3. Ukuran Teks (getTextSize) ---")

teks = "Computer Vision 2024"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1.0
thickness = 2

# cv2.getTextSize(text, fontFace, fontScale, thickness)
# Mengembalikan: (lebar, tinggi), baseline
(lebar_teks, tinggi_teks), baseline = cv2.getTextSize(teks, font, scale, thickness)
print(f"  Teks: '{teks}'")
print(f"  Lebar: {lebar_teks} piksel")
print(f"  Tinggi: {tinggi_teks} piksel")
print(f"  Baseline: {baseline} piksel")

# ============================================================
# 4. Teks di tengah gambar (centered text)
# ============================================================

canvas3 = np.zeros((300, 600, 3), dtype=np.uint8)

# Menghitung posisi agar teks berada tepat di tengah canvas
tinggi_canvas, lebar_canvas = canvas3.shape[:2]

# Posisi x: setengah lebar canvas dikurangi setengah lebar teks
x_center = (lebar_canvas - lebar_teks) // 2

# Posisi y: setengah tinggi canvas ditambah setengah tinggi teks
# (karena org adalah posisi kiri-BAWAH teks, bukan kiri-atas)
y_center = (tinggi_canvas + tinggi_teks) // 2

# Menggambar kotak background untuk teks
padding = 10
cv2.rectangle(canvas3,
              (x_center - padding, y_center - tinggi_teks - padding),
              (x_center + lebar_teks + padding, y_center + baseline + padding),
              (50, 50, 50), -1)

# Menulis teks di tengah
cv2.putText(canvas3, teks, (x_center, y_center),
            font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

print("[INFO] Teks di-center di gambar")

# ============================================================
# 5. Teks dengan background (label box)
# ============================================================

def buat_label(img, teks, posisi, font_scale=0.7,
               warna_teks=(255, 255, 255), warna_bg=(0, 0, 200)):
    """Helper function untuk membuat label teks dengan background box."""
    # Mendapatkan ukuran teks
    (w, h), baseline = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, 2)
    x, y = posisi

    # Menggambar rectangle background
    cv2.rectangle(img, (x, y - h - 5), (x + w + 5, y + baseline + 5),
                  warna_bg, -1)

    # Menulis teks di atas background
    cv2.putText(img, teks, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, warna_teks, 2, cv2.LINE_AA)

# Membaca gambar dan menambahkan label
img_berlabel = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"))
if img_berlabel is not None:
    # Menambahkan beberapa label
    buat_label(img_berlabel, "Kucing Lucu", (50, 50))
    buat_label(img_berlabel, "OpenCV Drawing", (50, 100), warna_bg=(200, 0, 0))
    buat_label(img_berlabel, "Modul 01", (50, 150), warna_bg=(0, 150, 0))

    # Menambahkan watermark semi-transparan
    overlay = img_berlabel.copy()
    cv2.putText(overlay, "WATERMARK", (100, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
    # cv2.addWeighted untuk efek semi-transparan
    img_berlabel = cv2.addWeighted(overlay, 0.3, img_berlabel, 0.7, 0)

    print("[INFO] Label dan watermark ditambahkan ke gambar")

# ============================================================
# 6. Multi-line text
# ============================================================

canvas4 = np.zeros((300, 500, 3), dtype=np.uint8)

# OpenCV tidak mendukung multi-line text secara langsung
# Kita harus menulis baris per baris
paragraf = [
    "Baris 1: Computer Vision",
    "Baris 2: Menggunakan OpenCV",
    "Baris 3: Python & NumPy",
    "Baris 4: Praktikum 2024",
]

# Menulis setiap baris teks dengan jarak antar baris
y_awal = 50
spasi_baris = 40  # Jarak antar baris dalam piksel

for i, baris in enumerate(paragraf):
    y_pos = y_awal + i * spasi_baris
    cv2.putText(canvas4, baris, (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

print("[INFO] Multi-line text berhasil ditulis")

# ============================================================
# 7. Visualisasi semua hasil
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

axes[0, 0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Berbagai Font OpenCV")
axes[0, 0].axis("off")

axes[0, 1].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Variasi Scale & Thickness")
axes[0, 1].axis("off")

axes[0, 2].imshow(cv2.cvtColor(canvas3, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Centered Text")
axes[0, 2].axis("off")

if img_berlabel is not None:
    axes[1, 0].imshow(cv2.cvtColor(img_berlabel, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Label + Watermark")
    axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(canvas4, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Multi-line Text")
axes[1, 1].axis("off")

# Info panel
axes[1, 2].text(0.1, 0.8, "Fungsi Teks OpenCV:", fontsize=12, fontweight="bold",
                transform=axes[1, 2].transAxes)
axes[1, 2].text(0.1, 0.6, "cv2.putText()\ncv2.getTextSize()\n\n"
                "8 Font + Italic\nAnti-aliasing\nCustom positioning",
                fontsize=10, transform=axes[1, 2].transAxes, family="monospace")
axes[1, 2].axis("off")

plt.suptitle("Percobaan 8: Menulis Teks pada Gambar", fontsize=16, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "08_menulis_teks_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 8")
print("=" * 60)
print("  1. cv2.putText()     → Menulis teks pada gambar")
print("  2. cv2.getTextSize() → Mendapatkan ukuran teks")
print("  3. 8 jenis font + italic mode")
print("  4. fontScale → ukuran, thickness → ketebalan")
print("  5. LINE_AA → anti-aliasing untuk teks halus")
print("  6. Multi-line: tulis baris per baris")
print("=" * 60)
