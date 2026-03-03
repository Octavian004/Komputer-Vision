import cv2
import numpy as np
import sys
import os
from datetime import datetime


# ==============================
# 1. LOAD GAMBAR
# ==============================
# Cara pemakaian:
# - `python "Sistem watermark otomatis.py" path/to/image.jpg`
# - Atau letakkan file bernama `input.jpg` di folder yang sama
# - Jika tidak ada argumen dan `input.jpg` tidak ditemukan, akan muncul file dialog

image_path = None
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    default = "input.jpg"
    if os.path.exists(default):
        image_path = default
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(title="Pilih gambar",
                                                    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
            root.destroy()
            if not image_path:
                print("Tidak ada file dipilih. Keluar.")
                sys.exit(0)
        except Exception:
            print("Gagal membuka dialog file. Jalankan dengan: python \"Sistem watermark otomatis.py\" path/to/image.jpg")
            sys.exit(1)

img = cv2.imread(image_path)

if img is None:
    print(f"Gambar tidak ditemukan atau format tidak didukung: {image_path}")
    sys.exit(1)

# ==============================
# 2. MENAMPILKAN GAMBAR ASLI
# ==============================
cv2.imshow("Gambar Asli", img)

# ==============================
# 3. PROPERTI GAMBAR
# ==============================
height, width, channels = img.shape
print("Resolusi :", width, "x", height)
print("Jumlah Channel :", channels)
print("Tipe Data :", img.dtype)

# ==============================
# 4. KONVERSI WARNA (GRAYSCALE)
# ==============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)

# ==============================
# 5. MANIPULASI PIXEL (BRIGHTNESS SEDIKIT)
# ==============================
bright = cv2.convertScaleAbs(img, alpha=1, beta=30)
cv2.imshow("Brightness +30", bright)

# ==============================
# 6. MENAMBAHKAN WATERMARK
# ==============================

# Copy gambar agar asli tidak berubah
watermark_img = img.copy()

# Teks watermark
text = " Design by TIM 1_TRO A"

# Tambahkan timestamp otomatis
timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
full_text = text + " | " + timestamp

# Posisi teks (pojok kanan bawah)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 2

(text_width, text_height), _ = cv2.getTextSize(full_text, font, font_scale, thickness)

x = width - text_width - 20
y = height - 20

# Buat overlay untuk transparansi
overlay = watermark_img.copy()

cv2.putText(overlay, full_text, (x, y),
            font, font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA)

# Transparansi (0.0 - 1.0)
alpha = 0.4
cv2.addWeighted(overlay, alpha, watermark_img, 1 - alpha, 0, watermark_img)

cv2.imshow("Watermark Otomatis", watermark_img)

# ==============================
# 7. MENYIMPAN OUTPUT
# ==============================
cv2.imwrite("hasil_watermark.jpg", watermark_img)
print("Gambar berhasil disimpan sebagai hasil_watermark.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()