"""
SCRIPT GENERATOR SAMPLE IMAGE
Membuat gambar uji coba dengan berbagai warna untuk testing pipeline.

Jalankan sebelum main.py jika belum ada gambar input.
"""

import cv2
import numpy as np
import os

def create_sample_image():
    """
    Membuat gambar sample dengan bentuk dan warna beragam
    untuk testing pipeline analisis citra.
    """
    # Ukuran canvas
    height, width = 600, 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # Background putih
    
    # --- BAGIAN 1: WARNA MERAH ---
    # Lingkaran merah besar (atas kiri)
    cv2.circle(img, (200, 150), 80, (0, 0, 255), -1)  # BGR: merah
    
    # Kotak merah (atas tengah)
    cv2.rectangle(img, (350, 80), (550, 220), (0, 0, 255), -1)
    
    # --- BAGIAN 2: WARNA BIRU ---
    # Lingkaran biru (atas kanan)
    cv2.circle(img, (650, 150), 70, (255, 0, 0), -1)  # BGR: biru
    
    # --- BAGIAN 3: WARNA HIJAU ---
    # Elips hijau (sedang-atas)
    cv2.ellipse(img, (150, 350), (100, 60), 45, 0, 360, (0, 255, 0), -1)
    
    # --- BAGIAN 4: GRADASI ---
    # Gradient biru-merah (bawah kiri)
    for i in range(100):
        color_val = int(255 * (i / 100))
        cv2.line(img, (50, 450 + i), (250, 450 + i), (color_val, 0, 255 - color_val), 2)
    
    # --- BAGIAN 5: NOISE & DETAIL ---
    # Beberapa titik dan garis untuk membuat citra lebih menarik
    cv2.line(img, (300, 350), (450, 500), (50, 50, 50), 3)
    cv2.polylines(img, [np.array([[500, 350], [600, 400], [550, 500], [450, 470]])], 
                   True, (128, 128, 0), 2)
    
    # Random noise untuk test
    noise = np.random.randint(0, 100, img.shape, dtype=np.uint8)
    img = cv2.addWeighted(img, 0.95, noise, 0.05, 0)
    
    # Simpan gambar
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)
    
    output_path = os.path.join(input_dir, "sample.jpg")
    cv2.imwrite(output_path, img)
    
    print("=" * 60)
    print("✓ SAMPLE IMAGE BERHASIL DIBUAT".center(60))
    print("=" * 60)
    print(f"Lokasi   : {os.path.abspath(output_path)}")
    print(f"Ukuran   : {height}x{width} px")
    print(f"Konten   : Lingkaran, kotak, elips (merah, biru, hijau)")
    print("\nSekarang Anda bisa menjalankan: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    create_sample_image()
