"""
Praktikum 04 - Segmentasi Warna menggunakan HSV dan inRange
Modul 03: Pemrosesan Citra

Topik: cv2.inRange(), HSV masking, color-based object detection
Referensi: OpenCV-Python Tutorial Ch.5, Mastering OpenCV 4 Ch.3,
           Learning Image Processing with OpenCV Ch.2
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_hsv_color_space():
    """Visualisasi distribusi warna di ruang HSV."""
    # Buat palette warna HSV
    hsv_bars = np.zeros((200, 360, 3), dtype=np.uint8)
    for h in range(360):
        hsv_bars[:, h] = [h // 2, 255, 255]
    bgr_bars = cv2.cvtColor(hsv_bars, cv2.COLOR_HSV2BGR)

    print("HSV Color Space:")
    print("  H (Hue)       : 0-179 di OpenCV (0=Merah, 60=Hijau, 120=Biru)")
    print("  S (Saturation): 0-255 (0=putih/abu, 255=warna penuh)")
    print("  V (Value)     : 0-255 (0=hitam, 255=terang penuh)")

    plt.figure(figsize=(12, 3))
    plt.imshow(cv2.cvtColor(bgr_bars, cv2.COLOR_BGR2RGB))
    plt.title("Hue Spectrum (H=0–179 di OpenCV)")
    plt.xlabel("Hue value (0–179)")
    plt.yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_04_hsv_spectrum.png"), dpi=100); plt.show()


# Rentang HSV standar untuk deteksi warna di OpenCV
COLOR_RANGES = {
    'Merah_bawah':  ([0,   120,  70], [10,  255, 255]),
    'Merah_atas':   ([170, 120,  70], [179, 255, 255]),
    'Kuning':       ([20,  100,  100], [35, 255, 255]),
    'Hijau':        ([36,  100,  100], [86, 255, 255]),
    'Biru':         ([100, 100,  100], [130, 255, 255]),
    'Oranye':       ([10,  100,  100], [20, 255, 255]),
    'Ungu':         ([130, 50,   50],  [160, 255, 255]),
    'Putih':        ([0,   0,    200], [179, 30, 255]),
    'Hitam':        ([0,   0,    0],   [179, 255, 50]),
}


def demo_basic_color_segmentation(image_path=None):
    """Segmentasi warna dasar menggunakan cv2.inRange() pada ruang HSV."""
    if image_path and cv2.haveImageReader(image_path):
        img = cv2.imread(image_path)
    else:
        # Buat gambar sintetis dengan beberapa objek berwarna
        img = np.ones((300, 500, 3), dtype=np.uint8) * 200  # latar abu-abu
        cv2.circle(img, (80, 150), 60, (0, 0, 200), -1)      # merah (BGR)
        cv2.rectangle(img, (180, 80), (310, 210), (0, 180, 0), -1)  # hijau
        cv2.circle(img, (400, 150), 55, (200, 100, 0), -1)    # biru
        cv2.rectangle(img, (120, 230), (250, 290), (0, 200, 200), -1)  # kuning
        cv2.circle(img, (350, 260), 35, (0, 100, 200), -1)    # oranye

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Deteksi warna merah (perlu dua range karena wrap di hue 179-0)
    mask_merah1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_merah2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([179, 255, 255]))
    mask_merah = cv2.bitwise_or(mask_merah1, mask_merah2)

    mask_hijau = cv2.inRange(hsv, np.array([36, 100, 100]), np.array([86, 255, 255]))
    mask_biru = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))

    # Terapkan mask
    result_merah = cv2.bitwise_and(img, img, mask=mask_merah)
    result_hijau = cv2.bitwise_and(img, img, mask=mask_hijau)
    result_biru = cv2.bitwise_and(img, img, mask=mask_biru)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Gambar Asli"); axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)); axes[0, 1].set_title("Ruang HSV"); axes[0, 1].axis('off')
    axes[0, 2].imshow(mask_merah, cmap='gray'); axes[0, 2].set_title("Mask Merah"); axes[0, 2].axis('off')
    axes[0, 3].imshow(cv2.cvtColor(result_merah, cv2.COLOR_BGR2RGB)); axes[0, 3].set_title("Hasil Merah"); axes[0, 3].axis('off')
    axes[1, 0].imshow(mask_hijau, cmap='gray'); axes[1, 0].set_title("Mask Hijau"); axes[1, 0].axis('off')
    axes[1, 1].imshow(cv2.cvtColor(result_hijau, cv2.COLOR_BGR2RGB)); axes[1, 1].set_title("Hasil Hijau"); axes[1, 1].axis('off')
    axes[1, 2].imshow(mask_biru, cmap='gray'); axes[1, 2].set_title("Mask Biru"); axes[1, 2].axis('off')
    axes[1, 3].imshow(cv2.cvtColor(result_biru, cv2.COLOR_BGR2RGB)); axes[1, 3].set_title("Hasil Biru"); axes[1, 3].axis('off')
    plt.suptitle("Segmentasi Warna dengan cv2.inRange() di Ruang HSV")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_04_color_segmentation.png"), dpi=100); plt.show()
    print("[OK] Segmentasi warna dasar selesai.")


def demo_morphology_refinement():
    """Perbaiki mask warna dengan operasi morfologi (hapus noise kecil)."""
    img = np.ones((300, 500, 3), dtype=np.uint8) * 190
    cv2.circle(img, (100, 150), 65, (0, 0, 200), -1)
    # Tambah noise kecil merah di background
    for _ in range(30):
        cx, cy = np.random.randint(0, 500), np.random.randint(0, 300)
        cv2.circle(img, (cx, cy), np.random.randint(2, 6), (0, 0, 200), -1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_raw = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))

    # Refinement dengan morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_open = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)    # hapus noise kecil
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)  # tutup lubang

    result_raw = cv2.bitwise_and(img, img, mask=mask_raw)
    result_clean = cv2.bitwise_and(img, img, mask=mask_close)

    print(f"  Pixel merah (raw):   {np.sum(mask_raw > 0)}")
    print(f"  Pixel merah (clean): {np.sum(mask_close > 0)}")

    plt.figure(figsize=(16, 4))
    for i, (title, vis) in enumerate([
        ("Asli + Noise", img),
        ("Mask Raw", mask_raw),
        ("Mask Setelah Morfologi", mask_close),
        ("Hasil Bersih", result_clean),
    ]):
        plt.subplot(1, 4, i + 1)
        if len(vis.shape) == 2:
            plt.imshow(vis, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(title); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_04_morphology_clean.png"), dpi=100); plt.show()


def demo_multi_color_detection():
    """Deteksi dan label beberapa objek berbeda warna sekaligus."""
    img = np.ones((350, 600, 3), dtype=np.uint8) * 240
    objects = [
        ((80, 120),  60, (0, 0, 220), "Merah"),
        ((220, 120), 55, (20, 200, 20), "Hijau"),
        ((360, 120), 50, (200, 80, 0),  "Biru"),
        ((490, 120), 45, (0, 200, 220), "Kuning"),
        ((150, 270), 55, (0, 120, 220), "Oranye"),
        ((350, 270), 50, (180, 0, 180), "Ungu"),
    ]
    for (cx, cy), r, color, _ in objects:
        cv2.circle(img, (cx, cy), r, color, -1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_defs = {
        'Merah': (np.array([0, 120, 70]), np.array([10, 255, 255]), (0, 255, 0)),
        'Hijau': (np.array([36, 80, 80]), np.array([86, 255, 255]), (0, 200, 0)),
        'Biru': (np.array([100, 100, 100]), np.array([130, 255, 255]), (255, 0, 0)),
        'Kuning': (np.array([20, 100, 100]), np.array([35, 255, 255]), (0, 200, 255)),
        'Oranye': (np.array([10, 100, 100]), np.array([22, 255, 255]), (0, 165, 255)),
        'Ungu': (np.array([130, 50, 50]), np.array([160, 255, 255]), (255, 0, 255)),
    }

    result = img.copy()
    detected_colors = []
    for color_name, (lower, upper, draw_color) in color_defs.items():
        mask = cv2.inRange(hsv, lower, upper)
        # Jika warna merah, cek juga wrap
        if color_name == 'Merah':
            mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([179, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 500:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                cv2.circle(result, (int(cx), int(cy)), int(r) + 3, draw_color, 2)
                cv2.putText(result, color_name, (int(cx) - 30, int(cy) - int(r) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
                detected_colors.append(color_name)
                print(f"  Terdeteksi: {color_name} di ({int(cx)}, {int(cy)}), radius={int(r)}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Gambar Asli"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title(f"Deteksi Multi-Warna ({len(detected_colors)} objek)"); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "output_04_multi_color_detection.png"), dpi=100); plt.show()


def demo_color_picker_interactive():
    """Panduan mencari nilai HSV dari warna tertentu."""
    print("\n[Tips] Cara Mencari Nilai HSV untuk Warna Tertentu:")
    print("  1. Buka gambar referensi dengan warna target")
    print("  2. Konversi ke HSV: hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)")
    print("  3. Ambil piksel target: hsv[y, x]")
    print("  4. Gunakan nilai tersebut ± toleransi sebagai lower/upper")
    print()

    # Tabel referensi warna umum
    print(f"{'Warna':<12} {'H_low':>6} {'H_high':>7} {'S_low':>6} {'V_low':>6}")
    print("-" * 45)
    warna_ref = [
        ("Merah",   0,  10,  120, 70),
        ("Merah2",  170, 179, 120, 70),
        ("Oranye",  10, 22,  100, 100),
        ("Kuning",  20, 35,  100, 100),
        ("Hijau",   36, 86,  100, 100),
        ("Cyan",    80, 100, 100, 100),
        ("Biru",    100, 130, 100, 100),
        ("Ungu",    130, 160,  50, 50),
        ("Putih",   0,  179,   0, 200),
        ("Hitam",   0,  179,   0,   0),
        ("Abu-abu", 0,  179,  0,   50),
    ]
    for row in warna_ref:
        print(f"{row[0]:<12} {row[1]:>6} {row[2]:>7} {row[3]:>6} {row[4]:>6}")


if __name__ == "__main__":
    print("=" * 58)
    print("PRAKTIKUM 04: SEGMENTASI WARNA HSV (COLOR SEGMENTATION)")
    print("=" * 58)

    print("\n[1] Visualisasi Ruang Warna HSV")
    demo_hsv_color_space()

    print("\n[2] Segmentasi Warna Dasar dengan inRange()")
    demo_basic_color_segmentation()

    print("\n[3] Perbaikan Mask dengan Operasi Morfologi")
    demo_morphology_refinement()

    print("\n[4] Deteksi Multi-Warna Sekaligus")
    demo_multi_color_detection()

    print("\n[5] Tips Memilih Nilai HSV")
    demo_color_picker_interactive()

    print("\n[SELESAI] Semua demo segmentasi warna berhasil dijalankan.")
