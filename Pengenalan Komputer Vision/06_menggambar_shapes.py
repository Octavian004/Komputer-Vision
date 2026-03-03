# ============================================================
# PROGRAM: 06_menggambar_shapes.py
# PRAKTIKUM COMPUTER VISION - BAB 1: PENDAHULUAN
# ============================================================
# Deskripsi: Program ini mendemonstrasikan cara menggambar
#            berbagai bentuk dan teks pada gambar
# 
# Tujuan Pembelajaran:
#   1. Menggambar garis, persegi, lingkaran, elips, polygon
#   2. Menulis teks pada gambar
#   3. Membuat anotasi dan markup pada gambar
#   4. Memahami koordinat dan parameter drawing
# ============================================================

# ====================
# IMPORT LIBRARY
# ====================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ============================================================
# VARIABEL YANG BISA DIUBAH-UBAH (EKSPERIMEN)
# ============================================================

# 1. Ukuran canvas (tinggi, lebar)
CANVAS_SIZE = (400, 600)

# 2. Warna latar belakang canvas (BGR)
WARNA_BACKGROUND = (30, 30, 30)  # Abu-abu gelap

# 3. Warna untuk menggambar (BGR) - Coba ubah!
WARNA_GARIS = (0, 255, 0)        # Hijau
WARNA_KOTAK = (255, 0, 0)        # Biru
WARNA_LINGKARAN = (0, 0, 255)    # Merah
WARNA_ELIPS = (255, 255, 0)      # Cyan
WARNA_POLYGON = (0, 0, 255)    # Magenta
WARNA_TEKS = (255, 0, 255)     # Putih

# 4. Ketebalan garis (piksel) - -1 untuk filled
KETEBALAN_GARIS = 3

# 5. Font untuk teks
FONT = cv2.FONT_HERSHEY_SIMPLEX
UKURAN_FONT = 0.9

# 6. Nama file output
NAMA_FILE_OUTPUT = "hasil_drawing.png"

# ============================================================
# FUNGSI HELPER
# ============================================================

def buat_canvas(ukuran=CANVAS_SIZE, warna=WARNA_BACKGROUND):
    """Membuat canvas kosong untuk menggambar"""
    tinggi, lebar = ukuran
    canvas = np.full((tinggi, lebar, 3), warna, dtype=np.uint8)
    return canvas


def simpan_gambar(gambar, nama_file):
    """Menyimpan gambar ke file"""
    direktori_script = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(direktori_script, "output", "output6")
    os.makedirs(output_dir, exist_ok=True)
    path_output = os.path.join(output_dir, nama_file)
    cv2.imwrite(path_output, gambar)
    print(f"[SUKSES] Gambar disimpan: {path_output}")


# ============================================================
# FUNGSI MENGGAMBAR SHAPES
# ============================================================

def demo_menggambar_garis():
    """
    Mendemonstrasikan cara menggambar garis
    """
    print("\n" + "=" * 60)
    print("MENGGAMBAR GARIS (cv2.line)")
    print("=" * 60)
    
    canvas = buat_canvas()
    
    # Sintaks: cv2.line(img, pt1, pt2, color, thickness, lineType)
    # pt1, pt2: koordinat (x, y)
    
    print("""
Sintaks:
    cv2.line(gambar, titik_awal, titik_akhir, warna, ketebalan, tipe_garis)
    
Parameter:
    - titik_awal: (x1, y1)
    - titik_akhir: (x2, y2)  
    - warna: (B, G, R)
    - ketebalan: piksel
    - tipe_garis: cv2.LINE_AA (antialiased), cv2.LINE_8 (default)
    """)
    
    # Garis horizontal
    cv2.line(canvas, (50, 50), (550, 50), (255, 0, 0), 2)  # Biru
    cv2.putText(canvas, "Horizontal", (260, 45), FONT, 0.5, (255, 255, 255), 1)
    
    # Garis vertikal
    cv2.line(canvas, (300, 100), (300, 350), (0, 255, 0), 2)  # Hijau
    cv2.putText(canvas, "Vertikal", (310, 220), FONT, 0.5, (255, 255, 255), 1)
    
    # Garis diagonal
    cv2.line(canvas, (50, 100), (250, 350), (0, 0, 255), 2)  # Merah
    cv2.putText(canvas, "Diagonal", (100, 250), FONT, 0.5, (255, 255, 255), 1)
    
    # Garis tebal
    cv2.line(canvas, (350, 100), (550, 100), (255, 255, 0), 5)  # Cyan tebal
    cv2.putText(canvas, "Tebal (5px)", (400, 95), FONT, 0.5, (255, 255, 255), 1)
    
    # Garis antialiased vs normal
    cv2.line(canvas, (350, 150), (550, 200), (255, 0, 255), 2, cv2.LINE_AA)  # AA
    cv2.putText(canvas, "LINE_AA", (450, 180), FONT, 0.4, (255, 255, 255), 1)
    
    cv2.line(canvas, (350, 220), (550, 270), (255, 0, 255), 2, cv2.LINE_8)   # Normal
    cv2.putText(canvas, "LINE_8", (450, 250), FONT, 0.4, (255, 255, 255), 1)
    
    return canvas


def demo_menggambar_rectangle():
    """
    Mendemonstrasikan cara menggambar persegi/kotak
    """
    print("\n" + "=" * 60)
    print("MENGGAMBAR PERSEGI (cv2.rectangle)")
    print("=" * 60)
    
    canvas = buat_canvas()
    
    print("""
Sintaks:
    cv2.rectangle(gambar, titik_kiri_atas, titik_kanan_bawah, warna, ketebalan)
    
Parameter:
    - titik_kiri_atas: (x1, y1)
    - titik_kanan_bawah: (x2, y2)
    - ketebalan: -1 untuk filled (diisi penuh)
    """)
    
    # Rectangle outline
    cv2.rectangle(canvas, (50, 50), (200, 150), WARNA_KOTAK, 2)
    cv2.putText(canvas, "Outline", (90, 105), FONT, 0.5, (255, 255, 255), 1)
    
    # Rectangle filled
    cv2.rectangle(canvas, (250, 50), (400, 150), (0, 165, 255), -1)  # Orange filled
    cv2.putText(canvas, "Filled", (295, 105), FONT, 0.5, (0, 0, 0), 1)
    
    # Rectangle dengan ketebalan berbeda
    cv2.rectangle(canvas, (450, 50), (550, 150), (0, 255, 255), 5)  # Tebal
    cv2.putText(canvas, "5px", (485, 105), FONT, 0.5, (255, 255, 255), 1)
    
    # Square (persegi sama sisi)
    cv2.rectangle(canvas, (50, 200), (150, 300), (128, 0, 128), -1)  # Ungu
    cv2.putText(canvas, "Square", (65, 255), FONT, 0.5, (255, 255, 255), 1)
    
    # Rounded rectangle (simulasi dengan lingkaran)
    cv2.rectangle(canvas, (200, 200), (400, 350), (0, 200, 200), 2)
    cv2.putText(canvas, "Rectangle", (260, 280), FONT, 0.5, (255, 255, 255), 1)
    
    return canvas


def demo_menggambar_circle():
    """
    Mendemonstrasikan cara menggambar lingkaran
    """
    print("\n" + "=" * 60)
    print("MENGGAMBAR LINGKARAN (cv2.circle)")
    print("=" * 60)
    
    canvas = buat_canvas()
    
    print("""
Sintaks:
    cv2.circle(gambar, pusat, radius, warna, ketebalan)
    
Parameter:
    - pusat: (x, y) titik pusat
    - radius: jari-jari dalam piksel
    - ketebalan: -1 untuk filled
    """)
    
    # Lingkaran outline
    cv2.circle(canvas, (100, 100), 50, WARNA_LINGKARAN, 2)
    cv2.putText(canvas, "Outline", (70, 100), FONT, 0.4, (255, 255, 255), 1)
    
    # Lingkaran filled
    cv2.circle(canvas, (250, 100), 50, (0, 255, 0), -1)  # Hijau filled
    cv2.putText(canvas, "Filled", (225, 100), FONT, 0.4, (0, 0, 0), 1)
    
    # Lingkaran dengan radius berbeda
    for i, r in enumerate([20, 35, 50, 65, 80]):
        cv2.circle(canvas, (450, 200), r, (255 - i*50, i*50, 128), 2)
    cv2.putText(canvas, "Radius bervariasi", (380, 320), FONT, 0.4, (255, 255, 255), 1)
    
    # Lingkaran konsentris
    for r in range(10, 100, 15):
        cv2.circle(canvas, (150, 280), r, (255, 255, 0), 1)
    cv2.putText(canvas, "Konsentris", (110, 280), FONT, 0.4, (255, 255, 255), 1)
    
    # Titik (lingkaran kecil filled)
    for i in range(10):
        x = 250 + i * 15
        cv2.circle(canvas, (x, 250), 3, (255, 0, 255), -1)
    cv2.putText(canvas, "Titik-titik", (270, 280), FONT, 0.4, (255, 255, 255), 1)
    
    return canvas


def demo_menggambar_ellipse():
    """
    Mendemonstrasikan cara menggambar elips
    """
    print("\n" + "=" * 60)
    print("MENGGAMBAR ELIPS (cv2.ellipse)")
    print("=" * 60)
    
    canvas = buat_canvas()
    
    print("""
Sintaks:
    cv2.ellipse(gambar, pusat, (axis_x, axis_y), sudut, 
                sudut_mulai, sudut_akhir, warna, ketebalan)
    
Parameter:
    - pusat: (x, y)
    - (axis_x, axis_y): panjang sumbu x dan y
    - sudut: rotasi elips (derajat)
    - sudut_mulai, sudut_akhir: untuk menggambar arc (0-360 untuk penuh)
    """)
    
    # Elips horizontal
    cv2.ellipse(canvas, (100, 100), (80, 40), 0, 0, 360, WARNA_ELIPS, 2)
    cv2.putText(canvas, "Horizontal", (60, 100), FONT, 0.4, (255, 255, 255), 1)
    
    # Elips vertikal
    cv2.ellipse(canvas, (280, 100), (40, 80), 0, 0, 360, (0, 255, 255), 2)
    cv2.putText(canvas, "Vertikal", (250, 100), FONT, 0.4, (255, 255, 255), 1)
    
    # Elips dirotasi
    cv2.ellipse(canvas, (450, 100), (80, 40), 45, 0, 360, (255, 0, 255), 2)
    cv2.putText(canvas, "Rotasi 45", (410, 100), FONT, 0.4, (255, 255, 255), 1)
    
    # Elips filled
    cv2.ellipse(canvas, (100, 280), (80, 50), 0, 0, 360, (100, 200, 100), -1)
    cv2.putText(canvas, "Filled", (75, 280), FONT, 0.4, (0, 0, 0), 1)
    
    # Arc (busur) - elips parsial
    cv2.ellipse(canvas, (280, 280), (80, 50), 0, 0, 180, (255, 128, 0), 3)
    cv2.putText(canvas, "Arc 0-180", (240, 280), FONT, 0.4, (255, 255, 255), 1)
    
    cv2.ellipse(canvas, (450, 280), (80, 50), 0, 45, 270, (0, 200, 255), 3)
    cv2.putText(canvas, "Arc 45-270", (410, 280), FONT, 0.4, (255, 255, 255), 1)
    
    return canvas


def demo_menggambar_polygon():
    """
    Mendemonstrasikan cara menggambar polygon
    """
    print("\n" + "=" * 60)
    print("MENGGAMBAR POLYGON (cv2.polylines / cv2.fillPoly)")
    print("=" * 60)
    
    canvas = buat_canvas()
    
    print("""
Sintaks:
    cv2.polylines(gambar, [pts], isClosed, warna, ketebalan)
    cv2.fillPoly(gambar, [pts], warna)
    
Parameter:
    - pts: array koordinat titik-titik [(x1,y1), (x2,y2), ...]
    - isClosed: True untuk menutup polygon
    """)
    
    # Segitiga (outline)
    pts_segitiga = np.array([[100, 150], [50, 250], [150, 250]], np.int32)
    pts_segitiga = pts_segitiga.reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts_segitiga], True, (0, 255, 255), 2)
    cv2.putText(canvas, "Segitiga", (60, 280), FONT, 0.4, (255, 255, 255), 1)
    
    # Pentagon (filled)
    pts_pentagon = []
    center = (280, 200)
    radius = 70
    for i in range(5):
        angle = math.radians(i * 72 - 90)  # -90 untuk titik atas
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        pts_pentagon.append([x, y])
    pts_pentagon = np.array(pts_pentagon, np.int32)
    cv2.fillPoly(canvas, [pts_pentagon], (128, 0, 255))
    cv2.putText(canvas, "Pentagon", (245, 200), FONT, 0.4, (255, 255, 255), 1)
    
    # Hexagon (outline)
    pts_hexagon = []
    center = (450, 200)
    radius = 70
    for i in range(6):
        angle = math.radians(i * 60)
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        pts_hexagon.append([x, y])
    pts_hexagon = np.array(pts_hexagon, np.int32)
    cv2.polylines(canvas, [pts_hexagon], True, (0, 255, 0), 2)
    cv2.putText(canvas, "Hexagon", (420, 200), FONT, 0.4, (255, 255, 255), 1)
    
    # Star (bintang)
    pts_star = []
    center = (100, 350)
    outer_r, inner_r = 40, 20
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        r = outer_r if i % 2 == 0 else inner_r
        x = int(center[0] + r * math.cos(angle))
        y = int(center[1] + r * math.sin(angle))
        pts_star.append([x, y])
    pts_star = np.array(pts_star, np.int32)
    cv2.fillPoly(canvas, [pts_star], (0, 200, 255))
    cv2.putText(canvas, "Star", (80, 320), FONT, 0.4, (255, 255, 255), 1)
    
    # Arrow (panah)
    pts_arrow = np.array([
        [280, 300], [320, 350], [300, 350], 
        [300, 390], [260, 390], [260, 350], [240, 350]
    ], np.int32)
    cv2.fillPoly(canvas, [pts_arrow], (255, 100, 100))
    cv2.putText(canvas, "Arrow", (260, 330), FONT, 0.4, (255, 255, 255), 1)
    
    return canvas


def demo_menulis_teks():
    """
    Mendemonstrasikan cara menulis teks pada gambar
    """
    print("\n" + "=" * 60)
    print("MENULIS TEKS (cv2.putText)")
    print("=" * 60)
    
    canvas = buat_canvas()
    
    print("""
Sintaks:
    cv2.putText(gambar, teks, posisi, font, skala, warna, ketebalan, lineType)
    
Font yang tersedia:
    - cv2.FONT_HERSHEY_SIMPLEX
    - cv2.FONT_HERSHEY_PLAIN
    - cv2.FONT_HERSHEY_DUPLEX
    - cv2.FONT_HERSHEY_COMPLEX
    - cv2.FONT_HERSHEY_TRIPLEX
    - cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    - cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    - cv2.FONT_ITALIC (flag untuk italic)
    """)
    
    # Berbagai font
    fonts = [
        (cv2.FONT_HERSHEY_SIMPLEX, "SIMPLEX"),
        (cv2.FONT_HERSHEY_PLAIN, "PLAIN"),
        (cv2.FONT_HERSHEY_DUPLEX, "DUPLEX"),
        (cv2.FONT_HERSHEY_COMPLEX, "COMPLEX"),
        (cv2.FONT_HERSHEY_TRIPLEX, "TRIPLEX"),
        (cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, "SCRIPT_SIMPLEX"),
        (cv2.FONT_HERSHEY_SCRIPT_COMPLEX, "SCRIPT_COMPLEX"),
    ]
    
    y = 40
    for font, nama in fonts:
        cv2.putText(canvas, f"{nama}", (50, y), font, 0.7, WARNA_TEKS, 1)
        y += 45
    
    # Teks dengan ukuran berbeda
    cv2.putText(canvas, "Kecil (0.5)", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(canvas, "Medium (1.0)", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 1)
    cv2.putText(canvas, "Besar", (350, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
    
    # Teks dengan ketebalan berbeda
    cv2.putText(canvas, "Thin (1)", (350, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(canvas, "Bold (2)", (350, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(canvas, "Extra Bold (3)", (350, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 3)
    
    # Teks dengan outline (efek shadow)
    teks = "Shadow Effect"
    posisi = (350, 360)
    # Shadow (background)
    cv2.putText(canvas, teks, (posisi[0]+2, posisi[1]+2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # Foreground
    cv2.putText(canvas, teks, posisi, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return canvas


def demo_anotasi_gambar():
    """
    Contoh penggunaan praktis: Anotasi gambar
    """
    print("\n" + "=" * 60)
    print("CONTOH PRAKTIS: ANOTASI GAMBAR")
    print("=" * 60)
    
    # Buat gambar simulasi object detection
    canvas = buat_canvas((400, 600), (50, 50, 50))
    
    # Simulasi bounding box untuk object detection
    deteksi = [
        {"nama": "Person", "bbox": (50, 50, 150, 250), "conf": 0.95, "warna": (0, 255, 0)},
        {"nama": "Car", "bbox": (200, 150, 350, 300), "conf": 0.87, "warna": (255, 0, 0)},
        {"nama": "Dog", "bbox": (400, 200, 550, 350), "conf": 0.72, "warna": (0, 0, 255)},
    ]
    
    for obj in deteksi:
        x1, y1, x2, y2 = obj["bbox"]
        nama = obj["nama"]
        conf = obj["conf"]
        warna = obj["warna"]
        
        # Gambar bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), warna, 2)
        
        # Gambar label background
        label = f"{nama}: {conf:.0%}"
        (lebar_teks, tinggi_teks), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(canvas, (x1, y1 - tinggi_teks - 10), 
                     (x1 + lebar_teks + 10, y1), warna, -1)
        
        # Tulis label
        cv2.putText(canvas, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Tambah judul
    cv2.putText(canvas, "Object Detection Result", (180, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Tambah info
    cv2.putText(canvas, "Objects: 3 | FPS: 30", (420, 390), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    return canvas


def tampilkan_semua_demo():
    """
    Menampilkan semua demo dalam satu figure
    """
    # Jalankan semua demo
    canvas_garis = demo_menggambar_garis()
    canvas_rectangle = demo_menggambar_rectangle()
    canvas_circle = demo_menggambar_circle()
    canvas_ellipse = demo_menggambar_ellipse()
    canvas_polygon = demo_menggambar_polygon()
    canvas_teks = demo_menulis_teks()
    canvas_anotasi = demo_anotasi_gambar()
    
    # Tampilkan dalam grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(cv2.cvtColor(canvas_garis, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Garis (cv2.line)")
    
    axes[0, 1].imshow(cv2.cvtColor(canvas_rectangle, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. Rectangle (cv2.rectangle)")
    
    axes[0, 2].imshow(cv2.cvtColor(canvas_circle, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("3. Circle (cv2.circle)")
    
    axes[1, 0].imshow(cv2.cvtColor(canvas_ellipse, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("4. Ellipse (cv2.ellipse)")
    
    axes[1, 1].imshow(cv2.cvtColor(canvas_polygon, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. Polygon (cv2.polylines)")
    
    axes[1, 2].imshow(cv2.cvtColor(canvas_teks, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Teks (cv2.putText)")
    
    axes[2, 0].imshow(cv2.cvtColor(canvas_anotasi, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title("7. Contoh: Object Detection Annotation")
    
    # Kosongkan 2 subplot terakhir
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    for ax in axes.flat[:7]:
        ax.axis('off')
    
    plt.suptitle("Demo Menggambar Shapes dan Teks dengan OpenCV", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return canvas_anotasi


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    """Fungsi utama program"""
    
    print("\n" + "=" * 60)
    print("PRAKTIKUM 6: MENGGAMBAR SHAPES DAN TEKS")
    print("=" * 60)
    
    # Tampilkan semua demo
    hasil_anotasi = tampilkan_semua_demo()
    
    # Simpan hasil anotasi
    simpan_gambar(hasil_anotasi, NAMA_FILE_OUTPUT)
    
    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN MENGGAMBAR SHAPES")
    print("=" * 60)
    print("""
FUNGSI MENGGAMBAR:
├── cv2.line(img, pt1, pt2, color, thickness)
├── cv2.rectangle(img, pt1, pt2, color, thickness)
├── cv2.circle(img, center, radius, color, thickness)
├── cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
├── cv2.polylines(img, [pts], isClosed, color, thickness)
├── cv2.fillPoly(img, [pts], color)
└── cv2.putText(img, text, org, font, fontScale, color, thickness)

PARAMETER UMUM:
├── thickness = -1  → Filled (diisi penuh)
├── thickness > 0   → Outline dengan ketebalan tertentu
├── lineType = cv2.LINE_AA → Antialiased (halus)
└── lineType = cv2.LINE_8  → Default

KOORDINAT:
├── OpenCV menggunakan (x, y) untuk titik
├── x → horizontal (kolom), y → vertikal (baris)
└── Origin (0,0) di sudut kiri atas

TIPS:
- Gunakan cv2.getTextSize() untuk mendapatkan ukuran teks
- Kombinasikan shapes untuk membuat anotasi
- Untuk animasi, gambar ulang setiap frame
""")


# Jalankan program utama
if __name__ == "__main__":
    main()
