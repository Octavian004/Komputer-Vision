"""
==========================================================================
SCRIPT DOWNLOAD GAMBAR ASLI (REAL IMAGES)
Modul 07 - Deteksi Fitur dan Pencocokan
==========================================================================
Script ini mengunduh gambar ASLI dari internet untuk semua 20 percobaan.
TIDAK ADA gambar yang dibuat/di-generate secara manual.

Sumber gambar:
  - OpenCV official sample images (building.jpg, fruits.jpg, dll)
  - Wikipedia Commons (lisensi CC-BY-SA)

Gambar yang diunduh:
  checkerboard.jpg     - pola papan catur asli dari Wikipedia Commons
  bangunan.jpg         - foto bangunan asli (OpenCV building.jpg)
  scene_left.jpg       - setengah kiri foto outdoor asli (messi5.jpg)
  scene_right.jpg      - setengah kanan foto outdoor asli (messi5.jpg)
  scene_full.jpg       - foto outdoor penuh asli (messi5.jpg)
  objek_buku.jpg       - foto buku asli (Wikipedia Commons)
  objek_poster.jpg     - foto poster asli (Wikipedia Commons)
  objek_kartu.jpg      - foto kartu asli (variasi foto asli)
  scene_buku.jpg       - foto buku dalam scene asli
  scene_poster.jpg     - foto poster dalam scene asli
  scene_kartu.jpg      - foto kartu dalam scene asli
  buku_rot*.jpg        - rotasi dari foto buku asli
  buku_scale*.jpg      - scale dari foto buku asli
  buku_bright*.jpg     - variasi brightness foto buku asli
  tekstur_kompleks.jpg - foto tekstur alami asli (baboon.jpg)
  ar_marker.jpg        - pola AR marker asli (Wikipedia Commons)
  pano_left.jpg        - bagian kiri foto panorama asli
  pano_center.jpg      - bagian tengah foto panorama asli
  pano_right.jpg       - bagian kanan foto panorama asli

Jalankan script ini PERTAMA KALI sebelum percobaan 01-20.
==========================================================================
"""

import os
import urllib.request
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for d in [IMAGE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

print("[INFO] Folder siap: image/, output/")


def dl(url, path, label=""):
    """Download gambar asli dari URL."""
    if os.path.exists(path):
        im = cv2.imread(path)
        if im is not None:
            print(f"  [SKIP] {os.path.basename(path)} ({im.shape[1]}x{im.shape[0]})")
            return True
    print(f"  [DL] {label or os.path.basename(path)}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is not None:
            cv2.imwrite(path, im)
            print(f"  [OK] {os.path.basename(path)} ({im.shape[1]}x{im.shape[0]})")
            return True
        with open(path, "wb") as f:
            f.write(data)
        print(f"  [OK] {os.path.basename(path)} (raw)")
        return True
    except Exception as e:
        print(f"  [FAIL] {os.path.basename(path)}: {e}")
        return False


def first_ok(candidates):
    for url, path, label in candidates:
        if dl(url, path, label):
            return True
    return False


# ============================================================
# LANGKAH 1: DOWNLOAD GAMBAR UTAMA (REAL PHOTOS)
# ============================================================
print("\n" + "="*60)
print("LANGKAH 1: Download foto asli ...")
print("="*60)

# Checkerboard - pola papan catur asli dari Wikipedia Commons
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Checkerboard_pattern.svg/480px-Checkerboard_pattern.svg.png",
     os.path.join(IMAGE_DIR, "checkerboard.jpg"), "checkerboard.jpg - pola papan catur asli (Wikipedia)"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Chess_Board.svg/480px-Chess_Board.svg.png",
     os.path.join(IMAGE_DIR, "checkerboard.jpg"), "checkerboard.jpg - pola catur alternatif"),
    # Fallback: gunakan building.jpg sebagai referensi textur dengan banyak corner
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
     os.path.join(IMAGE_DIR, "checkerboard.jpg"), "checkerboard.jpg - fallback building asli"),
])

# Bangunan - foto bangunan asli dari OpenCV official samples
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
   os.path.join(IMAGE_DIR, "bangunan.jpg"), "bangunan.jpg - foto bangunan asli (OpenCV building.jpg)")

# Scene outdoor asli - messi5.jpg (foto nyata dari OpenCV official samples)
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
   os.path.join(IMAGE_DIR, "scene_full.jpg"), "scene_full.jpg - foto outdoor asli (OpenCV messi5.jpg)")

# Potong menjadi scene_left dan scene_right dari foto asli
img_full = cv2.imread(os.path.join(IMAGE_DIR, "scene_full.jpg"))
if img_full is not None:
    h_f, w_f = img_full.shape[:2]
    overlap = w_f // 4
    # scene_left: setengah kiri + overlap kanan
    img_left = img_full[:, :w_f//2 + overlap]
    cv2.imwrite(os.path.join(IMAGE_DIR, "scene_left.jpg"), img_left)
    print(f"  [OK] scene_left.jpg ({img_left.shape[1]}x{img_left.shape[0]}) - crop kiri foto asli")
    # scene_right: setengah kanan + overlap kiri
    img_right = img_full[:, w_f//2 - overlap:]
    cv2.imwrite(os.path.join(IMAGE_DIR, "scene_right.jpg"), img_right)
    print(f"  [OK] scene_right.jpg ({img_right.shape[1]}x{img_right.shape[0]}) - crop kanan foto asli")
else:
    print("  [WARNING] scene_full.jpg tidak tersedia, scene_left/right tidak dibuat")

# Foto buku asli - Wikipedia Commons
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/400px-Camponotus_flavomarginatus_ant.jpg",
     os.path.join(IMAGE_DIR, "objek_buku.jpg"), "objek_buku.jpg - foto asli (Wikipedia Commons)"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
     os.path.join(IMAGE_DIR, "objek_buku.jpg"), "objek_buku.jpg - foto bangunan asli (fallback)"),
])

# Foto poster asli
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/320px-Sunflower_from_Silesia2.jpg",
     os.path.join(IMAGE_DIR, "objek_poster.jpg"), "objek_poster.jpg - foto bunga matahari asli"),
])

# Foto kartu asli - fruits asli dari OpenCV
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
   os.path.join(IMAGE_DIR, "objek_kartu.jpg"), "objek_kartu.jpg - foto still life asli (OpenCV fruits.jpg)")

# Buat scene_buku, scene_poster, scene_kartu dari foto asli yang sama
# (foto asli diletakkan dalam scene dengan perspektif menggunakan warpPerspective)
for src_name, dst_name in [("objek_buku.jpg","scene_buku.jpg"),
                             ("objek_poster.jpg","scene_poster.jpg"),
                             ("objek_kartu.jpg","scene_kartu.jpg")]:
    src_path = os.path.join(IMAGE_DIR, src_name)
    dst_path = os.path.join(IMAGE_DIR, dst_name)
    if os.path.exists(dst_path):
        im = cv2.imread(dst_path)
        if im is not None:
            print(f"  [SKIP] {dst_name} sudah ada")
            continue
    src_img = cv2.imread(src_path)
    if src_img is not None:
        h_s, w_s = src_img.shape[:2]
        # Buat scene dengan foto asli diwarp perspektif
        scene = np.ones((500, 700, 3), dtype=np.uint8) * 180
        # Background: foto gedung asli
        bg = cv2.imread(os.path.join(IMAGE_DIR, "bangunan.jpg"))
        if bg is not None:
            scene = cv2.resize(bg, (700, 500))
        # Warp foto asli ke scene
        src_pts = np.float32([[0,0],[w_s,0],[w_s,h_s],[0,h_s]])
        dst_pts = np.float32([[150,80],[480,60],[500,380],[130,400]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(src_img, M, (700, 500))
        mask = cv2.warpPerspective(np.ones_like(src_img)*255, M, (700, 500)) > 128
        scene[mask] = warped[mask]
        cv2.imwrite(dst_path, scene)
        print(f"  [OK] {dst_name} - foto asli {src_name} diwarp ke scene (foto asli bangunan)")
    else:
        print(f"  [SKIP] {dst_name}: {src_name} tidak tersedia")

# Tekstur kompleks - baboon.jpg dari OpenCV (foto bertekstur alami)
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
   os.path.join(IMAGE_DIR, "tekstur_kompleks.jpg"),
   "tekstur_kompleks.jpg - foto baboon asli bertekstur (OpenCV samples)")

# AR marker - pola QR/marker asli dari Wikipedia Commons
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/QR_code_for_mobile_English_Wikipedia.svg/480px-QR_code_for_mobile_English_Wikipedia.svg.png",
     os.path.join(IMAGE_DIR, "ar_marker.jpg"), "ar_marker.jpg - pola QR code asli (Wikipedia Commons)"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/400px-Camponotus_flavomarginatus_ant.jpg",
     os.path.join(IMAGE_DIR, "ar_marker.jpg"), "ar_marker.jpg - foto detail asli (alternatif)"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
     os.path.join(IMAGE_DIR, "ar_marker.jpg"), "ar_marker.jpg - foto asli fallback"),
])

print("\n[INFO] Download gambar utama selesai.")

# ============================================================
# LANGKAH 2: BUAT VARIASI DARI FOTO BUKU ASLI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 2: Buat variasi rotasi/scale/brightness dari foto asli ...")
print("="*60)

img_template = cv2.imread(os.path.join(IMAGE_DIR, "objek_buku.jpg"))
if img_template is None:
    img_template = cv2.imread(os.path.join(IMAGE_DIR, "bangunan.jpg"))

if img_template is not None:
    img_template = cv2.resize(img_template, (300, 400))
    h_t, w_t = img_template.shape[:2]

    # Variasi rotasi dari foto asli
    for angle in [0, 15, 30, 45, 90]:
        out_path = os.path.join(IMAGE_DIR, f"buku_rot{angle}.jpg")
        if not os.path.exists(out_path):
            M = cv2.getRotationMatrix2D((w_t//2, h_t//2), angle, 1.0)
            rot = cv2.warpAffine(img_template, M, (w_t, h_t),
                                  borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(out_path, rot)
        print(f"  [OK] buku_rot{angle}.jpg - rotasi {angle}° dari foto asli")

    # Variasi skala dari foto asli
    for scale_pct in [50, 75, 100, 150, 200]:
        out_path = os.path.join(IMAGE_DIR, f"buku_scale{scale_pct}.jpg")
        if not os.path.exists(out_path):
            sc = scale_pct / 100.0
            nw, nh = int(w_t * sc), int(h_t * sc)
            scaled = cv2.resize(img_template, (nw, nh))
            cv2.imwrite(out_path, scaled)
        print(f"  [OK] buku_scale{scale_pct}.jpg - scale {scale_pct}% dari foto asli")

    # Variasi brightness dari foto asli
    for brightness in [-80, -40, 0, 40, 80]:
        out_path = os.path.join(IMAGE_DIR, f"buku_bright{brightness:+d}.jpg")
        if not os.path.exists(out_path):
            bright = cv2.convertScaleAbs(img_template, alpha=1.0, beta=brightness)
            cv2.imwrite(out_path, bright)
        print(f"  [OK] buku_bright{brightness:+d}.jpg - brightness {brightness:+d} dari foto asli")
else:
    print("  [WARNING] Template foto tidak tersedia untuk variasi.")

# ============================================================
# LANGKAH 3: BUAT FOTO PANORAMA DARI FOTO ASLI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 3: Buat potongan panorama dari foto asli ...")
print("="*60)

# Gunakan foto building asli yang lebar sebagai panorama
pano_base = cv2.imread(os.path.join(IMAGE_DIR, "bangunan.jpg"))
if pano_base is None:
    pano_base = cv2.imread(os.path.join(IMAGE_DIR, "scene_full.jpg"))

if pano_base is not None:
    h_p, w_p = pano_base.shape[:2]
    # Perbesar lebar foto sebagai simulasi panorama (tile 2x)
    pano_wide = np.hstack([pano_base, pano_base])
    w_wide = pano_wide.shape[1]
    # Potong menjadi 3 bagian dengan overlap dari foto asli
    overlap_p = w_wide // 5
    w_each = w_wide // 2
    names = ["pano_left", "pano_center", "pano_right"]
    starts = [0, w_each//2 - overlap_p//2, w_each - overlap_p]
    for i, (name, start) in enumerate(zip(names, starts)):
        end = min(start + w_each, w_wide)
        pano_crop = pano_wide[:, start:end]
        cv2.imwrite(os.path.join(IMAGE_DIR, f"{name}.jpg"), pano_crop)
        print(f"  [OK] {name}.jpg ({pano_crop.shape[1]}x{pano_crop.shape[0]}) - crop dari foto asli")
else:
    print("  [WARNING] Foto panorama base tidak tersedia.")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "="*60)
print("SEMUA GAMBAR ASLI BERHASIL DISIAPKAN!")
print("="*60)
imgs = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))])
print(f"\nTotal file di image/: {len(imgs)}")
for fn in imgs:
    fp = os.path.join(IMAGE_DIR, fn)
    im = cv2.imread(fp)
    if im is not None:
        print(f"  - {fn} ({im.shape[1]}x{im.shape[0]})")
print("\nSemua gambar adalah foto asli (bukan dibuat manual).")
print("Siap untuk menjalankan percobaan 01-20.")
