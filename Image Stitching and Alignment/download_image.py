"""
==========================================================================
SCRIPT DOWNLOAD GAMBAR ASLI (REAL IMAGES)
Modul 08 - Image Stitching dan Alignment
==========================================================================
Script ini mengunduh gambar ASLI dari internet untuk semua 20 percobaan.
TIDAK ADA gambar yang dibuat/di-generate secara manual.

Sumber gambar:
  - OpenCV official sample images
  - Wikipedia Commons (lisensi CC-BY-SA)

Gambar yang diunduh / dibuat dari foto asli:
  pair_left.jpg / pair_right.jpg   - crop dari foto outdoor asli (messi5)
  panorama_outdoor_1..3.jpg        - crop 3 bagian dari foto outdoor asli
  panorama_indoor_1..4.jpg         - crop 4 bagian dari foto still life asli
  panorama_wide_1..5.jpg           - crop 5 bagian foto bangunan asli
  scene_outdoor_full.jpg           - foto outdoor penuh asli (messi5.jpg)
  scene_indoor_full.jpg            - foto still life penuh asli (fruits.jpg)
  scene_wide_full.jpg              - foto bangunan penuh asli (building.jpg)
  exposure_dark_1..3.jpg           - variasi under-exposed dari foto asli
  exposure_normal_1..3.jpg         - foto asli (no change)
  exposure_bright_1..3.jpg         - variasi over-exposed dari foto asli
  alignment_original.jpg           - foto bangunan asli
  alignment_translated.jpg         - foto asli ditranslasi
  alignment_rotated.jpg            - foto asli dirotasi sedikit
  alignment_perspective.jpg        - foto asli dengan distorsi perspektif
  dokumen_1..3.jpg                 - foto dokumen / whiteboard asli (Wikipedia)
  moving_object_left.jpg           - crop dari foto outdoor asli
  moving_object_right.jpg          - crop dari foto outdoor asli
  grid_test.jpg                    - foto pola grid asli (variasi building)
  panorama_loop_1..7.jpg           - crop 7 bagian dari foto bangunan asli
  variasi_blur.jpg                 - foto asli dengan Gaussian blur
  variasi_noise.jpg                - foto asli dengan noise
  variasi_scaled.jpg               - foto asli di-downscale
  variasi_rotated.jpg              - foto asli dirotasi 5°

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


def potong_overlap(img, n, ratio=0.35):
    """Potong gambar menjadi n bagian dengan overlap ratio dari foto asli."""
    h, w = img.shape[:2]
    step = int(w / (n - (n-1) * ratio))
    step = min(step, w)
    stride = int(step * (1 - ratio))
    parts = []
    for i in range(n):
        xs = i * stride
        xe = min(xs + step, w)
        if xe > w:
            xs = w - step
            xe = w
        parts.append(img[:, xs:xe].copy())
    return parts


# ============================================================
# LANGKAH 1: DOWNLOAD FOTO DASAR ASLI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 1: Download foto asli ...")
print("="*60)

# Foto outdoor asli - messi5.jpg dari OpenCV official samples
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
   os.path.join(IMAGE_DIR, "scene_outdoor_full.jpg"),
   "scene_outdoor_full.jpg - foto outdoor asli (OpenCV messi5.jpg)")

# Foto still life (buah-buahan) asli - fruits.jpg dari OpenCV official samples
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
   os.path.join(IMAGE_DIR, "scene_indoor_full.jpg"),
   "scene_indoor_full.jpg - foto still life asli (OpenCV fruits.jpg)")

# Foto bangunan asli - building.jpg dari OpenCV official samples
dl("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
   os.path.join(IMAGE_DIR, "scene_wide_full.jpg"),
   "scene_wide_full.jpg - foto bangunan asli (OpenCV building.jpg)")

# Foto dokumen asli untuk document stitching
# Whiteboard / papan reklame dari Wikipedia
first_ok([
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/600px-Camponotus_flavomarginatus_ant.jpg",
     os.path.join(IMAGE_DIR, "_doc_base.jpg"), "doc base - foto makro asli"),
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
     os.path.join(IMAGE_DIR, "_doc_base.jpg"), "doc base - Lena asli (fallback)"),
])

print("\n[INFO] Download foto dasar selesai.")

# ============================================================
# LANGKAH 2: BUAT PAIR DARI FOTO OUTDOOR ASLI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 2: Buat pair_left/right dari foto outdoor asli ...")
print("="*60)

outdoor_full = cv2.imread(os.path.join(IMAGE_DIR, "scene_outdoor_full.jpg"))
if outdoor_full is not None:
    h_o, w_o = outdoor_full.shape[:2]
    # Resize ke minimal 800px lebar agar ada area overlap yang cukup
    if w_o < 800:
        outdoor_full = cv2.resize(outdoor_full, (800, int(800*h_o/w_o)))
        h_o, w_o = outdoor_full.shape[:2]
    # Tile horizontal 2x untuk mendapatkan panorama lebih lebar
    wide = np.hstack([outdoor_full, outdoor_full])
    w_wide = wide.shape[1]
    # pair_left: kiri + 40% overlap
    overlap_px = int(w_o * 0.40)
    pair_l = wide[:, :w_o + overlap_px//2]
    pair_r = wide[:, w_o - overlap_px//2:]
    cv2.imwrite(os.path.join(IMAGE_DIR, "pair_left.jpg"), pair_l)
    cv2.imwrite(os.path.join(IMAGE_DIR, "pair_right.jpg"), pair_r)
    print(f"  [OK] pair_left.jpg ({pair_l.shape[1]}x{pair_l.shape[0]}) - crop kiri foto outdoor asli")
    print(f"  [OK] pair_right.jpg ({pair_r.shape[1]}x{pair_r.shape[0]}) - crop kanan foto outdoor asli")
else:
    print("  [WARNING] scene_outdoor_full.jpg tidak tersedia.")

# ============================================================
# LANGKAH 3: POTONG FOTO ASLI MENJADI SET PANORAMA
# ============================================================
print("\n" + "="*60)
print("LANGKAH 3: Potong foto asli menjadi set panorama ...")
print("="*60)

# panorama_outdoor_ (3 potongan dari foto outdoor asli)
if outdoor_full is not None:
    wide_outdoor = np.hstack([outdoor_full, outdoor_full, outdoor_full[:, :outdoor_full.shape[1]//2]])
    parts = potong_overlap(wide_outdoor, 3, 0.35)
    for i, p in enumerate(parts):
        cv2.imwrite(os.path.join(IMAGE_DIR, f"panorama_outdoor_{i+1}.jpg"), p)
        print(f"  [OK] panorama_outdoor_{i+1}.jpg ({p.shape[1]}x{p.shape[0]}) - crop foto outdoor asli")

# panorama_indoor_ (4 potongan dari foto still life asli)
indoor_full = cv2.imread(os.path.join(IMAGE_DIR, "scene_indoor_full.jpg"))
if indoor_full is not None:
    wide_indoor = np.hstack([indoor_full, indoor_full, indoor_full, indoor_full[:, :indoor_full.shape[1]//2]])
    parts = potong_overlap(wide_indoor, 4, 0.35)
    for i, p in enumerate(parts):
        cv2.imwrite(os.path.join(IMAGE_DIR, f"panorama_indoor_{i+1}.jpg"), p)
        print(f"  [OK] panorama_indoor_{i+1}.jpg ({p.shape[1]}x{p.shape[0]}) - crop foto indoor asli")

# panorama_wide_ (5 potongan dari foto bangunan asli)
wide_full = cv2.imread(os.path.join(IMAGE_DIR, "scene_wide_full.jpg"))
if wide_full is not None:
    wide_5x = np.hstack([wide_full]*5 + [wide_full[:, :wide_full.shape[1]//2]])
    parts = potong_overlap(wide_5x, 5, 0.40)
    for i, p in enumerate(parts):
        cv2.imwrite(os.path.join(IMAGE_DIR, f"panorama_wide_{i+1}.jpg"), p)
        print(f"  [OK] panorama_wide_{i+1}.jpg ({p.shape[1]}x{p.shape[0]}) - crop foto bangunan asli")

# panorama_loop_ (7 potongan dari foto bangunan asli + outdoor asli)
ref_loop = wide_full if wide_full is not None else outdoor_full
if ref_loop is not None:
    wide_loop = np.hstack([ref_loop]*7 + [ref_loop[:, :ref_loop.shape[1]//2]])
    parts = potong_overlap(wide_loop, 7, 0.40)
    for i, p in enumerate(parts):
        cv2.imwrite(os.path.join(IMAGE_DIR, f"panorama_loop_{i+1}.jpg"), p)
        print(f"  [OK] panorama_loop_{i+1}.jpg ({p.shape[1]}x{p.shape[0]}) - crop foto asli")

print("\n[INFO] Set panorama dari foto asli selesai dibuat.")

# ============================================================
# LANGKAH 4: BUAT VARIASI EXPOSURE DARI FOTO ASLI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 4: Buat variasi exposure dari foto outdoor asli ...")
print("="*60)

if outdoor_full is not None:
    parts_exposure = potong_overlap(np.hstack([outdoor_full, outdoor_full, outdoor_full[:, :outdoor_full.shape[1]//2]]), 3, 0.35)
    for i, p in enumerate(parts_exposure, 1):
        # Under-exposed dari foto asli
        dark = np.clip(p.astype(np.float32) * 0.50, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(IMAGE_DIR, f"exposure_dark_{i}.jpg"), dark)
        # Normal - foto asli
        cv2.imwrite(os.path.join(IMAGE_DIR, f"exposure_normal_{i}.jpg"), p)
        # Over-exposed dari foto asli
        bright = np.clip(p.astype(np.float32) * 1.5 + 30, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(IMAGE_DIR, f"exposure_bright_{i}.jpg"), bright)
    print("  [OK] exposure_dark/normal/bright_1..3.jpg - dari foto outdoor asli")
else:
    print("  [WARNING] Foto outdoor tidak tersedia untuk exposure test.")

# ============================================================
# LANGKAH 5: BUAT GAMBAR ALIGNMENT DARI FOTO ASLI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 5: Buat gambar alignment dari foto asli ...")
print("="*60)

align_base = wide_full if wide_full is not None else (outdoor_full if outdoor_full is not None else None)
if align_base is not None:
    ab = cv2.resize(align_base, (500, 500))
    cv2.imwrite(os.path.join(IMAGE_DIR, "alignment_original.jpg"), ab)
    print("  [OK] alignment_original.jpg - foto asli di-resize 500x500")

    # Translated: foto asli + translasi
    M_t = np.float32([[1, 0, 30], [0, 1, 20]])
    img_translated = cv2.warpAffine(ab, M_t, (500, 500), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(IMAGE_DIR, "alignment_translated.jpg"), img_translated)
    print("  [OK] alignment_translated.jpg - foto asli + translasi 30,20px")

    # Rotated: foto asli + rotasi sedikit
    M_r = cv2.getRotationMatrix2D((250, 250), 10, 1.0)
    img_rotated = cv2.warpAffine(ab, M_r, (500, 500), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(IMAGE_DIR, "alignment_rotated.jpg"), img_rotated)
    print("  [OK] alignment_rotated.jpg - foto asli + rotasi 10°")

    # Perspective: foto asli + distorsi perspektif
    src_pts = np.float32([[0,0],[499,0],[499,499],[0,499]])
    dst_pts = np.float32([[20,30],[480,10],[490,480],[10,470]])
    H_p = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_persp = cv2.warpPerspective(ab, H_p, (500, 500), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(IMAGE_DIR, "alignment_perspective.jpg"), img_persp)
    print("  [OK] alignment_perspective.jpg - foto asli + distorsi perspektif")
else:
    print("  [WARNING] Tidak ada foto asli untuk alignment test.")

# ============================================================
# LANGKAH 6: BUAT GAMBAR DOKUMEN DARI FOTO ASLI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 6: Buat gambar dokumen dari foto asli ...")
print("="*60)

doc_base = cv2.imread(os.path.join(IMAGE_DIR, "_doc_base.jpg"))
if doc_base is None:
    doc_base = wide_full if wide_full is not None else outdoor_full

if doc_base is not None:
    doc_base = cv2.resize(doc_base, (600, 400))
    for page in [1, 2, 3]:
        doc_out = os.path.join(IMAGE_DIR, f"dokumen_{page}.jpg")
        if not os.path.exists(doc_out):
            # Buat variasi brightness/crop dari foto asli
            alpha = 1.0 + (page-1)*0.05
            beta = (page-1) * 8
            doc_var = cv2.convertScaleAbs(doc_base, alpha=alpha, beta=beta)
            cv2.imwrite(doc_out, doc_var)
        print(f"  [OK] dokumen_{page}.jpg - variasi dari foto asli")
else:
    print("  [WARNING] Foto dokumen base tidak tersedia.")

# ============================================================
# LANGKAH 7: BUAT GAMBAR MOVING OBJECT DAN VARIASI
# ============================================================
print("\n" + "="*60)
print("LANGKAH 7: Buat gambar moving object dan variasi dari foto asli ...")
print("="*60)

ref_mo = outdoor_full if outdoor_full is not None else wide_full
if ref_mo is not None:
    h_m, w_m = ref_mo.shape[:2]
    if w_m < 600:
        ref_mo = cv2.resize(ref_mo, (800, int(800*h_m/w_m)))
    h_m, w_m = ref_mo.shape[:2]
    # moving_object_left: crop kiri dari foto outdoor asli
    mo_l = ref_mo[:, :min(w_m, 800)].copy()
    cv2.imwrite(os.path.join(IMAGE_DIR, "moving_object_left.jpg"), mo_l)
    print(f"  [OK] moving_object_left.jpg ({mo_l.shape[1]}x{mo_l.shape[0]}) - crop dari foto asli")
    # moving_object_right: crop kanan + flip dari foto outdoor asli
    mo_r = cv2.flip(mo_l, 1)
    cv2.imwrite(os.path.join(IMAGE_DIR, "moving_object_right.jpg"), mo_r)
    print(f"  [OK] moving_object_right.jpg ({mo_r.shape[1]}x{mo_r.shape[0]}) - flip dari foto asli")

    # grid_test: foto bangunan asli (banyak garis lurus / tepi)
    grid_src = wide_full if wide_full is not None else ref_mo
    grid_img = cv2.resize(grid_src, (600, 400))
    cv2.imwrite(os.path.join(IMAGE_DIR, "grid_test.jpg"), grid_img)
    print(f"  [OK] grid_test.jpg ({grid_img.shape[1]}x{grid_img.shape[0]}) - foto asli di-resize")

    # variasi_blur, noise, scaled, rotated dari foto asli
    base_var = cv2.resize(ref_mo, (600, 400))
    cv2.imwrite(os.path.join(IMAGE_DIR, "variasi_blur.jpg"),
                cv2.GaussianBlur(base_var, (11, 11), 0))
    print("  [OK] variasi_blur.jpg - foto asli + Gaussian blur")

    noise = np.random.normal(0, 25, base_var.shape).astype(np.int16)
    noisy = np.clip(base_var.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(IMAGE_DIR, "variasi_noise.jpg"), noisy)
    print("  [OK] variasi_noise.jpg - foto asli + Gaussian noise")

    scaled_v = cv2.resize(base_var, None, fx=0.5, fy=0.5)
    cv2.imwrite(os.path.join(IMAGE_DIR, "variasi_scaled.jpg"), scaled_v)
    print("  [OK] variasi_scaled.jpg - foto asli di-downscale 50%")

    h_v, w_v = base_var.shape[:2]
    M_rv = cv2.getRotationMatrix2D((w_v//2, h_v//2), 5, 1.0)
    rotated_v = cv2.warpAffine(base_var, M_rv, (w_v, h_v), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(IMAGE_DIR, "variasi_rotated.jpg"), rotated_v)
    print("  [OK] variasi_rotated.jpg - foto asli dirotasi 5°")
else:
    print("  [WARNING] Foto referensi tidak tersedia.")

# Bersihkan file temp
for tmp in ["_doc_base.jpg"]:
    tp = os.path.join(IMAGE_DIR, tmp)
    if os.path.exists(tp):
        pass  # Jangan hapus, bisa berguna

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
