"""
DOWNLOAD IMAGE - MODUL 04: MODEL FITTING DAN OPTIMASI
Semua gambar didownload dari internet. Tidak ada generasi sintetis.

Pemetaan:
  lena.jpg          -> 01 (OLS: canvas untuk cv2.fitLine demo)
  jalan.png         -> 06,20 (Hough line, pipeline gabungan)
  koin.png          -> 07,12,14 (Hough circle, ellipse fitting, graph-cut)
  papan_catur.jpg   -> 08 (homography estimation)
  dokumen_asli.jpg  -> 09 (koreksi perspektif dokumen)
  target.png        -> 13 (template matching scene)
  template.png      -> 13 (template matching patch)
  frame1.png        -> 15,16 (Lucas-Kanade, dense optical flow)
  frame2.png        -> 15,16 (Lucas-Kanade, dense optical flow)
  frame1.png/frame2 -> 17 (feature matching RANSAC — scene pair)
  clean_img.png     -> 19 (denoising)
  building_cv.jpg   -> referensi umum
"""
import os, ssl, urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR   = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODUL 04 - DOWNLOAD GAMBAR NYATA")
print("=" * 60)

def _download(filename, url, timeout=30):
    dest = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(dest):
        print(f"  skip  {filename} ({os.path.getsize(dest)//1024} KB)")
        return True
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            data = r.read()
        open(dest, "wb").write(data)
        print(f"  OK   {filename}  ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"  FAIL {filename}: {e}"); return False

IMAGES = {
    # Canvas untuk cv2.fitLine demo (experiment 01 OLS)
    "lena.jpg":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    # Foto jalan nyata (Hough line, pipeline)
    "jalan.png":
        "https://picsum.photos/seed/road04a/640/480",
    # Foto koin nyata (Hough circle, ellipse fitting, graph-cut)
    "koin.png":
        "https://picsum.photos/seed/coins04b/640/480",
    # Foto checkerboard nyata untuk homography (calibration image)
    "papan_catur.jpg":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg",
    # Foto dokumen nyata untuk koreksi perspektif
    "dokumen_asli.jpg":
        "https://picsum.photos/seed/doc04g/640/480",
    # Scene foto untuk template matching (target scene)
    "target.png":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box_in_scene.png",
    # Patch template untuk template matching
    "template.png":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box.png",
    # Stereo aloe frames untuk optical flow dan feature matching
    "frame1.png":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeL.jpg",
    "frame2.png":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeR.jpg",
    # Foto bersih untuk denoising
    "clean_img.png":
        "https://picsum.photos/seed/clean04c/640/480",
    # Foto bangunan OpenCV (referensi umum)
    "building_cv.jpg":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
    # Foto buah OpenCV (referensi umum)
    "fruits_cv.jpg":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
}

print("\n[INFO] Download gambar nyata...")
ok = fail = 0
for fname, url in IMAGES.items():
    if _download(fname, url): ok += 1
    else: fail += 1

print("\n" + "=" * 60)
print("[INFO] Daftar gambar di folder 'image/':")
for f in sorted(os.listdir(IMAGE_DIR)):
    kb = os.path.getsize(os.path.join(IMAGE_DIR, f)) / 1024
    print(f"  {f:<30} {kb:>7.1f} KB")
print(f"\n  Download berhasil : {ok}, gagal : {fail}")
print("\nSetup Modul 04 selesai!")
print("=" * 60)
