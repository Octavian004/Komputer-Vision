"""
DOWNLOAD IMAGE - MODUL 03: PEMROSESAN CITRA (IMAGE PROCESSING)
Semua gambar didownload dari internet. Tidak ada generasi sintetis.

Pemetaan:
  kota.jpg          -> 01,02,03,08,09,10,16,18,19,20 (brightness, blur, filter, Fourier)
  nature.jpg        -> 04,06,20 (gamma correction, thresholding, blending)
  dokumen.jpg       -> 05,06,07,17 (thresholding, adaptive, tophat)
  buah.jpg          -> 06 (Otsu thresholding)
  teks_buram.jpg    -> 11 (sharpening)
  garis_tepi.jpg    -> 12,13,14 (edge detection: Sobel, Canny, Laplacian)
  lena.jpg          -> 15,16 (morfologi — konversi ke binary, then morph ops)
  baboon.jpg        -> 17 (tophat/blackhat)
"""
import os, ssl, urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR   = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODUL 03 - DOWNLOAD GAMBAR NYATA")
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
    "kota.jpg":         "https://picsum.photos/seed/city03a/640/480",
    "nature.jpg":       "https://picsum.photos/seed/forest03c/640/480",
    "dokumen.jpg":      "https://picsum.photos/seed/document03b/640/480",
    "buah.jpg":         "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
    "teks_buram.jpg":   "https://picsum.photos/seed/text03d/640/480",
    "garis_tepi.jpg":   "https://picsum.photos/seed/lines03e/640/480",
    "lena.jpg":         "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "baboon.jpg":       "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
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
print("\nSetup Modul 03 selesai!")
print("=" * 60)
