"""
DOWNLOAD IMAGE - MODUL 02: PEMBENTUKAN CITRA (IMAGE FORMATION)
Semua gambar didownload dari internet. Tidak ada generasi sintetis.

Pemetaan:
  gedung.jpg        -> 01,02,03,04,05,08,09,10,13,14,15,16,17,18,19
  dokumen.jpg       -> 05 (transformasi perspektif)
  kotak_warna.png   -> 12,13 (koordinat polar, shearing)
  baboon.jpg        -> 02,11,12,14,16 (rotasi, interpolasi, polar, refleksi)
  building_cv.jpg   -> 18,19 (remapping, log/power transform)
"""
import os, ssl, urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR   = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODUL 02 - DOWNLOAD GAMBAR NYATA")
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
    "gedung.jpg":       "https://picsum.photos/seed/arch02a/640/480",
    "dokumen.jpg":      "https://picsum.photos/seed/book02e/640/480",
    "kotak_warna.png":  "https://picsum.photos/seed/color02f/400/400",
    "baboon.jpg":       "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
    "building_cv.jpg":  "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
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
print("\nSetup Modul 02 selesai!")
print("=" * 60)
