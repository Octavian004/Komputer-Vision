"""
panorama_stitcher.py
--------------------
Pipeline utama untuk stitching panorama.

Menjalankan proses:
1. Load gambar
2. Proyeksi Silindris
3. Feature detection & matching
4. Global Alignment & Homografi
5. Multi-band Blending
6. Cropping & Enhancement
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import time

from utils.image_utils import load_images_from_folder, crop_black_borders, enhance_panorama
from utils.warping import warp_images, compute_canvas_size, warp_perspective_with_translation
from utils.feature_matching import create_detector, create_matcher, match_image_pair
from utils.blending import blend_two_images_multiband


def stitch_panorama(input_folder, output_path, projection="cylindrical", use_blending=True):
    start_time = time.time()
    
    print("=" * 60)
    print("  === SISTEM PANORAMA STITCHING ===")
    print("=" * 60)
    
    # 1. Load Images
    images, filenames = load_images_from_folder(input_folder)
    if len(images) < 2:
        print("[Error] Butuh minimal 2 gambar untuk stitching panorama.")
        return False
        
    # 2. Base Warping (Proyeksi)
    print(f"\n[Tahap 1] Melakukan proyeksi {projection}...")
    warped_images, focal_len = warp_images(images, projection=projection)
    
    # 3 & 4. Feature Matching & Homography Estimation
    print("\n[Tahap 2] Deteksi fitur, pencocokan, dan estimasi pergerakan...")
    detector, algo_name = create_detector("SIFT")
    matcher = create_matcher(algo_name)
    
    homographies = [None] * len(warped_images)
    homographies[0] = np.eye(3, dtype=np.float64) # Gambar pertama sebagai referensi dasar
    
    # Hitung matriks transformasi akumulatif dari kiri ke kanan (atau relatif satu sama lain)
    # Pendekatan berurutan: H_i = H_{i-1} * H_{i ke i-1}
    for i in range(1, len(warped_images)):
        print(f"\n  Mencocokkan Gambar {i} dengan Gambar {i+1}...")
        
        img_prev = warped_images[i-1]
        img_curr = warped_images[i]
        
        # Cari H yang memetakan curr ke prev (Karena kita maju ke kanan)
        H_pair, kp_prev, kp_curr, matches, _ = match_image_pair(
            img_prev, img_curr, detector, matcher, algo_name, ratio=0.75, min_match=15
        )
        
        if H_pair is None:
            print(f"[Error] Gagal mendapatkan transformasi stabil antaram Gambar {i} dan {i+1}.")
            return False
            
        # Akumulatif
        homographies[i] = homographies[i-1] @ H_pair
        
    print("\n[Tahap 3] Menyiapkan Canvas untuk Global Alignment...")
    # Hitung ukuran total canvas agar semua termuat tanpa terpotong
    canvas_size, offset = compute_canvas_size(warped_images, homographies)
    print(f"  Ukuran Canvas Global: {canvas_size[0]} x {canvas_size[1]}")
    
    # 5. Image Warping dan Caching Mask
    print("\n[Tahap 4] Melakukan Warping semua gambar ke Global Canvas...")
    warped_to_canvas = []
    masks_mapped = []
    
    for i in range(len(warped_images)):
        # Warna putih di mana gambar ada
        img_mask = np.ones((warped_images[i].shape[0], warped_images[i].shape[1]), dtype=np.uint8) * 255
        
        # Warp ke canvas besar
        warped_img = warp_perspective_with_translation(warped_images[i], homographies[i], canvas_size, offset)
        warped_mask = warp_perspective_with_translation(img_mask, homographies[i], canvas_size, offset)
        
        warped_to_canvas.append(warped_img)
        masks_mapped.append(warped_mask)
        print(f"  ✓ Gambar {i+1} ter-warp")
        
    # 6. Blending
    print("\n[Tahap 5] Menyusun Panorama dan Blending...")
    # Mulai dengan gambar pertama
    panorama = warped_to_canvas[0].copy()
    current_mask = masks_mapped[0].copy()
    
    for i in range(1, len(warped_to_canvas)):
        print(f"  Blending bagian {i+1}/{len(warped_to_canvas)}...")
        new_img = warped_to_canvas[i]
        new_mask = masks_mapped[i]
        
        if use_blending:
            # Menggunakan Multi-band blending
            panorama = blend_two_images_multiband(panorama, new_img, current_mask, new_mask, levels=5)
        else:
            # Overwrite sederhana
            panorama[new_mask > 0] = new_img[new_mask > 0]
            
        # Update current mask dengan region gabungan
        current_mask = cv2.bitwise_or(current_mask, new_mask)
        
    # 7. Post-Processing (Crop & Enhance)
    print("\n[Tahap 6] Post-Processing...")
    
    print("  Melakukan auto-cropping...")
    cropped_panorama = crop_black_borders(panorama)
    
    print("  Melakukan image enhancement...")
    final_panorama = enhance_panorama(cropped_panorama)
    
    # 8. Simpan
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, final_panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    lap_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  🎉 PANORAMA SELESAI DALAM {lap_time:.2f} DETIK 🎉")
    print(f"  Tersimpan di: {output_path}")
    print(f"  Dimensi: {final_panorama.shape[1]}x{final_panorama.shape[0]}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Stitching Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Folder berisi gambar input")
    parser.add_argument("--output", "-o", type=str, default="output/panorama_result.jpg", help="Path file output (default: output/panorama_result.jpg)")
    parser.add_argument("--projection", "-p", type=str, choices=["cylindrical", "spherical"], default="cylindrical", help="Tipe proyeksi awal")
    parser.add_argument("--no-blend", action="store_true", help="Nonaktifkan multi-band blending (lebih cepat tapi bergaris)")
    args = parser.parse_args()
    
    stitch_panorama(args.input, args.output, args.projection, use_blending=not args.no_blend)
