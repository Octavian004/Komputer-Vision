"""
generate_test_images.py
-----------------------
Script untuk membuat gambar-gambar test dari satu gambar panorama sintetis.
Menghasilkan 8 gambar dengan overlap ~30% untuk simulasi pengambilan foto ruangan.

Cara pakai:
    python generate_test_images.py
    python generate_test_images.py --rooms 2
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def create_synthetic_room(width=3000, height=800, room_id=1):
    """Buat gambar ruangan sintetis dengan tekstur realistis."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Warna dinding (berbeda per ruangan)
    room_colors = [
        ([180, 200, 220], [160, 180, 200]),  # Biru muda
        ([200, 220, 180], [180, 200, 160]),  # Hijau muda
        ([220, 200, 180], [200, 180, 160]),  # Krem
    ]
    color_main, color_shadow = room_colors[(room_id - 1) % len(room_colors)]

    # Background dinding dengan gradasi
    for y in range(height):
        ratio = y / height
        color = [
            int(color_main[c] * (1 - ratio * 0.3) + color_shadow[c] * ratio * 0.3)
            for c in range(3)
        ]
        img[y, :] = color

    # Gambar lantai (bagian bawah 25%)
    floor_start = int(height * 0.75)
    for y in range(floor_start, height):
        ratio = (y - floor_start) / (height - floor_start)
        floor_color = [
            int(120 + ratio * 40),
            int(90 + ratio * 30),
            int(60 + ratio * 20)
        ]
        img[y, :] = floor_color

    # Gambar langit-langit (top 10%)
    for y in range(int(height * 0.1)):
        img[y, :] = [240, 240, 235]

    # Tambah tekstur bata di dinding
    np.random.seed(42 + room_id)
    _add_wall_texture(img, width, height)

    # Tambah furnitur & objek
    _add_furniture(img, width, height, room_id)

    # Tambah efek pencahayaan
    _add_lighting(img, width, height)

    return img


def _add_wall_texture(img, width, height):
    """Tambah tekstur bata ke dinding."""
    wall_end = int(height * 0.75)
    brick_h, brick_w = 30, 80
    for row in range(0, wall_end, brick_h + 3):
        offset = (brick_w // 2) if (row // (brick_h + 3)) % 2 else 0
        for col in range(-offset, width, brick_w + 4):
            if col + brick_w > 0 and col < width:
                x1, x2 = max(0, col), min(width, col + brick_w)
                y1, y2 = max(0, row), min(wall_end, row + brick_h)
                # Sedikit variasi warna bata
                noise = np.random.randint(-15, 15)
                brick_region = img[y1:y2, x1:x2].astype(np.int32)
                brick_region += noise
                brick_region = np.clip(brick_region, 0, 255)
                img[y1:y2, x1:x2] = brick_region.astype(np.uint8)
                # Garis mortar
                if y2 < wall_end:
                    img[y2:min(y2+3, wall_end), x1:x2] = [200, 200, 195]
                if x2 < width:
                    img[y1:y2, x2:min(x2+4, width)] = [200, 200, 195]


def _add_furniture(img, width, height, room_id):
    """Tambah objek furnitur sederhana."""
    floor_y = int(height * 0.75)
    
    # Jendela
    window_positions = [int(width * 0.15), int(width * 0.45), int(width * 0.75)]
    for wx in window_positions:
        if wx + 150 < width:
            wy = int(height * 0.15)
            ww, wh = 150, 200
            # Frame jendela
            cv2.rectangle(img, (wx, wy), (wx + ww, wy + wh), (100, 80, 60), 8)
            # Kaca (cahaya terang)
            cv2.rectangle(img, (wx + 8, wy + 8), (wx + ww - 8, wy + wh - 8),
                          (200, 230, 255), -1)
            # Palang tengah
            cv2.line(img, (wx + ww // 2, wy), (wx + ww // 2, wy + wh), (100, 80, 60), 4)
            cv2.line(img, (wx, wy + wh // 2), (wx + ww, wy + wh // 2), (100, 80, 60), 4)

    # Meja
    table_positions = [int(width * 0.08), int(width * 0.35), int(width * 0.62), int(width * 0.85)]
    for tx in table_positions:
        if tx + 200 < width:
            ty = floor_y - 80
            # Permukaan meja
            cv2.rectangle(img, (tx, ty), (tx + 200, ty + 15),
                          (100, 70, 40), -1)
            # Kaki meja
            cv2.rectangle(img, (tx + 10, ty + 15), (tx + 30, floor_y),
                          (80, 55, 30), -1)
            cv2.rectangle(img, (tx + 170, ty + 15), (tx + 190, floor_y),
                          (80, 55, 30), -1)

    # Kursi
    chair_x = int(width * 0.25)
    for cx in [chair_x, int(width * 0.55), int(width * 0.80)]:
        if cx + 80 < width:
            # Dudukan
            cv2.rectangle(img, (cx, floor_y - 50), (cx + 80, floor_y - 40),
                          (60, 40, 20), -1)
            # Sandaran
            cv2.rectangle(img, (cx + 5, floor_y - 120), (cx + 75, floor_y - 50),
                          (70, 50, 30), -1)
            # Kaki
            for kx in [cx + 5, cx + 65]:
                cv2.rectangle(img, (kx, floor_y - 40), (kx + 10, floor_y),
                              (50, 35, 15), -1)

    # Label nama ruangan
    label = f"ROOM {room_id}"
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, label, (width // 2 - 100, height // 2),
                font, 2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (width // 2 - 100, height // 2),
                font, 2, (255, 255, 255), 2, cv2.LINE_AA)


def _add_lighting(img, width, height):
    """Tambah efek pencahayaan dari lampu."""
    light_positions = [width // 4, width // 2, 3 * width // 4]
    for lx in light_positions:
        # Buat overlay cahaya (radial gradient)
        overlay = np.zeros_like(img, dtype=np.float32)
        ly = int(height * 0.05)
        for y in range(height):
            for x in range(max(0, lx - 300), min(width, lx + 300)):
                dist = np.sqrt((x - lx) ** 2 + (y - ly) ** 2)
                intensity = max(0, 1 - dist / 400) * 0.2
                overlay[y, x] = [intensity * 255] * 3
        img[:] = np.clip(img.astype(np.float32) + overlay, 0, 255).astype(np.uint8)


def slice_panorama_to_images(panorama, num_images=8, overlap=0.35, output_dir=""):
    """
    Potong gambar panorama menjadi num_images gambar dengan overlap.
    Setiap gambar merepresentasikan satu 'foto' dari kamera.
    """
    h, w = panorama.shape[:2]
    
    # Hitung ukuran window yang menutupi gambar
    # w = slice_w + (num_images - 1) * step
    # step = slice_w * (1 - overlap)
    # w = slice_w * (1 + (num_images - 1) * (1 - overlap))
    slice_w = int(w / (1 + (num_images - 1) * (1 - overlap)))
    step = int(slice_w * (1 - overlap))
    
    images = []
    filenames = []
    
    print(f"  Membuat {num_images} gambar dari panorama {w}x{h}px...")
    print(f"  Lebar tiap gambar: {slice_w}px | Step: {step}px | Overlap: {overlap*100:.0f}%")
    
    for i in range(num_images):
        x_start = int(i * step)
        x_end = min(x_start + slice_w, w)
        
        # Jika gambar terakhir, ambil yang tersisa
        if i == num_images - 1:
            x_start = max(0, w - slice_w)
            x_end = w
        
        slice_img = panorama[:, x_start:x_end].copy()
        
        # Tambah sedikit noise per gambar (simulasi kondisi kamera berbeda)
        noise = np.random.normal(0, 3, slice_img.shape).astype(np.int16)
        slice_img = np.clip(slice_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Sedikit variasi brightness
        brightness_factor = np.random.uniform(0.92, 1.08)
        slice_img = np.clip(slice_img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        
        # Simpan
        filename = os.path.join(output_dir, f"img_{i+1:02d}.jpg")
        cv2.imwrite(filename, slice_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        images.append(slice_img)
        filenames.append(filename)
        print(f"  ✓ Disimpan: {filename} ({slice_img.shape[1]}x{slice_img.shape[0]}px)")
    
    return images, filenames


def generate_all_rooms(num_rooms=2, num_images_per_room=8, output_base="images"):
    """Generate gambar test untuk semua ruangan."""
    print("=" * 60)
    print("  GENERATOR GAMBAR TEST - PANORAMA STITCHING")
    print("=" * 60)
    
    # np.random.seed(123) # Dinonaktifkan agar gambar bervariasi setiap di-generate
    
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    for room_id in range(1, num_rooms + 1):
        room_dir = os.path.join(output_base, f"room{room_id}")
        Path(room_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n[ROOM {room_id}] Membuat gambar sintetis...")
        
        # Buat panorama lengkap
        panorama = create_synthetic_room(width=3200, height=800, room_id=room_id)
        
        # Simpan panorama referensi
        ref_path = os.path.join(output_base, f"room{room_id}_reference.jpg")
        cv2.imwrite(ref_path, panorama)
        print(f"  ✓ Panorama referensi: {ref_path}")
        
        # Potong jadi beberapa gambar
        images, filenames = slice_panorama_to_images(
            panorama,
            num_images=num_images_per_room,
            overlap=0.35,
            output_dir=room_dir
        )
        
        print(f"  ✓ {len(filenames)} gambar tersimpan di '{room_dir}/'")
    
    print("\n" + "=" * 60)
    print("  SELESAI! Struktur folder:")
    print("=" * 60)
    for room_id in range(1, num_rooms + 1):
        print(f"  images/room{room_id}/ → {num_images_per_room} gambar test")
    print("\n  Jalankan:")
    print(f"  python panorama_stitcher.py --input images/room1 --output output/panorama_room1.jpg")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generator gambar test untuk Panoramic Image Stitching"
    )
    parser.add_argument("--rooms", type=int, default=2,
                        help="Jumlah ruangan yang di-generate (default: 2)")
    parser.add_argument("--images", type=int, default=8,
                        help="Jumlah gambar per ruangan (default: 8)")
    parser.add_argument("--output", type=str, default="images",
                        help="Folder output gambar (default: images/)")
    args = parser.parse_args()
    
    generate_all_rooms(
        num_rooms=args.rooms,
        num_images_per_room=args.images,
        output_base=args.output
    )
