"""Document Scanner Project

Jalankan:
    python scanner.py --input input --output output

Opsional:
    --calib configs/calib_camera.json
    --dpi 300
    --no-gamma
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from utils import (
    auto_gamma,
    estimate_gamma,
    find_document_contour,
    load_calibration,
    make_thumbnail,
    undistort_image,
    warp_to_a4,
)


def process_image(
    src_path: Path,
    dst_path: Path,
    thumb_path: Path,
    calib: bool = False,
    camera_data=None,
    dpi: int = 300,
    apply_gamma: bool = True,
):
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Gagal membuka {src_path}")

    if calib and camera_data is not None:
        img = undistort_image(img, *camera_data)

    pts = find_document_contour(img)
    if pts is None:
        # Jika kontur dokumen tidak terdeteksi, gunakan seluruh image sebagai fallback.
        # Ini mencegah proses berhenti dan tetap menghasilkan output.
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        print("[WARN] Kontur dokumen tidak terdeteksi; menggunakan seluruh gambar sebagai fallback.")

    warped = warp_to_a4(img, pts, dpi=dpi)

    if apply_gamma:
        gamma_val = estimate_gamma(warped)
        warped = auto_gamma(warped, gamma=gamma_val)
    else:
        gamma_val = 1.0

    cv2.imwrite(str(dst_path), warped)

    thumb = make_thumbnail(warped, max_dim=512)
    cv2.imwrite(str(thumb_path), thumb)

    return {
        "source": str(src_path.name),
        "output": str(dst_path.name),
        "thumb": str(thumb_path.name),
        "gamma": round(gamma_val, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Document Scanner (Modul 2)")

    # Defaults untuk kemudahan menjalankan tanpa argumen
    default_root = Path(__file__).resolve().parent
    default_input = default_root / "input"
    default_output = default_root / "output"

    parser.add_argument(
        "--input",
        default=str(default_input),
        help=f"Folder input (foto dokumen). Default: {default_input}",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help=f"Folder output hasil scan. Default: {default_output}",
    )
    parser.add_argument("--calib", help="File kalibrasi camera (JSON)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI output A4")
    parser.add_argument("--no-gamma", action="store_true", help="Matikan auto-gamma")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Pastikan folder input + output ada (agar bisa langsung dijalankan tanpa setup manual)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "thumbs").mkdir(exist_ok=True)

    camera_data = None
    if args.calib:
        camera_data = load_calibration(Path(args.calib))
        if camera_data is None:
            raise RuntimeError("File kalibrasi tidak valid atau tidak ditemukan")

    report_path = output_dir / "report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["source", "output", "thumb", "gamma"])
        writer.writeheader()

        for i, img_path in enumerate(sorted(input_dir.glob("*.*"))):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            out_name = f"scan_{i:04d}.png"
            thumb_name = f"scan_{i:04d}_thumb.png"
            try:
                row = process_image(
                    img_path,
                    output_dir / out_name,
                    output_dir / "thumbs" / thumb_name,
                    calib=bool(camera_data),
                    camera_data=camera_data,
                    dpi=args.dpi,
                    apply_gamma=not args.no_gamma,
                )
                writer.writerow(row)
                print(f"[OK] {img_path.name} → {out_name}")
            except Exception as exc:
                print(f"[ERROR] {img_path.name}: {exc}")

    print(f"\nSelesai. Laporan: {report_path}")


if __name__ == "__main__":
    main()
