"""
==========================================================================
 PERCOBAAN 20 — DETEKSI DAN ESTIMASI POSE MARKER ArUco
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Memahami cara kerja fiducial marker (ArUco) untuk lokalisasi
           kamera dan overlay 3D (Augmented Reality sederhana).
 Konsep  :
   - ArUco MARKER: pola kotak hitam-putih berisi ID biner (mis. 4×4, 5×5).
     Setiap marker memiliki ID unik yang mudah dideteksi dan didekode.
   - DETEKSI: cv2.aruco.ArucoDetector (OpenCV 4.7+) menemukan sudut
     (corners) 4 titik tiap marker dalam piksel.
   - ESTIMASI POSE: solvePnP / estimatePoseSingleMarkers mencari Rvec & Tvec
     (rotasi & translasi) yang memetakan 3D marker ke 2D kamera.
   - PROYEKSI SUMBU: cv2.projectPoints memproyeksikan titik 3D sumbu XYZ
     ke gambar → overlay 3D axes (merah/hijau/biru) di setiap marker.
   - KALIBRASI DIBUTUHKAN: akurasi estimasi pose bergantung pada tersedianya
     matrix kamera K dan koefisien distorsi D.
 Catatan : Pada OpenCV 4.7+ gunakan cv2.aruco.ArucoDetector.
           Pada OpenCV < 4.7 gunakan cv2.aruco.detectMarkers.
           Program mendeteksi versi otomatis.
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cek versi OpenCV & modul aruco ───────────────────────────────────────────
ARUCO_TERSEDIA = hasattr(cv2, "aruco")
OCV_MAJOR = int(cv2.__version__.split(".")[0])
OCV_MINOR = int(cv2.__version__.split(".")[1])


def buat_gambar_marker_aruco(aruco_dict, marker_id: int,
                              ukuran: int = 200) -> np.ndarray:
    """Membuat gambar marker ArUco sebagai numpy array."""
    if not ARUCO_TERSEDIA:
        return None
    marker_img = np.zeros((ukuran, ukuran), dtype=np.uint8)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id,
                                               ukuran, marker_img, 1)
    return marker_img


def buat_papan_aruco(aruco_dict, ids=(0, 1, 2, 3),
                     ukuran=150, margin=20) -> np.ndarray:
    """Buat papan 2×2 berisi 4 marker dengan ID berbeda."""
    n = len(ids)
    cols = 2
    rows = (n + 1) // 2
    total_w = cols * ukuran + (cols + 1) * margin
    total_h = rows * ukuran + (rows + 1) * margin
    papan = np.ones((total_h, total_w), dtype=np.uint8) * 255

    for idx, mid in enumerate(ids):
        row, col = divmod(idx, cols)
        y0 = margin + row * (ukuran + margin)
        x0 = margin + col * (ukuran + margin)
        m  = buat_gambar_marker_aruco(aruco_dict, mid, ukuran)
        if m is not None:
            papan[y0:y0+ukuran, x0:x0+ukuran] = m

    return papan


def deteksi_marker(gray: np.ndarray, aruco_dict, params=None):
    """
    Deteksi marker ArUco — kompatibel OpenCV <4.7 dan >=4.7.
    Mengembalikan (corners, ids, rejected).
    """
    if not ARUCO_TERSEDIA:
        return [], None, []

    if params is None:
        params = cv2.aruco.DetectorParameters()

    # OpenCV 4.7+ API baru
    if OCV_MAJOR > 4 or (OCV_MAJOR == 4 and OCV_MINOR >= 7):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        # API lama
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params)

    return corners, ids, rejected


def estimasi_pose_marker(corners, ukuran_marker_m: float,
                          K: np.ndarray, D: np.ndarray):
    """
    Estimasi pose tiap marker (Rvec, Tvec) menggunakan solvePnP.
    ukuran_marker_m: panjang sisi marker dalam meter.
    """
    half = ukuran_marker_m / 2.0
    obj_pts = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)

    rvecs, tvecs = [], []
    for corner in corners:
        img_pts = corner[0].astype(np.float32)
        _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs


def gambar_sumbu_3d(img: np.ndarray, rvec, tvec,
                    K: np.ndarray, D: np.ndarray,
                    panjang: float = 0.05) -> np.ndarray:
    """Overlay sumbu XYZ di pusat marker."""
    sumbu = np.float32([
        [panjang, 0, 0],
        [0, panjang, 0],
        [0, 0, -panjang],
        [0, 0, 0],
    ])
    pts, _ = cv2.projectPoints(sumbu, rvec, tvec, K, D)
    pts = pts.reshape(-1, 2).astype(int)
    orig = tuple(pts[3])

    cv2.arrowedLine(img, orig, tuple(pts[0]), (0,   0, 255), 2, tipLength=0.2)  # X merah
    cv2.arrowedLine(img, orig, tuple(pts[1]), (0, 255,   0), 2, tipLength=0.2)  # Y hijau
    cv2.arrowedLine(img, orig, tuple(pts[2]), (255, 0,   0), 2, tipLength=0.2)  # Z biru
    return img


# ── Demo utama ────────────────────────────────────────────────────────────────
def demo_buat_dan_simpan_marker():
    """Membuat & menyimpan marker ArUco individual."""
    if not ARUCO_TERSEDIA:
        print("  [SKIP] Modul cv2.aruco tidak tersedia.")
        return

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for i, ax in enumerate(axes):
        m = buat_gambar_marker_aruco(d, i, 200)
        fname = os.path.join(OUTPUT_DIR, f"20_marker_id{i}.png")
        cv2.imwrite(fname, m)
        ax.imshow(m, cmap="gray")
        ax.set_title(f"Marker ID={i}")
        ax.axis("off")

    plt.suptitle("Percobaan 20 — Marker ArUco DICT_4X4_50", fontweight="bold")
    out = os.path.join(OUTPUT_DIR, "20_marker_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    print(f"[SIMPAN] {out}")


def demo_deteksi_per_id():
    """Buat papan 4 marker, deteksi, dan visualisasi corners."""
    if not ARUCO_TERSEDIA:
        print("  [SKIP] Modul cv2.aruco tidak tersedia.")
        return

    d    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    papan = buat_papan_aruco(d, ids=[0, 1, 2, 3])

    # Tambahkan noise ringan agar lebih realistis
    noise = np.random.randint(-10, 10, papan.shape, dtype=np.int16)
    papan_noisy = np.clip(papan.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    corners, ids, rejected = deteksi_marker(papan_noisy, d)

    overlay = cv2.cvtColor(papan_noisy, cv2.COLOR_GRAY2BGR)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
        print(f"  Terdeteksi {len(ids)} marker: ID = {ids.flatten().tolist()}")
    else:
        print("  Tidak ada marker terdeteksi.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(papan, cmap="gray"); axes[0].set_title("Papan ArUco Asli"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Hasil Deteksi ({len(ids) if ids is not None else 0} marker)")
    axes[1].axis("off")

    plt.suptitle("Percobaan 20 — Deteksi Marker ArUco", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "20_deteksi_aruco.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


def demo_estimasi_pose():
    """Buat marker sintetis, estimasi pose, overlay sumbu 3D."""
    if not ARUCO_TERSEDIA:
        print("  [SKIP] Modul cv2.aruco tidak tersedia.")
        return

    d     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    ukuran = 250
    marker_gray = buat_gambar_marker_aruco(d, 0, ukuran)
    # Tambahkan border putih agar deteksi lebih mudah
    padded = cv2.copyMakeBorder(marker_gray, 40, 40, 40, 40,
                                cv2.BORDER_CONSTANT, value=255)
    corners, ids, _ = deteksi_marker(padded, d)

    if ids is None or len(ids) == 0:
        print("  Marker tidak terdeteksi di mode estimasi pose.")
        return

    # Kamera sintetis: focal ≈ lebar gambar, pusat di tengah
    h, w = padded.shape
    fx = fy = float(w)
    K = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    rvecs, tvecs = estimasi_pose_marker(corners, ukuran_marker_m=0.05, K=K, D=D)

    overlay = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
    for rvc, tvc in zip(rvecs, tvecs):
        gambar_sumbu_3d(overlay, rvc, tvc, K, D, panjang=0.03)
        # Tampilkan tvec info
        print(f"  ID={ids.flatten()[0]} | Tvec: {tvc.flatten().round(4)} m")

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Percobaan 20 — Estimasi Pose + Overlay Sumbu 3D\n"
              "(Merah=X, Hijau=Y, Biru=Z)")
    plt.axis("off")
    out = os.path.join(OUTPUT_DIR, "20_pose_estimation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


def demo_tanpa_aruco():
    """Fallback visualisasi saat modul aruco tidak tersedia."""
    print("  Membuat visualisasi ID biner manual sebagai pengganti ArUco.")
    bits = np.array([
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
    ], dtype=np.uint8)
    cell = 40
    img = np.kron(bits, np.ones((cell, cell), dtype=np.uint8)) * 255
    # Border hitam
    img = cv2.copyMakeBorder(img, cell, cell, cell, cell,
                             cv2.BORDER_CONSTANT, value=0)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray"); plt.title("Contoh Pola Biner (ArUco tidak tersedia)")
    plt.axis("off")
    out = os.path.join(OUTPUT_DIR, "20_pola_biner_manual.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[SIMPAN] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERCOBAAN 20: DETEKSI DAN ESTIMASI POSE MARKER ArUco")
    print("=" * 60)
    print(f"  OpenCV versi : {cv2.__version__}")
    print(f"  Modul aruco  : {'TERSEDIA' if ARUCO_TERSEDIA else 'TIDAK TERSEDIA'}")

    if ARUCO_TERSEDIA:
        print("\n[1] Membuat Marker ArUco")
        demo_buat_dan_simpan_marker()

        print("\n[2] Deteksi Marker pada Gambar")
        demo_deteksi_per_id()

        print("\n[3] Estimasi Pose dan Overlay Sumbu 3D")
        demo_estimasi_pose()
    else:
        print("\n  [INFO] Install opencv-contrib-python untuk fitur ArUco:")
        print("         pip install opencv-contrib-python")
        demo_tanpa_aruco()

    print("\n[SELESAI] Output tersimpan di folder:", OUTPUT_DIR)
