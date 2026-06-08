"""
=============================================================================
Modul 04 - Deteksi Fitur dan Pencocokan
File    : 13_deskriptor_perbandingan.py
Topik   : Perbandingan Deskriptor
Deskripsi:
    Membandingkan empat deskriptor populer — SIFT (128-float), ORB (256-bit
    uint8), AKAZE (61-uint8), BRISK (64-uint8) — dari sisi dimensi, kecepatan
    komputasi, kualitas matching, dan performa keseluruhan via radar chart.
=============================================================================
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# ── Path konfigurasi ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def muat_gambar_pasangan():
    """
    Memuat gambar dari IMAGE_DIR atau membuat gambar sintetis.
    Mengembalikan (gray1, gray2, img1_bgr, img2_bgr).
    gray2 adalah gray1 yang telah dirotasi 20° dan di-scale 0.9.
    """
    for nama in ["chess.png", "box.png", "simple.png", "scene.png"]:
        jalur = os.path.join(IMAGE_DIR, nama)
        if os.path.exists(jalur):
            img = cv2.imread(jalur)
            if img is not None:
                h, w  = img.shape[:2]
                M     = cv2.getRotationMatrix2D((w//2, h//2), 20, 0.9)
                img2  = cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT)
                return (cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                        img, img2)

    H, W = 480, 640
    img1 = np.full((H, W, 3), (40, 20, 60), dtype=np.uint8)
    cv2.rectangle(img1, (60,  50),  (260, 220), (180, 100, 60), -1)
    cv2.circle   (img1, (430, 150), 100,        (60, 200, 180), -1)
    pts = np.array([[200, 330], [500, 390], [120, 420]], np.int32)
    cv2.fillPoly (img1, [pts],               (220, 80,  80))
    cv2.rectangle(img1, (360, 280),  (600, 440),(60, 120, 220), -1)
    cv2.circle   (img1, (150, 380),   55,       (220, 200,  40), -1)
    noise = np.random.randint(0, 25, img1.shape, dtype=np.uint8)
    img1  = cv2.add(img1, noise)

    M    = cv2.getRotationMatrix2D((W//2, H//2), 20, 0.9)
    img2 = cv2.warpAffine(img1, M, (W, H), borderMode=cv2.BORDER_REFLECT)

    return (cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
            img1, img2)


# ─────────────────────────────────────────────────────────────────────────────
def demo_dimensi_tipe_deskriptor(gray1, img1_bgr):
    """
    Demo 1: Dimensi dan tipe deskriptor — SIFT(128-float), ORB(32-uint8/256-bit),
    AKAZE(61-uint8), BRISK(64-uint8). Visualisasi tabel dan contoh deskriptor.
    """
    print("\n[Demo 1] Dimensi dan tipe deskriptor berbagai algoritma")

    # Konfigurasi detektor
    detektors = {
        "SIFT" : cv2.SIFT_create(),
        "ORB"  : cv2.ORB_create(nfeatures=500),
        "AKAZE": cv2.AKAZE_create(),
        "BRISK": cv2.BRISK_create(),
    }

    info_deskriptor = {}
    for nama, det in detektors.items():
        try:
            kp, des = det.detectAndCompute(gray1, None)
            if des is not None and len(des) > 0:
                info_deskriptor[nama] = {
                    "n_kp"   : len(kp),
                    "dimensi": des.shape[1],
                    "dtype"  : str(des.dtype),
                    "contoh" : des[0]
                }
                print(f"  {nama:5s}: {len(kp):4d} keypoint | "
                      f"dim={des.shape[1]:3d} | dtype={des.dtype}")
            else:
                print(f"  {nama:5s}: deskriptor kosong")
        except Exception as e:
            print(f"  {nama:5s}: ERROR — {e}")

    # Tabel ringkasan
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [2, 3]})

    # Kiri: tabel properti
    kolom = ["Algoritma", "N Keypoint", "Dimensi", "Tipe Data", "Keterangan"]
    keterangan = {
        "SIFT" : "128 float32 | L2 norm",
        "ORB"  : "32 uint8 (256-bit) | Hamming",
        "AKAZE": "61 uint8 | Hamming",
        "BRISK": "64 uint8 | Hamming",
    }
    baris = []
    for nama in ["SIFT", "ORB", "AKAZE", "BRISK"]:
        if nama in info_deskriptor:
            d = info_deskriptor[nama]
            baris.append([nama, str(d["n_kp"]), str(d["dimensi"]),
                          d["dtype"], keterangan.get(nama, "-")])
        else:
            baris.append([nama, "N/A", "N/A", "N/A", "Tidak tersedia"])

    tabel = axes[0].table(cellText=baris, colLabels=kolom,
                          cellLoc="center", loc="center",
                          bbox=[0, 0, 1, 1])
    tabel.auto_set_font_size(False)
    tabel.set_fontsize(9)
    # Warna header
    for j in range(len(kolom)):
        tabel[0, j].set_facecolor("#2c3e50")
        tabel[0, j].set_text_props(color="white", fontweight="bold")
    # Warna baris bergantian
    warna_baris = ["#ecf0f1", "#d5e8d4", "#dae8fc", "#ffe6cc"]
    for i, w in enumerate(warna_baris):
        for j in range(len(kolom)):
            tabel[i+1, j].set_facecolor(w)
    axes[0].axis("off")
    axes[0].set_title("Perbandingan Properti Deskriptor", fontsize=11,
                       fontweight="bold")

    # Kanan: visualisasi nilai 20 elemen pertama deskriptor (normalisasi 0-1)
    for nama, w in zip(["SIFT", "ORB", "AKAZE", "BRISK"],
                        ["#27ae60", "#e74c3c", "#3498db", "#f39c12"]):
        if nama not in info_deskriptor:
            continue
        d = info_deskriptor[nama]["contoh"].astype(np.float32)
        d = d / (d.max() + 1e-6)   # normalisasi
        n_elem = min(20, len(d))
        axes[1].plot(range(n_elem), d[:n_elem], "o-", color=w,
                     linewidth=1.5, markersize=4, label=nama, alpha=0.85)

    axes[1].set_xlabel("Indeks Elemen Deskriptor", fontsize=11)
    axes[1].set_ylabel("Nilai (ternormalisasi 0–1)", fontsize=11)
    axes[1].set_title("Profil 20 Elemen Pertama Deskriptor (satu keypoint)",
                       fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Dimensi dan Tipe Data Deskriptor Fitur", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_demo1_dimensi_tipe_deskriptor.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_benchmark_kecepatan(gray1):
    """
    Demo 2: Benchmark kecepatan komputasi setiap detector+descriptor kombinasi
    menggunakan time.time(), dirata-rata dari beberapa pengulangan.
    """
    print("\n[Demo 2] Benchmark kecepatan komputasi detektor+deskriptor")

    detektors = {
        "SIFT" : cv2.SIFT_create(),
        "ORB"  : cv2.ORB_create(nfeatures=500),
        "AKAZE": cv2.AKAZE_create(),
        "BRISK": cv2.BRISK_create(),
    }
    ULANG   = 10
    hasil   = {}

    for nama, det in detektors.items():
        waktu_list = []
        for _ in range(ULANG):
            t0 = time.time()
            kp, des = det.detectAndCompute(gray1, None)
            waktu_list.append((time.time() - t0) * 1000)  # ms

        rata  = np.mean(waktu_list)
        sdev  = np.std(waktu_list)
        n_kp  = len(kp) if kp else 0
        hasil[nama] = {"rata": rata, "sdev": sdev, "n_kp": n_kp}
        print(f"  {nama:5s}: {rata:6.2f} ± {sdev:.2f} ms | {n_kp} keypoint")

    # Bar chart kecepatan
    nama_list  = list(hasil.keys())
    rata_list  = [hasil[n]["rata"]  for n in nama_list]
    sdev_list  = [hasil[n]["sdev"]  for n in nama_list]
    warna      = ["#27ae60", "#e74c3c", "#3498db", "#f39c12"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(nama_list, rata_list, yerr=sdev_list, color=warna,
                edgecolor="black", capsize=5, error_kw={"linewidth": 2})
    axes[0].set_ylabel("Waktu (ms)", fontsize=11)
    axes[0].set_title(f"Kecepatan Komputasi ({ULANG} pengulangan)", fontsize=11)
    for i, (v, s) in enumerate(zip(rata_list, sdev_list)):
        axes[0].text(i, v + s + 0.3, f"{v:.1f}ms",
                     ha="center", fontsize=10, fontweight="bold")

    # Scatter: kecepatan vs jumlah keypoint
    for i, nama in enumerate(nama_list):
        axes[1].scatter(hasil[nama]["rata"], hasil[nama]["n_kp"],
                        s=150, color=warna[i], label=nama,
                        zorder=3, edgecolors="black")
        axes[1].annotate(nama,
                         (hasil[nama]["rata"], hasil[nama]["n_kp"]),
                         textcoords="offset points",
                         xytext=(8, 4), fontsize=10)
    axes[1].set_xlabel("Waktu Komputasi (ms)", fontsize=11)
    axes[1].set_ylabel("Jumlah Keypoint", fontsize=11)
    axes[1].set_title("Trade-off: Kecepatan vs Jumlah Keypoint", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle("Benchmark Kecepatan Detektor+Deskriptor",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_demo2_benchmark_kecepatan.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_matching_quality_precision_k(gray1, gray2, img1_bgr, img2_bgr):
    """
    Demo 3: Precision@k — untuk setiap deskriptor, hitung presisi top-k match.
    Inlier ditentukan dengan RANSAC Homography sebagai ground truth.
    """
    print("\n[Demo 3] Precision@k — kualitas matching per deskriptor")

    K_LIST = [5, 10, 20, 30]

    detektor_info = {
        "SIFT" : (cv2.SIFT_create(),
                  cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),
                                        dict(checks=50)),
                  0.75, "float"),
        "ORB"  : (cv2.ORB_create(nfeatures=500),
                  cv2.BFMatcher(cv2.NORM_HAMMING),
                  0.75, "uint8"),
        "AKAZE": (cv2.AKAZE_create(),
                  cv2.BFMatcher(cv2.NORM_HAMMING),
                  0.80, "uint8"),
        "BRISK": (cv2.BRISK_create(),
                  cv2.BFMatcher(cv2.NORM_HAMMING),
                  0.80, "uint8"),
    }
    warna = {"SIFT": "#27ae60", "ORB": "#e74c3c",
              "AKAZE": "#3498db", "BRISK": "#f39c12"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for nama, (det, matcher, rasio, tipe) in detektor_info.items():
        try:
            kp1, des1 = det.detectAndCompute(gray1, None)
            kp2, des2 = det.detectAndCompute(gray2, None)

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                continue
            if tipe == "uint8":
                des1, des2 = np.uint8(des1), np.uint8(des2)

            pasang = matcher.knnMatch(des1, des2, k=2)
            match_andal = [p[0] for p in pasang
                           if len(p) == 2 and p[0].distance < rasio * p[1].distance]
            match_andal = sorted(match_andal, key=lambda m: m.distance)

            if len(match_andal) < 4:
                continue

            # Ground truth inlier via RANSAC
            src = np.float32([kp1[m.queryIdx].pt for m in match_andal]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in match_andal]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is None:
                continue
            mask = mask.ravel()

            # Hitung presisi top-k
            prec_list = []
            for k in K_LIST:
                k_eff  = min(k, len(match_andal))
                n_in   = int(mask[:k_eff].sum())
                prec   = n_in / k_eff * 100
                prec_list.append(prec)

            ax.plot(K_LIST, prec_list, "o-", color=warna[nama], linewidth=2,
                    markersize=7, label=f"{nama} (total={len(match_andal)})")
            print(f"  {nama:5s}: precision@k = {[f'{p:.0f}%' for p in prec_list]}")

        except Exception as e:
            print(f"  {nama:5s}: ERROR — {e}")

    ax.set_xlabel("k (jumlah top match yang dievaluasi)", fontsize=12)
    ax.set_ylabel("Precision@k (%)", fontsize=12)
    ax.set_title("Precision@k: Kualitas Matching per Deskriptor\n"
                 "(inlier ground truth via RANSAC Homography)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_demo3_precision_at_k.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def demo_radar_chart_perbandingan(gray1, gray2):
    """
    Demo 4: Radar chart perbandingan keempat deskriptor pada 4 dimensi:
    speed, nkeypoints, match_ratio, robustness_rotation.
    Nilai dinormalisasi ke [0, 1].
    """
    print("\n[Demo 4] Radar chart perbandingan SIFT / ORB / AKAZE / BRISK")

    # ── Kumpulkan data ─────────────────────────────────────────────────────────
    ULANG   = 5
    detektors = {
        "SIFT" : cv2.SIFT_create(),
        "ORB"  : cv2.ORB_create(nfeatures=500),
        "AKAZE": cv2.AKAZE_create(),
        "BRISK": cv2.BRISK_create(),
    }
    matcher_map = {
        "SIFT" : (cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),
                                         dict(checks=50)),  0.75, "float"),
        "ORB"  : (cv2.BFMatcher(cv2.NORM_HAMMING),  0.75, "uint8"),
        "AKAZE": (cv2.BFMatcher(cv2.NORM_HAMMING),  0.80, "uint8"),
        "BRISK": (cv2.BFMatcher(cv2.NORM_HAMMING),  0.80, "uint8"),
    }

    # Buat gambar rotasi 45° untuk uji robustness
    H, W  = gray1.shape[:2]
    M_rot = cv2.getRotationMatrix2D((W//2, H//2), 45, 1.0)
    gray_rot = cv2.warpAffine(gray1, M_rot, (W, H), borderMode=cv2.BORDER_REFLECT)

    data = {}    # nama → {speed_ms, n_kp, match_ratio, robustness}

    for nama, det in detektors.items():
        matcher, rasio, tipe = matcher_map[nama]

        # Kecepatan
        t_list = []
        for _ in range(ULANG):
            t0 = time.time()
            kp1, des1 = det.detectAndCompute(gray1, None)
            t_list.append((time.time() - t0) * 1000)
        speed_ms = np.mean(t_list)

        # Jumlah keypoint
        kp1, des1 = det.detectAndCompute(gray1, None)
        n_kp      = len(kp1)

        if des1 is None or len(kp1) < 2:
            data[nama] = {"speed_ms": speed_ms, "n_kp": 0,
                          "match_ratio": 0, "robustness": 0}
            continue

        if tipe == "uint8":
            des1 = np.uint8(des1)

        # Match ratio (gambar & gambar yang sama → harusnya tinggi)
        kp2, des2 = det.detectAndCompute(gray1, None)
        if des2 is None or len(kp2) < 2:
            match_ratio = 0
        else:
            if tipe == "uint8":
                des2 = np.uint8(des2)
            try:
                pasang = matcher.knnMatch(des1, des2, k=2)
                mb     = [p[0] for p in pasang
                          if len(p) == 2 and p[0].distance < rasio * p[1].distance]
                match_ratio = len(mb) / max(n_kp, 1)
            except Exception:
                match_ratio = 0

        # Robustness terhadap rotasi 45°
        kp_r, des_r = det.detectAndCompute(gray_rot, None)
        if des_r is None or len(kp_r) < 2:
            robustness = 0
        else:
            if tipe == "uint8":
                des_r = np.uint8(des_r)
            try:
                pasang_r = matcher.knnMatch(des1, des_r, k=2)
                mb_r     = [p[0] for p in pasang_r
                            if len(p) == 2 and p[0].distance < rasio * p[1].distance]
                robustness = len(mb_r) / max(n_kp, 1)
            except Exception:
                robustness = 0

        data[nama] = {
            "speed_ms"  : speed_ms,
            "n_kp"      : n_kp,
            "match_ratio": match_ratio,
            "robustness" : robustness
        }
        print(f"  {nama:5s}: speed={speed_ms:.1f}ms | n_kp={n_kp} | "
              f"match_ratio={match_ratio:.2f} | robustness={robustness:.2f}")

    # ── Normalisasi ────────────────────────────────────────────────────────────
    # Speed: lebih cepat = skor lebih tinggi → invers
    speeds      = [data[n]["speed_ms"]   for n in detektors]
    kps         = [data[n]["n_kp"]       for n in detektors]
    mrs         = [data[n]["match_ratio"] for n in detektors]
    robs        = [data[n]["robustness"]  for n in detektors]

    def norm(lst):
        mx = max(lst) if max(lst) > 0 else 1
        return [v / mx for v in lst]

    speed_norm = [1 - v for v in norm(speeds)]  # invers: lebih cepat = skor tinggi
    kp_norm    = norm(kps)
    mr_norm    = norm(mrs)
    rob_norm   = norm(robs)

    # ── Radar chart ────────────────────────────────────────────────────────────
    kategori = ["Speed\n(lebih cepat ↑)", "N Keypoint\n(lebih banyak ↑)",
                "Match Ratio\n(lebih tinggi ↑)", "Robustness\n(rotasi 45°)"]
    N_kat    = len(kategori)
    sudut    = np.linspace(0, 2 * np.pi, N_kat, endpoint=False).tolist()
    # Tutup polygon
    sudut   += sudut[:1]

    warna    = {"SIFT": "#27ae60", "ORB": "#e74c3c",
                "AKAZE": "#3498db", "BRISK": "#f39c12"}
    nilai_map = {
        "SIFT" : [speed_norm[0], kp_norm[0], mr_norm[0], rob_norm[0]],
        "ORB"  : [speed_norm[1], kp_norm[1], mr_norm[1], rob_norm[1]],
        "AKAZE": [speed_norm[2], kp_norm[2], mr_norm[2], rob_norm[2]],
        "BRISK": [speed_norm[3], kp_norm[3], mr_norm[3], rob_norm[3]],
    }

    fig, ax = plt.subplots(figsize=(8, 8),
                            subplot_kw=dict(polar=True))

    for nama, nilai in nilai_map.items():
        vals = nilai + nilai[:1]   # tutup poligon
        ax.plot(sudut, vals, "o-", color=warna[nama], linewidth=2, label=nama)
        ax.fill(sudut, vals, color=warna[nama], alpha=0.12)

    ax.set_thetagrids(np.degrees(sudut[:-1]), kategori, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title("Radar Chart: Perbandingan Deskriptor\n"
                 "SIFT vs ORB vs AKAZE vs BRISK\n"
                 "(nilai ternormalisasi 0–1)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_demo4_radar_chart_deskriptor.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] Gambar disimpan.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Modul 04 — 13: Perbandingan Deskriptor Fitur")
    print("=" * 60)

    gray1, gray2, img1_bgr, img2_bgr = muat_gambar_pasangan()
    print(f"Ukuran gambar-1: {gray1.shape}, gambar-2: {gray2.shape}")

    demo_dimensi_tipe_deskriptor     (gray1, img1_bgr)
    demo_benchmark_kecepatan         (gray1)
    demo_matching_quality_precision_k(gray1, gray2, img1_bgr, img2_bgr)
    demo_radar_chart_perbandingan    (gray1, gray2)

    print("\nSemua demo selesai. Output disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
