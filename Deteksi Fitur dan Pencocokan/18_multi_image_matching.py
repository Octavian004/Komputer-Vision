"""
==========================================================================
PERCOBAAN 18: MULTI-IMAGE MATCHING
==========================================================================
Modul  : Modul04 - Deteksi Fitur dan Pencocokan
Topik  : Multi-Image Matching (pencocokan fitur lintas banyak gambar)
Tujuan :
  1. Mencocokkan fitur ORB antar setiap pasangan dari sekumpulan gambar
     dan memvisualisasikan jumlah kecocokan sebagai matriks heatmap.
  2. Membangun sistem image retrieval sederhana: query image di-ranking
     terhadap basis data gambar berdasarkan skor kecocokan fitur.
  3. Membangun graf transisi (similarity graph) yang menghubungkan
     gambar-gambar dengan kekuatan kecocokan sebagai bobot sisi.
  4. Mengelompokkan gambar-gambar serupa berdasarkan ambang batas
     kemiripan fitur dan memvisualisasikan kluster yang terbentuk.

Konsep yang dipelajari:
  - N-way matching      : mencocokkan semua pasangan dari N gambar
  - Image retrieval     : pencarian gambar berdasarkan kemiripan fitur
  - Similarity graph    : graf berbobot yang merepresentasikan kemiripan
  - Clustering by feature: pengelompokan gambar berdasarkan deskriptor
  - ORB                 : detektor dan deskriptor fitur yang efisien
  - BFMatcher           : Brute-Force Matcher dengan uji rasio Hamming

Hasil disimpan ke folder output/ sebagai file PNG beresolusi tinggi.
==========================================================================
"""

# Mengimpor library OpenCV untuk deteksi fitur dan pencocokan
import cv2

# Mengimpor NumPy untuk komputasi matriks dan array numerik
import numpy as np

# Mengimpor os untuk pengelolaan path dan direktori
import os

# Mengimpor matplotlib untuk visualisasi dan penyimpanan hasil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Mengimpor itertools untuk menghasilkan kombinasi pasangan gambar
import itertools

# ---------------------------------------------------------------------------
# Konfigurasi path direktori
# ---------------------------------------------------------------------------

# Direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Direktori gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Direktori untuk menyimpan hasil visualisasi
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat direktori output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fungsi: muat_atau_buat_gambar
# ---------------------------------------------------------------------------

def muat_atau_buat_gambar():
    """
    Memuat gambar dari IMAGE_DIR atau membuat gambar sintetis
    dengan pola/tekstur berbeda jika file tidak tersedia.

    Returns
    -------
    list of dict
        Daftar 4-5 gambar dengan kunci 'nama' (str) dan 'gambar' (ndarray BGR).
    """
    # Nama file yang akan dicoba dimuat dari disk
    kandidat = [
        ("bangunan_a", "building.jpg"),
        ("bangunan_b", "bangunan2.jpg"),
        ("pemandangan", "landscape.jpg"),
        ("teks_cetak",  "buku.jpg"),
        ("wajah_orang", "face.jpg"),
    ]

    hasil = []

    for nama, fname in kandidat:
        fpath = os.path.join(IMAGE_DIR, fname)
        if os.path.isfile(fpath):
            # Memuat gambar dari disk dan mengubah ukuran agar seragam
            img = cv2.imread(fpath)
            if img is not None:
                img = cv2.resize(img, (320, 240))
                hasil.append({"nama": nama, "gambar": img})
                continue

        # Membuat gambar sintetis jika file tidak ditemukan
        img = _buat_gambar_sintetis(nama)
        hasil.append({"nama": nama, "gambar": img})

    return hasil[:5]  # Mengembalikan maksimum 5 gambar


def _buat_gambar_sintetis(nama: str) -> np.ndarray:
    """
    Membuat gambar sintetis dengan pola/tekstur unik berdasarkan nama.

    Parameters
    ----------
    nama : str
        Nama gambar, menentukan pola yang dihasilkan.

    Returns
    -------
    np.ndarray
        Gambar BGR berukuran 240 x 320 piksel.
    """
    # Ukuran gambar sintetis (tinggi x lebar)
    H, W = 240, 320
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    rng = np.random.default_rng(seed=abs(hash(nama)) % (2**31))

    if "bangunan_a" in nama:
        # Pola kotak grid menyerupai jendela bangunan
        canvas[:] = (50, 50, 80)
        for r in range(20, H - 20, 40):
            for c in range(20, W - 20, 50):
                cv2.rectangle(canvas, (c, r), (c + 30, r + 25),
                              (200, 210, 230), -1)
                cv2.rectangle(canvas, (c, r), (c + 30, r + 25),
                              (100, 120, 140), 1)

    elif "bangunan_b" in nama:
        # Pola bangunan dengan rotasi sedikit berbeda (uji invariansi)
        canvas[:] = (60, 70, 50)
        for r in range(15, H - 15, 40):
            for c in range(15, W - 15, 55):
                pts = np.array([
                    [c, r + 25], [c + 15, r], [c + 35, r], [c + 40, r + 25]
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], (180, 200, 160))
                cv2.polylines(canvas, [pts], True, (80, 100, 70), 1)

    elif "pemandangan" in nama:
        # Gradasi langit dan tanah dengan titik-titik tekstur
        for y in range(H):
            warna = int(200 - y * 0.5)
            canvas[y, :] = (warna, warna + 20, 255 - y // 2)
        # Menambahkan pohon sederhana
        for _ in range(12):
            cx, cy = rng.integers(20, W - 20), rng.integers(H // 2, H - 20)
            cv2.circle(canvas, (cx, cy), 18, (30, 120, 30), -1)
            cv2.line(canvas, (cx, cy + 18), (cx, cy + 35), (60, 40, 20), 3)

    elif "teks_cetak" in nama:
        # Simulasi halaman teks dengan garis-garis horizontal
        canvas[:] = (240, 240, 230)
        for y in range(25, H - 20, 18):
            panjang = rng.integers(W // 3, W - 30)
            cv2.line(canvas, (20, y), (panjang, y), (30, 30, 30), 1)
        # Menambahkan beberapa "huruf" sebagai kotak kecil
        for _ in range(30):
            x, y = rng.integers(20, W - 15), rng.integers(20, H - 15)
            cv2.rectangle(canvas, (x, y), (x + 6, y + 10), (20, 20, 20), -1)

    else:
        # Pola wajah sintetis: lingkaran mata dan lengkung mulut
        canvas[:] = (180, 160, 140)
        # Wajah bulat
        cv2.ellipse(canvas, (W // 2, H // 2), (90, 110), 0, 0, 360,
                    (210, 185, 160), -1)
        # Mata kiri dan kanan
        cv2.circle(canvas, (W // 2 - 30, H // 2 - 20), 12,
                   (40, 40, 40), -1)
        cv2.circle(canvas, (W // 2 + 30, H // 2 - 20), 12,
                   (40, 40, 40), -1)
        # Mulut
        cv2.ellipse(canvas, (W // 2, H // 2 + 40), (35, 18),
                    0, 0, 180, (80, 40, 40), 2)

    # Menambahkan noise Gaussian ringan agar fitur ORB lebih banyak terdeteksi
    noise = rng.integers(0, 18, canvas.shape, dtype=np.uint8)
    canvas = cv2.add(canvas, noise)

    return canvas


# ---------------------------------------------------------------------------
# Fungsi pembantu bersama
# ---------------------------------------------------------------------------

def _deteksi_dan_deskripsikan(gambar_bgr: np.ndarray):
    """
    Mendeteksi keypoint dan menghitung deskriptor ORB.

    Parameters
    ----------
    gambar_bgr : np.ndarray
        Gambar dalam format BGR.

    Returns
    -------
    tuple
        (keypoints, descriptors) dari ORB.
    """
    # Membuat detektor ORB dengan jumlah fitur maksimum
    orb = cv2.ORB_create(nfeatures=500)

    # Mengonversi ke grayscale untuk deteksi fitur
    abu = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)

    # Mendeteksi keypoint dan menghitung deskriptor sekaligus
    kp, des = orb.detectAndCompute(abu, None)
    return kp, des


def _cocokkan_pasangan(des1, des2, ambang_rasio: float = 0.75) -> int:
    """
    Mencocokkan dua set deskriptor ORB menggunakan BFMatcher + uji rasio.

    Parameters
    ----------
    des1, des2 : np.ndarray
        Deskriptor biner (ORB) dari dua gambar.
    ambang_rasio : float
        Ambang batas rasio untuk menyaring kecocokan palsu.

    Returns
    -------
    int
        Jumlah kecocokan yang lolos uji rasio.
    """
    # Mengembalikan 0 jika salah satu deskriptor tidak tersedia
    if des1 is None or des2 is None:
        return 0

    # Membuat BFMatcher dengan norma Hamming (sesuai deskriptor biner ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # kNN matching dengan k=2 untuk uji rasio Lowe
    try:
        pasangan = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0

    # Menyaring kecocokan menggunakan uji rasio
    baik = []
    for m_n in pasangan:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ambang_rasio * n.distance:
                baik.append(m)

    return len(baik)


# ---------------------------------------------------------------------------
# Demo 1: Pencocokan N-Arah â€” matriks heatmap jumlah kecocokan
# ---------------------------------------------------------------------------

def demo_pencocokan_n_arah(daftar_gambar: list):
    """
    Mencocokkan fitur ORB antar setiap pasangan dari N gambar dan
    memvisualisasikan jumlah kecocokan sebagai matriks heatmap.

    Parameters
    ----------
    daftar_gambar : list of dict
        Daftar gambar dengan kunci 'nama' dan 'gambar'.
    """
    print("\n=== Demo 1: Pencocokan N-Arah ===")

    N = len(daftar_gambar)
    nama_gambar = [d["nama"] for d in daftar_gambar]

    # --- Mendeteksi fitur untuk setiap gambar ---
    print("  Mendeteksi fitur ORB pada setiap gambar...")
    fitur = []
    for d in daftar_gambar:
        kp, des = _deteksi_dan_deskripsikan(d["gambar"])
        fitur.append({"nama": d["nama"], "kp": kp, "des": des})
        print(f"    {d['nama']}: {len(kp)} keypoint")

    # --- Membangun matriks kecocokan N x N ---
    matriks = np.zeros((N, N), dtype=int)
    print("  Mencocokkan setiap pasangan...")
    for i, j in itertools.combinations(range(N), 2):
        jml = _cocokkan_pasangan(fitur[i]["des"], fitur[j]["des"])
        matriks[i, j] = jml
        # Matriks simetris
        matriks[j, i] = jml
        print(f"    {nama_gambar[i]} â†” {nama_gambar[j]}: {jml} matches")

    # --- Visualisasi matriks sebagai heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Demo 1: Pencocokan N-Arah — Matriks Heatmap",
                 fontsize=14, fontweight="bold")

    # Membuat colormap biru-kuning untuk heatmap
    cmap_bk = LinearSegmentedColormap.from_list(
        "bk", ["#0a0a2a", "#1a4a8a", "#f0b030", "#ffffff"])

    # Plot heatmap matriks kecocokan
    ax = axes[0]
    im = ax.imshow(matriks, cmap=cmap_bk, aspect="auto")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(nama_gambar, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(nama_gambar, fontsize=9)
    ax.set_title("Matriks Jumlah Kecocokan", fontsize=11)

    # Menampilkan nilai di setiap sel heatmap
    for i in range(N):
        for j in range(N):
            warna_teks = "white" if matriks[i, j] < matriks.max() * 0.6 else "black"
            ax.text(j, i, str(matriks[i, j]),
                    ha="center", va="center",
                    color=warna_teks, fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Jumlah Kecocokan")

    # Plot bar chart jumlah kecocokan total per gambar
    ax2 = axes[1]
    total_per_gambar = matriks.sum(axis=1)
    warna_bar = ["#2266cc", "#cc4422", "#22aa44", "#aa22cc", "#ccaa00"]
    bar = ax2.bar(nama_gambar, total_per_gambar,
                  color=warna_bar[:N], edgecolor="black", linewidth=0.8)
    ax2.set_title("Total Kecocokan per Gambar", fontsize=11)
    ax2.set_xlabel("Gambar")
    ax2.set_ylabel("Total Kecocokan")
    ax2.tick_params(axis="x", rotation=30)

    # Menambahkan label nilai di atas setiap batang
    for b in bar:
        ax2.text(b.get_x() + b.get_width() / 2,
                 b.get_height() + 0.5,
                 str(int(b.get_height())),
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # Menyimpan hasil ke disk
    path_out = os.path.join(OUTPUT_DIR, "18_pencocokan_n_arah.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    print(f"  Disimpan: {path_out}")
    plt.show()

    # Mengembalikan matriks dan fitur untuk digunakan demo selanjutnya
    return matriks, fitur


# ---------------------------------------------------------------------------
# Demo 2: Image Retrieval Dasar â€” query vs basis data, ranking top-3
# ---------------------------------------------------------------------------

def demo_pengambilan_citra(daftar_gambar: list, fitur: list):
    """
    Mensimulasikan image retrieval: satu gambar dijadikan query,
    gambar-gambar lain dijadikan basis data. Hasilnya diurutkan
    berdasarkan skor kecocokan dan divisualisasikan top-3.

    Parameters
    ----------
    daftar_gambar : list of dict
        Daftar gambar dengan kunci 'nama' dan 'gambar'.
    fitur : list of dict
        Daftar fitur (kp, des) yang sudah dihitung di Demo 1.
    """
    print("\n=== Demo 2: Image Retrieval Dasar ===")

    # Menggunakan gambar pertama sebagai query
    idx_query = 0
    query = daftar_gambar[idx_query]
    des_query = fitur[idx_query]["des"]
    print(f"  Query gambar: {query['nama']}")

    # Menghitung skor kecocokan query terhadap semua gambar basis data
    skor = []
    for i, d in enumerate(daftar_gambar):
        if i == idx_query:
            # Melewati gambar query itu sendiri
            continue
        jml = _cocokkan_pasangan(des_query, fitur[i]["des"])
        skor.append({"nama": d["nama"], "gambar": d["gambar"],
                     "skor": jml, "indeks": i})
        print(f"    {d['nama']}: skor = {jml}")

    # Mengurutkan basis data dari yang paling cocok ke paling tidak cocok
    skor.sort(key=lambda x: x["skor"], reverse=True)
    top3 = skor[:3]

    # --- Visualisasi hasil retrieval ---
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle("Demo 2: Image Retrieval Dasar — Ranking Kecocokan",
                 fontsize=14, fontweight="bold")

    def tampilkan(ax, img_bgr, judul, sub="", warna_bingkai="gray"):
        """Menampilkan satu gambar pada subplot dengan bingkai berwarna."""
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(judul, fontsize=10, fontweight="bold", pad=3)
        if sub:
            ax.set_xlabel(sub, fontsize=9, color=warna_bingkai)
        for sp in ax.spines.values():
            sp.set_edgecolor(warna_bingkai)
            sp.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Baris atas: query dan 3 gambar hasil teratas
    tampilkan(axes[0, 0], query["gambar"], "QUERY", "", "#dd4400")
    axes[0, 0].set_ylabel("Query", fontsize=9, color="#dd4400",
                           fontweight="bold")

    warna_rank = ["#ffd700", "#c0c0c0", "#cd7f32"]  # emas, perak, perunggu
    label_rank = ["Rank 1 (Terbaik)", "Rank 2", "Rank 3"]
    for k, (hasil, warna, label) in enumerate(zip(top3, warna_rank, label_rank)):
        tampilkan(axes[0, k + 1], hasil["gambar"],
                  label, f"Skor: {hasil['skor']}", warna)

    # Baris bawah: bar chart skor kecocokan semua kandidat
    ax_gabung = fig.add_subplot(2, 1, 2)
    ax_gabung.set_position([0.08, 0.05, 0.88, 0.38])

    nama_kandidat = [s["nama"] for s in skor]
    skor_kandidat = [s["skor"] for s in skor]
    warna_bar_list = warna_rank[:len(skor)] + ["#aaaaaa"] * max(
        0, len(skor) - 3)
    bar = ax_gabung.barh(nama_kandidat, skor_kandidat,
                         color=warna_bar_list, edgecolor="black", height=0.5)
    ax_gabung.set_xlabel("Skor Kecocokan (jumlah matches)")
    ax_gabung.set_title("Peringkat Semua Gambar Basis Data", fontsize=10)
    ax_gabung.axvline(0, color="black", linewidth=0.8)

    # Menambahkan label skor di setiap batang
    for b, s in zip(bar, skor_kandidat):
        ax_gabung.text(s + 0.3, b.get_y() + b.get_height() / 2,
                       str(s), va="center", fontsize=9)

    # Menyembunyikan subplot baris bawah asli
    for ax in axes[1, :]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Menyimpan hasil ke disk
    path_out = os.path.join(OUTPUT_DIR, "18_pengambilan_citra.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    print(f"  Disimpan: {path_out}")
    plt.show()


# ---------------------------------------------------------------------------
# Demo 3: Graf Transisi â€” visualisasi kemiripan sebagai graf berbobot
# ---------------------------------------------------------------------------

def demo_graf_transisi(daftar_gambar: list, matriks: np.ndarray):
    """
    Membangun dan memvisualisasikan graf transisi (similarity graph).
    Setiap simpul mewakili satu gambar; setiap sisi diberi bobot berupa
    jumlah kecocokan. Sisi divisualisasikan sebagai panah dengan ketebalan
    dan warna proporsional terhadap kekuatan kecocokan.

    Parameters
    ----------
    daftar_gambar : list of dict
        Daftar gambar dengan kunci 'nama' dan 'gambar'.
    matriks : np.ndarray
        Matriks kecocokan N x N dari Demo 1.
    """
    print("\n=== Demo 3: Graf Transisi ===")

    N = len(daftar_gambar)
    nama = [d["nama"] for d in daftar_gambar]

    # Menempatkan simpul secara melingkar (circular layout)
    sudut = np.linspace(0, 2 * np.pi, N, endpoint=False)
    posisi = {i: (np.cos(sudut[i]), np.sin(sudut[i])) for i in range(N)}

    # Nilai skor minimum agar sisi ditampilkan (menghilangkan sisi lemah)
    ambang_tampil = matriks.max() * 0.1

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_title("Demo 3: Graf Transisi — Kemiripan Antar Gambar",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    skor_maks = matriks.max() if matriks.max() > 0 else 1

    # Menggambar sisi antar simpul (panah dua-arah)
    for i, j in itertools.combinations(range(N), 2):
        skor = matriks[i, j]
        if skor < ambang_tampil:
            # Melewati sisi yang terlalu lemah
            continue

        # Menghitung ketebalan dan opasitas proporsional terhadap skor
        tebal = 1.0 + 5.0 * (skor / skor_maks)
        opasitas = 0.3 + 0.7 * (skor / skor_maks)

        # Menginterpolasi warna dari merah (lemah) ke hijau (kuat)
        t = skor / skor_maks
        warna = (1 - t, t * 0.8, 0.2)

        xi, yi = posisi[i]
        xj, yj = posisi[j]

        # Menggambar panah dari i ke j
        ax.annotate("",
                    xy=(xj * 0.82, yj * 0.82),
                    xytext=(xi * 0.82, yi * 0.82),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=warna,
                        lw=tebal,
                        alpha=opasitas,
                        connectionstyle="arc3,rad=0.12",
                    ))

        # Menampilkan label skor di tengah sisi
        xm = (xi + xj) / 2 * 0.88
        ym = (yi + yj) / 2 * 0.88
        ax.text(xm, ym, str(skor), ha="center", va="center",
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc=warna, ec="none", alpha=0.85))

    # Memilih warna simpul yang berbeda untuk setiap gambar
    warna_simpul = ["#3366cc", "#cc3322", "#22aa44", "#aa22cc", "#cc8800"]

    # Menggambar simpul (lingkaran) dan label nama gambar
    for i in range(N):
        xi, yi = posisi[i]
        # Lingkaran simpul
        lingkaran = plt.Circle((xi, yi), 0.18, color=warna_simpul[i],
                               zorder=5, linewidth=2, ec="white")
        ax.add_patch(lingkaran)

        # Nomor indeks di dalam simpul
        ax.text(xi, yi, str(i + 1), ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=6)

        # Nama gambar di luar simpul (mengarah ke luar lingkaran)
        offset_x = xi * 0.42
        offset_y = yi * 0.42
        ax.text(xi + offset_x, yi + offset_y, nama[i],
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color=warna_simpul[i],
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec=warna_simpul[i],
                          alpha=0.9, linewidth=1.5))

    # Menambahkan catatan legenda skala
    ax.text(-1.55, -1.55,
            "Ketebalan sisi \u221d kekuatan kecocokan\n"
            "Warna merah=lemah, hijau=kuat",
            fontsize=8, color="gray", va="bottom")

    plt.tight_layout()

    # Menyimpan hasil graf ke disk
    path_out = os.path.join(OUTPUT_DIR, "18_graf_transisi.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    print(f"  Disimpan: {path_out}")
    plt.show()


# ---------------------------------------------------------------------------
# Demo 4: Kluster Citra Serupa â€” pengelompokan berdasarkan fitur
# ---------------------------------------------------------------------------

def demo_kluster_citra_serupa(daftar_gambar: list, matriks: np.ndarray):
    """
    Mengelompokkan gambar-gambar serupa menggunakan algoritma greedy
    berbasis ambang batas kemiripan (Union-Find). Setiap kluster
    divisualisasikan dengan bingkai berwarna dan matriks blok diagonal.

    Parameters
    ----------
    daftar_gambar : list of dict
        Daftar gambar dengan kunci 'nama' dan 'gambar'.
    matriks : np.ndarray
        Matriks kecocokan N x N dari Demo 1.
    """
    print("\n=== Demo 4: Kluster Citra Serupa ===")

    N = len(daftar_gambar)

    # Menentukan ambang batas kemiripan secara adaptif
    # (menggunakan median skor non-nol sebagai batas kluster)
    skor_non_nol = matriks[matriks > 0]
    if len(skor_non_nol) > 0:
        ambang = float(np.median(skor_non_nol))
    else:
        ambang = 5.0
    print(f"  Ambang batas kluster: {ambang:.1f} kecocokan")

    # Algoritma Union-Find sederhana untuk membangun kluster
    induk = list(range(N))

    def cari(x):
        """Mencari akar dari simpul x dalam Union-Find."""
        while induk[x] != x:
            induk[x] = induk[induk[x]]
            x = induk[x]
        return x

    def gabung(x, y):
        """Menggabungkan dua kluster yang mengandung x dan y."""
        rx, ry = cari(x), cari(y)
        if rx != ry:
            induk[rx] = ry

    # Menggabungkan gambar yang skor pasangannya melebihi ambang batas
    for i, j in itertools.combinations(range(N), 2):
        if matriks[i, j] >= ambang:
            gabung(i, j)
            print(f"    Menggabungkan: {daftar_gambar[i]['nama']} â†” "
                  f"{daftar_gambar[j]['nama']} (skor={matriks[i,j]})")

    # Mengumpulkan anggota setiap kluster
    dari_kluster: dict = {}
    for i in range(N):
        akar = cari(i)
        dari_kluster.setdefault(akar, []).append(i)

    kluster_list = list(dari_kluster.values())
    Nk = len(kluster_list)
    print(f"  Jumlah kluster yang terbentuk: {Nk}")
    for k, anggota in enumerate(kluster_list):
        nama_anggota = [daftar_gambar[i]["nama"] for i in anggota]
        print(f"    Kluster {k+1}: {nama_anggota}")

    # Warna untuk setiap kluster
    warna_kluster = ["#3366cc", "#cc3322", "#22aa44",
                     "#aa22cc", "#cc8800"][:Nk]

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle("Demo 4: Kluster Citra Serupa — Pengelompokan Berbasis Fitur",
                 fontsize=13, fontweight="bold")

    # --- Panel kiri: thumbnail gambar dikelompokkan per kluster ---
    ax_kiri = fig.add_subplot(1, 2, 1)
    ax_kiri.set_xlim(0, 1)
    ax_kiri.set_ylim(0, 1)
    ax_kiri.axis("off")
    ax_kiri.set_title("Hasil Pengelompokan Gambar", fontsize=11)

    tinggi_baris = 1.0 / N
    thumb_lebar = 0.18
    thumb_tinggi = tinggi_baris * 0.85

    for i, d in enumerate(daftar_gambar):
        # Menentukan kluster dari gambar i
        akar_i = cari(i)
        idx_k = list(dari_kluster.keys()).index(akar_i)
        warna = warna_kluster[idx_k % len(warna_kluster)]

        y_pos = 1.0 - (i + 1) * tinggi_baris + 0.02

        # Menampilkan thumbnail gambar
        img_rgb = cv2.cvtColor(d["gambar"], cv2.COLOR_BGR2RGB)
        ax_thumb = fig.add_axes([0.06, y_pos, thumb_lebar, thumb_tinggi])
        ax_thumb.imshow(img_rgb)
        ax_thumb.set_xticks([])
        ax_thumb.set_yticks([])
        for sp in ax_thumb.spines.values():
            sp.set_edgecolor(warna)
            sp.set_linewidth(3)

        # Label nama dan kluster di sebelah thumbnail
        ax_kiri.text(0.32, y_pos + thumb_tinggi / 2,
                     f"{i+1}. {d['nama']}",
                     va="center", ha="left", fontsize=9,
                     fontweight="bold", color=warna)
        ax_kiri.text(0.32, y_pos + thumb_tinggi / 2 - 0.03,
                     f"Kluster {idx_k + 1}",
                     va="center", ha="left", fontsize=8,
                     color=warna, style="italic")

        # Menggambar kotak penanda kluster
        rect = mpatches.FancyBboxPatch(
            (0.28, y_pos - 0.01), 0.68, thumb_tinggi + 0.01,
            boxstyle="round,pad=0.01",
            linewidth=2, edgecolor=warna, facecolor=warna + "22",
            transform=ax_kiri.transData, zorder=0)
        ax_kiri.add_patch(rect)

    # --- Panel kanan: matriks kemiripan dengan blok kluster ---
    ax_mat = fig.add_subplot(1, 2, 2)
    ax_mat.set_title("Matriks Kemiripan (blok = kluster)", fontsize=11)

    # Mengurutkan indeks gambar berdasarkan kluster agar blok terlihat
    urutan = []
    for anggota in kluster_list:
        urutan.extend(anggota)

    matriks_terurut = matriks[np.ix_(urutan, urutan)]
    nama_terurut = [daftar_gambar[i]["nama"] for i in urutan]

    cmap_merah = LinearSegmentedColormap.from_list(
        "mr", ["#0d0d0d", "#7a0000", "#ff6600", "#ffdd00"])
    im = ax_mat.imshow(matriks_terurut, cmap=cmap_merah, aspect="auto")
    ax_mat.set_xticks(range(N))
    ax_mat.set_yticks(range(N))
    ax_mat.set_xticklabels(nama_terurut, rotation=30,
                            ha="right", fontsize=8)
    ax_mat.set_yticklabels(nama_terurut, fontsize=8)
    plt.colorbar(im, ax=ax_mat, label="Jumlah Kecocokan")

    # Menampilkan nilai di setiap sel matriks
    for ii in range(N):
        for jj in range(N):
            val = matriks_terurut[ii, jj]
            tc = "white" if val < matriks.max() * 0.55 else "black"
            ax_mat.text(jj, ii, str(val),
                        ha="center", va="center",
                        color=tc, fontsize=7, fontweight="bold")

    # Menggambar kotak melingkari blok diagonal (dalam kluster)
    pos = 0
    for anggota in kluster_list:
        uk = len(anggota)
        if uk > 1:
            # Menggambar persegi dashed di sekitar blok kluster
            rect_k = mpatches.Rectangle(
                (pos - 0.5, pos - 0.5), uk, uk,
                linewidth=2.5, edgecolor="cyan",
                facecolor="none", linestyle="--")
            ax_mat.add_patch(rect_k)
        pos += uk

    plt.tight_layout()

    # Menyimpan hasil kluster ke disk
    path_out = os.path.join(OUTPUT_DIR, "18_kluster_citra_serupa.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    print(f"  Disimpan: {path_out}")
    plt.show()


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------

def main():
    """Menjalankan seluruh demo Multi-Image Matching secara berurutan."""
    print("=" * 60)
    print("PERCOBAAN 18: MULTI-IMAGE MATCHING")
    print("=" * 60)

    # Memuat atau membuat gambar input
    print("\nMemuat gambar...")
    daftar_gambar = muat_atau_buat_gambar()
    print(f"Jumlah gambar: {len(daftar_gambar)}")
    for d in daftar_gambar:
        h, w = d["gambar"].shape[:2]
        print(f"  {d['nama']}: {w}x{h} piksel")

    # Demo 1: Pencocokan N-Arah â€” matriks heatmap
    matriks, fitur = demo_pencocokan_n_arah(daftar_gambar)

    # Demo 2: Image Retrieval â€” ranking query vs basis data
    demo_pengambilan_citra(daftar_gambar, fitur)

    # Demo 3: Graf Transisi â€” kemiripan sebagai graf berbobot
    demo_graf_transisi(daftar_gambar, matriks)

    # Demo 4: Kluster Citra Serupa â€” pengelompokan berbasis fitur
    demo_kluster_citra_serupa(daftar_gambar, matriks)

    print("\n" + "=" * 60)
    print("Semua demo selesai. Hasil disimpan di folder output/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

