"""
PIPELINE ANALISIS CITRA - MODUL 3 PEMROSESAN CITRA
Praktikum Komputer Vision - Semester 6

Deskripsi: Program lengkap untuk pipeline analisis citra dengan preprocessing,
segmentasi, thresholding, edge detection, morfologi, transformasi Fourier,
dan visualisasi hasil dengan laporan statistik otomatis.

Author: Praktikum CV
Date: 2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
import os

matplotlib.use('Agg')  # Non-interactive backend

# =====================================================================
# KONFIGURASI PARAMETER - BAGIAN INI DAPAT DISESUAIKAN
# =====================================================================

# Path
INPUT_IMAGE_NAME = "sample.jpg"  # Ganti dengan nama file gambar di folder input/
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# Preprocessing Parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# HSV Segmentation Parameters
# Range untuk warna merah (2 range karena red ada di 0 dan 180)
HSV_RED_LOWER1 = np.array([0, 100, 100])
HSV_RED_UPPER1 = np.array([10, 255, 255])
HSV_RED_LOWER2 = np.array([170, 100, 100])
HSV_RED_UPPER2 = np.array([180, 255, 255])

# Range untuk warna kedua (bisa disesuaikan - contoh: biru)
HSV_BLUE_LOWER = np.array([100, 100, 100])
HSV_BLUE_UPPER = np.array([130, 255, 255])

# Thresholding Parameters
OTSU_THRESHOLD = 127
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_CONSTANT = 2

# Edge Detection Parameters
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200
SOBEL_KSIZE = 3

# Morphological Operation Parameters
MORPH_KERNEL_SIZE = (5, 5)
MORPH_KERNEL_TYPE = cv2.MORPH_ELLIPSE

# Connected Components Parameters
CONNECTIVITY = 8

# FFT Parameters
FILTER_D = 30  # Diameter untuk low-pass/high-pass filter

# Visualization Parameters
FIGURE_DPI = 100
SUBPLOT_FIGSIZE = (16, 20)

# =====================================================================
# FUNGSI PREPROCESSING
# =====================================================================

def load_image(image_path):
    """
    Memuat citra dari file.
    
    Args:
        image_path (str): Path ke file image
        
    Returns:
        np.ndarray: Citra dalam BGR format, atau None jika gagal
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' tidak ditemukan!")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Gagal membaca file '{image_path}'")
        return None
    
    print(f"✓ Citra berhasil dimuat: {img.shape}")
    return img


def apply_clahe(img, clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_SIZE):
    """
    Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    pada channel L dari LAB colorspace.
    
    Args:
        img (np.ndarray): Citra input dalam BGR
        clip_limit (float): Batas contrast
        tile_size (tuple): Ukuran tile grid
        
    Returns:
        np.ndarray: Citra hasil CLAHE dalam BGR
    """
    # Konversi BGR ke LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Aplikasikan CLAHE pada channel L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_clahe = clahe.apply(l_channel)
    
    # Merge kembali
    lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])
    
    # Konversi kembali ke BGR
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    print("✓ CLAHE preprocessing selesai")
    return img_clahe


def apply_bilateral_filter(img, d=BILATERAL_D, 
                          sigma_color=BILATERAL_SIGMA_COLOR,
                          sigma_space=BILATERAL_SIGMA_SPACE):
    """
    Menerapkan bilateral filter untuk noise reduction sambil 
    mempertahankan edge.
    
    Args:
        img (np.ndarray): Citra input
        d (int): Diameter dari setiap pixel
        sigma_color (float): Filter sigma dalam color space
        sigma_space (float): Filter sigma dalam coordinate space
        
    Returns:
        np.ndarray: Citra hasil bilateral filter
    """
    img_filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    print("✓ Bilateral filter selesai")
    return img_filtered


def preprocessing(img):
    """
    Pipeline preprocessing lengkap: CLAHE + Bilateral Filter.
    
    Args:
        img (np.ndarray): Citra input
        
    Returns:
        np.ndarray: Citra hasil preprocessing
    """
    img = apply_clahe(img)
    img = apply_bilateral_filter(img)
    return img


# =====================================================================
# FUNGSI SEGMENTASI DAN THRESHOLDING
# =====================================================================

def hsv_segmentation(img):
    """
    Segmentasi HSV untuk mendeteksi objek berdasarkan warna.
    Mendeteksi warna merah (2 range) dan biru.
    
    Args:
        img (np.ndarray): Citra input dalam BGR
        
    Returns:
        tuple: (mask_merah, mask_biru, img_hsv)
    """
    # Konversi ke HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Segmentasi merah (2 range)
    mask_red1 = cv2.inRange(img_hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
    mask_red2 = cv2.inRange(img_hsv, HSV_RED_LOWER2, HSV_RED_UPPER2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Segmentasi biru
    mask_blue = cv2.inRange(img_hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    
    print("✓ Segmentasi HSV selesai (merah + biru)")
    return mask_red, mask_blue, img_hsv


def otsu_thresholding(img):
    """
    Thresholding menggunakan Otsu's method.
    
    Args:
        img (np.ndarray): Citra input grayscale
        
    Returns:
        tuple: (threshold_value, binary_img)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print("✓ Otsu thresholding selesai")
    return binary


def adaptive_thresholding(img):
    """
    Thresholding menggunakan adaptive method (Gaussian).
    
    Args:
        img (np.ndarray): Citra input grayscale
        
    Returns:
        np.ndarray: Citra hasil adaptive thresholding
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE,
                                   ADAPTIVE_CONSTANT)
    
    print("✓ Adaptive thresholding selesai")
    return binary


# =====================================================================
# FUNGSI EDGE DETECTION
# =====================================================================

def canny_edge_detection(img, threshold1=CANNY_THRESHOLD1, 
                        threshold2=CANNY_THRESHOLD2):
    """
    Deteksi edge menggunakan algoritma Canny.
    
    Args:
        img (np.ndarray): Citra input
        threshold1 (float): Lower threshold
        threshold2 (float): Upper threshold
        
    Returns:
        np.ndarray: Citra hasil Canny edge detection
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(img, threshold1, threshold2)
    print("✓ Canny edge detection selesai")
    return edges


def sobel_edge_detection(img, ksize=SOBEL_KSIZE):
    """
    Deteksi edge menggunakan operasi Sobel (X dan Y).
    
    Args:
        img (np.ndarray): Citra input
        ksize (int): Kernel size untuk Sobel
        
    Returns:
        tuple: (magnitude, direction)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Hitung gradient X dan Y
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Hitung magnitude dan direction
    magnitude = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    direction = np.arctan2(sobely, sobelx)
    
    print("✓ Sobel edge detection selesai")
    return magnitude, direction


# =====================================================================
# FUNGSI MORFOLOGI
# =====================================================================

def erode(img, kernel_size=MORPH_KERNEL_SIZE, iterations=1):
    """
    Operasi erosion (transformasi morfologi).
    
    Args:
        img (np.ndarray): Citra input binary
        kernel_size (tuple): Ukuran kernel
        iterations (int): Jumlah iterasi
        
    Returns:
        np.ndarray: Citra hasil erosion
    """
    kernel = cv2.getStructuringElement(MORPH_KERNEL_TYPE, kernel_size)
    result = cv2.erode(img, kernel, iterations=iterations)
    print(f"✓ Erosion selesai (iterasi: {iterations})")
    return result


def dilate(img, kernel_size=MORPH_KERNEL_SIZE, iterations=1):
    """
    Operasi dilation (transformasi morfologi).
    
    Args:
        img (np.ndarray): Citra input binary
        kernel_size (tuple): Ukuran kernel
        iterations (int): Jumlah iterasi
        
    Returns:
        np.ndarray: Citra hasil dilation
    """
    kernel = cv2.getStructuringElement(MORPH_KERNEL_TYPE, kernel_size)
    result = cv2.dilate(img, kernel, iterations=iterations)
    print(f"✓ Dilation selesai (iterasi: {iterations})")
    return result


def opening(img, kernel_size=MORPH_KERNEL_SIZE):
    """
    Operasi opening = Erosion + Dilation (hapus noise kecil).
    
    Args:
        img (np.ndarray): Citra input binary
        kernel_size (tuple): Ukuran kernel
        
    Returns:
        np.ndarray: Citra hasil opening
    """
    kernel = cv2.getStructuringElement(MORPH_KERNEL_TYPE, kernel_size)
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    print("✓ Opening selesai")
    return result


def closing(img, kernel_size=MORPH_KERNEL_SIZE):
    """
    Operasi closing = Dilation + Erosion (isi hole kecil).
    
    Args:
        img (np.ndarray): Citra input binary
        kernel_size (tuple): Ukuran kernel
        
    Returns:
        np.ndarray: Citra hasil closing
    """
    kernel = cv2.getStructuringElement(MORPH_KERNEL_TYPE, kernel_size)
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    print("✓ Closing selesai")
    return result


# =====================================================================
# FUNGSI CONNECTED COMPONENTS
# =====================================================================

def connected_components_analysis(img):
    """
    Analisis connected components dan ekstraksi statistik.
    
    Args:
        img (np.ndarray): Citra input binary
        
    Returns:
        tuple: (num_labels, labels, stats, centroids)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, CONNECTIVITY, cv2.CV_32S
    )
    
    print(f"✓ Connected components analysis selesai (label: {num_labels})")
    return num_labels, labels, stats, centroids


def visualize_connected_components(labels, num_labels):
    """
    Visualisasi connected components dengan pewarnaan berbeda.
    
    Args:
        labels (np.ndarray): Label map dari connectedComponentsWithStats
        num_labels (int): Jumlah label
        
    Returns:
        np.ndarray: Citra hasil visualisasi
    """
    # Warna random untuk setiap label
    colors = np.random.randint(0, 256, size=(num_labels, 3))
    colors[0] = [0, 0, 0]  # Background hitam
    
    colored = colors[labels]
    return colored


# =====================================================================
# FUNGSI FOURIER TRANSFORM
# =====================================================================

def compute_fft(img):
    """
    Menghitung Fast Fourier Transform dari citra.
    
    Args:
        img (np.ndarray): Citra input grayscale
        
    Returns:
        tuple: (f_shift, magnitude_spectrum, phase_spectrum)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Hitung FFT
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    
    # Magnitude dan Phase
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    phase = np.angle(f_shift)
    
    print("✓ FFT computation selesai")
    return f_shift, magnitude, phase


def create_lowpass_filter(shape, d=FILTER_D):
    """
    Membuat low-pass filter (Gaussian).
    
    Args:
        shape (tuple): Ukuran citra (height, width)
        d (int): Parameter diameter filter
        
    Returns:
        np.ndarray: Low-pass filter
    """
    h, w = shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Center
    cx, cy = w // 2, h // 2
    distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Gaussian lowpass filter
    lpf = np.exp(-(distance**2) / (2 * (d**2)))
    return lpf


def create_highpass_filter(shape, d=FILTER_D):
    """
    Membuat high-pass filter (Gaussian).
    
    Args:
        shape (tuple): Ukuran citra (height, width)
        d (int): Parameter diameter filter
        
    Returns:
        np.ndarray: High-pass filter
    """
    lpf = create_lowpass_filter(shape, d)
    hpf = 1 - lpf
    return hpf


def apply_frequency_filter(f_shift, filter_mask):
    """
    Menerapkan frequency domain filter pada FFT hasil.
    
    Args:
        f_shift (np.ndarray): Shifted FFT
        filter_mask (np.ndarray): Filter mask
        
    Returns:
        np.ndarray: Citra hasil filtered dalam spatial domain
    """
    # Aplikasikan filter
    f_filtered = f_shift * filter_mask
    
    # Inverse shift dan IFFT
    f_ishift = np.fft.ifftshift(f_filtered)
    img_filtered = np.fft.ifft2(f_ishift).real
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    
    return img_filtered


# =====================================================================
# FUNGSI VISUALISASI
# =====================================================================

def create_visualization_grid(results_dict, title_prefix=""):
    """
    Membuat grid visualisasi semua hasil processing.
    
    Args:
        results_dict (dict): Dictionary berisi nama dan citra hasil
        title_prefix (str): Prefiks untuk judul
        
    Returns:
        str: Path file hasil visualisasi
    """
    num_images = len(results_dict)
    rows = (num_images + 2) // 3  # 3 kolom
    cols = min(3, num_images)
    
    fig, axes = plt.subplots(rows, cols, figsize=SUBPLOT_FIGSIZE, dpi=FIGURE_DPI)
    
    # Handle single or multiple axes
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    axes = axes.flatten()
    
    idx = 0
    for name, img in results_dict.items():
        ax = axes[idx]
        
        # Display logic
        if len(img.shape) == 3 and img.shape[2] == 3:
            # RGB/BGR image
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(display_img)
        elif len(img.shape) == 3:
            # Multi-channel
            ax.imshow(img)
        else:
            # Grayscale
            ax.imshow(img, cmap='gray')
        
        ax.set_title(f"{name}", fontsize=12, fontweight='bold')
        ax.axis('off')
        idx += 1
    
    # Hide unused subplots
    for idx in range(idx, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"{title_prefix} - Pipeline Analisis Citra", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f"{title_prefix.replace(' ', '_')}_visualization.png")
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualisasi tersimpan: {output_path}")
    return output_path


# =====================================================================
# FUNGSI LAPORAN STATISTIK
# =====================================================================

def generate_statistics_report(original_img, num_labels, stats, centroids):
    """
    Menghasilkan laporan statistik dari analisis citra.
    
    Args:
        original_img (np.ndarray): Citra asli
        num_labels (int): Jumlah label
        stats (np.ndarray): Statistics dari connectedComponentsWithStats
        centroids (np.ndarray): Centroids dari connectedComponentsWithStats
        
    Returns:
        str: Teks laporan
    """
    report = []
    report.append("=" * 70)
    report.append("LAPORAN STATISTIK ANALISIS CITRA".center(70))
    report.append("=" * 70)
    report.append("")
    
    # Informasi citra
    report.append("INFORMASI CITRA")
    report.append("-" * 70)
    report.append(f"Dimensi        : {original_img.shape[0]} x {original_img.shape[1]} px")
    if len(original_img.shape) == 3:
        report.append(f"Channel        : {original_img.shape[2]}")
    else:
        report.append(f"Channel        : Grayscale")
    report.append(f"Total Pixel    : {original_img.shape[0] * original_img.shape[1]} px")
    report.append("")
    
    # Connected Components
    report.append("ANALISIS CONNECTED COMPONENTS")
    report.append("-" * 70)
    report.append(f"Jumlah Label   : {num_labels}")
    report.append(f"Jumlah Objek   : {num_labels - 1}")  # Exclude background
    report.append("")
    
    # Detail setiap komponen
    report.append("DETAIL SETIAP KOMPONEN:")
    report.append("")
    
    # Skip label 0 (background)
    for i in range(1, min(num_labels, 11)):  # Max 10 komponen
        area = stats[i, cv2.CC_STAT_AREA]
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        
        report.append(f"Komponen {i}:")
        report.append(f"  Area (pixel)    : {area}")
        report.append(f"  Bounding Box    : ({left}, {top}) - {width}x{height}")
        report.append(f"  Centroid        : ({cx:.2f}, {cy:.2f})")
        report.append("")
    
    if num_labels > 11:
        report.append(f"... dan {num_labels - 11} komponen lainnya")
        report.append("")
    
    # Summary statistik area
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        report.append("STATISTIK AREA KOMPONEN:")
        report.append("-" * 70)
        report.append(f"Area Minimum   : {np.min(areas)} px")
        report.append(f"Area Maksimum  : {np.max(areas)} px")
        report.append(f"Area Rata-rata : {np.mean(areas):.2f} px")
        report.append(f"Area Median    : {np.median(areas):.2f} px")
        report.append(f"Std Deviasi    : {np.std(areas):.2f} px")
        report.append("")
    
    report.append("=" * 70)
    report.append(f"Tanggal Report : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    return "\n".join(report)


def save_report(report_text, filename="report.txt"):
    """
    Menyimpan laporan statistik ke file teks.
    
    Args:
        report_text (str): Teks laporan
        filename (str): Nama file output
    """
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Laporan tersimpan: {output_path}")


def save_image(img, filename, description=""):
    """
    Menyimpan citra hasil ke file.
    
    Args:
        img (np.ndarray): Citra yang akan disimpan
        filename (str): Nama file output
        description (str): Deskripsi citra
    """
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, img)
    if description:
        print(f"✓ {description}: {output_path}")
    else:
        print(f"✓ Tersimpan: {output_path}")


# =====================================================================
# FUNGSI UTAMA - MAIN PIPELINE
# =====================================================================

def main():
    """
    Main pipeline - eksekusi semua tahap processing.
    """
    print("\n" + "=" * 70)
    print("PIPELINE ANALISIS CITRA - MODUL 3 PEMROSESAN CITRA".center(70))
    print("=" * 70 + "\n")
    
    # 1. LOAD IMAGE
    print("[1/8] LOADING IMAGE")
    image_path = os.path.join(INPUT_DIR, INPUT_IMAGE_NAME)
    original_img = load_image(image_path)
    
    if original_img is None:
        print("❌ Pipeline dihentikan!")
        return
    
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    save_image(original_img, "00_original.jpg", "Citra original")
    
    # 2. PREPROCESSING
    print("\n[2/8] PREPROCESSING (CLAHE + Bilateral Filter)")
    preprocessed = preprocessing(original_img.copy())
    save_image(preprocessed, "01_preprocessed.jpg", "Citra setelah preprocessing")
    
    prep_gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    
    # 3. HSV SEGMENTATION
    print("\n[3/8] HSV SEGMENTATION")
    mask_red, mask_blue, img_hsv = hsv_segmentation(preprocessed)
    save_image(mask_red, "02_hsv_red_mask.jpg", "Mask segmentasi merah")
    save_image(mask_blue, "02_hsv_blue_mask.jpg", "Mask segmentasi biru")
    
    # 4. THRESHOLDING
    print("\n[4/8] THRESHOLDING (Otsu + Adaptive)")
    otsu_binary = otsu_thresholding(prep_gray)
    adaptive_binary = adaptive_thresholding(prep_gray)
    save_image(otsu_binary, "03_otsu_threshold.jpg", "Otsu thresholding")
    save_image(adaptive_binary, "03_adaptive_threshold.jpg", "Adaptive thresholding")
    
    # 5. EDGE DETECTION
    print("\n[5/8] EDGE DETECTION (Canny + Sobel)")
    canny_edges = canny_edge_detection(prep_gray)
    sobel_mag, sobel_dir = sobel_edge_detection(prep_gray)
    save_image(canny_edges, "04_canny_edges.jpg", "Canny edge detection")
    save_image(sobel_mag, "04_sobel_magnitude.jpg", "Sobel magnitude")
    
    # 6. MORFOLOGI
    print("\n[6/8] MORFOLOGI (Erosi, Dilasi, Opening, Closing)")
    eroded = erode(otsu_binary.copy(), iterations=1)
    dilated = dilate(otsu_binary.copy(), iterations=1)
    opened = opening(otsu_binary.copy())
    closed = closing(otsu_binary.copy())
    
    save_image(eroded, "05_erosion.jpg", "Hasil erosion")
    save_image(dilated, "05_dilation.jpg", "Hasil dilation")
    save_image(opened, "05_opening.jpg", "Hasil opening")
    save_image(closed, "05_closing.jpg", "Hasil closing")
    
    # 7. CONNECTED COMPONENTS
    print("\n[7/8] CONNECTED COMPONENTS ANALYSIS")
    num_labels, labels, stats, centroids = connected_components_analysis(closed)
    colored_cc = visualize_connected_components(labels, num_labels)
    save_image(colored_cc.astype(np.uint8), "06_connected_components.jpg", 
               "Connected components visualization")
    
    # 8. FOURIER TRANSFORM
    print("\n[8/8] FOURIER TRANSFORM")
    fft_shift, magnitude_spectrum, phase_spectrum = compute_fft(prep_gray)
    
    # Low-pass filter
    lpf = create_lowpass_filter(prep_gray.shape)
    lpf_result = apply_frequency_filter(fft_shift, lpf)
    mag_lpf = 20 * np.log(np.abs(fft_shift * lpf) + 1)
    save_image(lpf_result, "07_lowpass_filtered.jpg", "Low-pass filter result")
    
    # High-pass filter
    hpf = create_highpass_filter(prep_gray.shape)
    hpf_result = apply_frequency_filter(fft_shift, hpf)
    mag_hpf = 20 * np.log(np.abs(fft_shift * hpf) + 1)
    save_image(hpf_result, "07_highpass_filtered.jpg", "High-pass filter result")
    
    # Magnitude spectrum untuk visualisasi
    mag_spec_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_image(mag_spec_normalized, "07_magnitude_spectrum.jpg", "Magnitude spectrum")
    
    # =====================================================================
    # VISUALISASI LENGKAP
    # =====================================================================
    print("\n[VISUALISASI] Membuat subplot grid...")
    
    viz_dict = {
        "00. Original": original_img,
        "01. CLAHE + Bilateral": preprocessed,
        "02. HSV Red Mask": mask_red,
        "02. HSV Blue Mask": mask_blue,
        "03. Otsu Threshold": otsu_binary,
        "03. Adaptive Threshold": adaptive_binary,
        "04. Canny Edges": canny_edges,
        "04. Sobel Magnitude": sobel_mag,
        "05. Erosion": eroded,
        "05. Dilation": dilated,
        "05. Opening": opened,
        "05. Closing": closed,
        "06. Connected Components": colored_cc.astype(np.uint8),
        "07. Magnitude Spectrum": np.stack([mag_spec_normalized]*3, axis=2),
        "07. Low-Pass Result": lpf_result,
        "07. High-Pass Result": hpf_result,
    }
    
    create_visualization_grid(viz_dict, "Complete_Results")
    
    # =====================================================================
    # GENERATE REPORT
    # =====================================================================
    print("\n[REPORT] Generating statistics report...")
    report = generate_statistics_report(original_img, num_labels, stats, centroids)
    save_report(report)
    
    # Print report ke console
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE SELESAI!".center(70))
    print("=" * 70)
    print(f"\n📁 Semua hasil tersimpan di folder: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nFile output:")
    print("  - 00_original.jpg")
    print("  - 01_preprocessed.jpg")
    print("  - 02_hsv_*_mask.jpg")
    print("  - 03_*_threshold.jpg")
    print("  - 04_*_edges.jpg")
    print("  - 05_*.jpg (morfologi)")
    print("  - 06_connected_components.jpg")
    print("  - 07_*.jpg (Fourier)")
    print("  - Complete_Results_visualization.png")
    print("  - report.txt")
    print("\n")


if __name__ == "__main__":
    main()
