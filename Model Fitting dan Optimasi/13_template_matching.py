"""
==========================================================================
PERCOBAAN 13: TEMPLATE MATCHING
==========================================================================
Template matching mencari lokasi template kecil di dalam gambar besar
dengan menggeser template dan menghitung skor kecocokan di setiap posisi.

Fungsi utama:
- cv2.matchTemplate()  : hitung similarity map antara gambar dan template
- cv2.minMaxLoc()      : cari lokasi nilai min/max di result map
- cv2.TM_CCOEFF_NORMED : normalized cross-correlation (skor tinggi = cocok)
- cv2.TM_SQDIFF_NORMED : normalized squared difference (skor rendah = cocok)
- cv2.TM_CCORR_NORMED  : normalized cross-correlation coefficient
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 13: TEMPLATE MATCHING")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Memuat gambar dan template
# ============================================================
print("\n--- 1. Memuat Gambar dan Template ---")

img_path = os.path.join(IMAGE_DIR, "target.png")
tmpl_path = os.path.join(IMAGE_DIR, "template.png")

if not os.path.exists(img_path):
    print("[ERROR] target.png tidak ditemukan. Jalankan download_image.py!"); exit()

if not os.path.exists(tmpl_path):
    print("[ERROR] template.png tidak ditemukan. Jalankan download_image.py!"); exit()

img = cv2.imread(img_path)
template = cv2.imread(tmpl_path)
h, w = template.shape[:2]
print(f"  Gambar: {img.shape}")
print(f"  Template: {template.shape}")

# ============================================================
# 2. Template matching dengan berbagai metode
# ============================================================
print("\n--- 2. Template Matching - Berbagai Metode ---")

# Daftar metode template matching yang tersedia di OpenCV
methods = {
    'TM_CCOEFF': cv2.TM_CCOEFF,
    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
    'TM_CCORR': cv2.TM_CCORR,
    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    'TM_SQDIFF': cv2.TM_SQDIFF,
    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED,
}

results = {}

for name, method in methods.items():
    # cv2.matchTemplate menggeser template di seluruh gambar
    # Hasil: matriks (H-h+1) x (W-w+1) berisi skor kecocokan
    result = cv2.matchTemplate(img, template, method)
    
    # cv2.minMaxLoc: cari lokasi nilai minimum dan maksimum
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Untuk SQDIFF, lokasi minimum = paling cocok
    # Untuk CCOEFF/CCORR, lokasi maksimum = paling cocok
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        best_loc = min_loc
        best_val = min_val
    else:
        best_loc = max_loc
        best_val = max_val
    
    results[name] = (result, best_loc, best_val)
    print(f"  {name}: best_loc={best_loc}, best_val={best_val:.4f}")

# ============================================================
# 3. Visualisasi result map
# ============================================================
print("\n--- 3. Visualisasi Result Map ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for ax, (name, (result, best_loc, best_val)) in zip(axes.flat, results.items()):
    # Tampilkan result map sebagai heatmap
    ax.imshow(result, cmap='hot')
    ax.set_title(f"{name}\nbest={best_val:.4f}")
    ax.plot(best_loc[0], best_loc[1], 'rx', markersize=15, markeredgewidth=3)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "13_result_maps.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# ============================================================
# 4. Gambar kotak di lokasi terbaik
# ============================================================
print("\n--- 4. Menandai Lokasi Terbaik ---")

img_marked = img.copy()

# Gunakan TM_CCOEFF_NORMED (paling umum digunakan)
result_normed = results['TM_CCOEFF_NORMED'][0]
best_loc = results['TM_CCOEFF_NORMED'][1]

# Gambar kotak di lokasi terbaik
top_left = best_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_marked, top_left, bottom_right, (0, 255, 0), 3)
cv2.putText(img_marked, "Match!", (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite(os.path.join(OUTPUT_DIR, "13_single_match.png"), img_marked)
print(f"  Lokasi match: {top_left}")

# ============================================================
# 5. Deteksi multiple matches (multi-object)
# ============================================================
print("\n--- 5. Multi-Object Template Matching ---")

img_multi = img.copy()

# Tentukan threshold untuk memfilter skor tinggi
threshold = 0.8
result_normed = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# np.where: cari semua lokasi dengan skor >= threshold
locations = np.where(result_normed >= threshold)
matches = list(zip(*locations[::-1]))  # format: [(x, y), ...]
print(f"  Threshold: {threshold}")
print(f"  Lokasi dengan skor >= threshold: {len(matches)}")

# Gambar kotak di semua lokasi match
for pt in matches:
    cv2.rectangle(img_multi, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

cv2.imwrite(os.path.join(OUTPUT_DIR, "13_multi_match_raw.png"), img_multi)

# ============================================================
# 6. Non-Maximum Suppression (NMS)
# ============================================================
print("\n--- 6. Non-Maximum Suppression ---")

def non_max_suppression(matches, scores, w, h, overlap_thresh=0.3):
    """
    Non-maximum suppression untuk menghilangkan deteksi yang tumpang tindih.
    """
    if len(matches) == 0:
        return []
    
    # Konversi ke format kotak [x1, y1, x2, y2]
    boxes = np.array([[x, y, x+w, y+h] for (x, y) in matches], dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Urutkan berdasarkan skor (descending)
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        # Ambil box dengan skor tertinggi
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Hitung IoU dengan box lain
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        intersection = inter_w * inter_h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_other = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                     (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_other - intersection
        
        iou = intersection / (union + 1e-6)
        
        # Hapus box dengan IoU tinggi
        remaining = np.where(iou <= overlap_thresh)[0]
        order = order[remaining + 1]
    
    return keep

# Dapatkan skor untuk setiap match
match_scores = [result_normed[y, x] for (x, y) in matches]

# Terapkan NMS
keep_idx = non_max_suppression(matches, match_scores, w, h, overlap_thresh=0.3)
final_matches = [matches[i] for i in keep_idx]

print(f"  Sebelum NMS: {len(matches)} deteksi")
print(f"  Setelah NMS: {len(final_matches)} deteksi")

# Gambar hasil setelah NMS
img_nms = img.copy()
for pt in final_matches:
    cv2.rectangle(img_nms, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
    score = result_normed[pt[1], pt[0]]
    cv2.putText(img_nms, f"{score:.2f}", (pt[0], pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite(os.path.join(OUTPUT_DIR, "13_multi_match_nms.png"), img_nms)

# ============================================================
# 7. Template matching pada skala berbeda
# ============================================================
print("\n--- 7. Multi-Scale Template Matching ---")

best_scale_val = -1
best_scale_loc = None
best_scale = 1.0

# Coba template matching pada berbagai skala gambar
scales = np.linspace(0.5, 2.0, 16)

for scale in scales:
    # Resize gambar target
    resized = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Pastikan template tidak lebih besar dari gambar
    if resized.shape[0] < h or resized.shape[1] < w:
        continue
    
    result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val > best_scale_val:
        best_scale_val = max_val
        best_scale_loc = max_loc
        best_scale = scale

print(f"  Skala terbaik: {best_scale:.2f}")
print(f"  Skor terbaik: {best_scale_val:.4f}")
print(f"  Lokasi (pada skala tsb): {best_scale_loc}")

# ============================================================
# 8. Visualisasi gabungan
# ============================================================
print("\n--- 8. Visualisasi Gabungan ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Gambar Asli")

axes[0, 1].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"Template ({w}x{h})")

axes[0, 2].imshow(result_normed, cmap='hot')
axes[0, 2].set_title("Result Map (CCOEFF_NORMED)")
axes[0, 2].colorbar = plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2])

axes[1, 0].imshow(cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Single Best Match")

axes[1, 1].imshow(cv2.cvtColor(img_multi, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"All Matches (>= {threshold})")

axes[1, 2].imshow(cv2.cvtColor(img_nms, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f"After NMS ({len(final_matches)} deteksi)")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "13_template_matching_all.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Disimpan: {output_path}")

print("\n" + "=" * 60)
print("PERCOBAAN 13 SELESAI")
print("=" * 60)
