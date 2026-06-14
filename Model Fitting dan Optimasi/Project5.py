# ============================================================
# SMART VISION SYSTEM (IMAGE DATASET VERSION)
# Menggunakan dataset otomatis dari folder /image
# ============================================================

import cv2
import numpy as np
import os

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image")

# ==============================
# LOAD IMAGE (AUTO)
# ==============================
def load_image(filename):
    path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        print(f"ERROR: Gambar {filename} tidak ditemukan di folder image/")
        exit()

    return img

# ==============================
# PREPROCESSING
# ==============================
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# ==============================
# HOUGH LINE (LANE DETECTION)
# ==============================
def detect_lanes(frame, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                            minLineLength=50, maxLineGap=10)

    lane_img = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_img, (x1,y1), (x2,y2), (0,255,0), 2)
    return lane_img

# ==============================
# HOUGH CIRCLE (COIN DETECTION)
# ==============================
def detect_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=100, param2=30,
                               minRadius=10, maxRadius=100)

    out = frame.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(out, (i[0], i[1]), i[2], (0,255,0), 2)
            cv2.circle(out, (i[0], i[1]), 2, (0,0,255), 3)

    return out

# ==============================
# HOMOGRAPHY (DOCUMENT SCAN)
# ==============================
def get_bird_eye(frame):
    h, w = frame.shape[:2]

    src = np.float32([
        [w*0.2, h*0.2],
        [w*0.8, h*0.2],
        [w*0.9, h*0.9],
        [w*0.1, h*0.9]
    ])

    dst = np.float32([
        [0,0],
        [w,0],
        [w,h],
        [0,h]
    ])

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, H, (w,h))
    return warped

# ==============================
# TEMPLATE MATCHING
# ==============================
def template_matching(scene, template):
    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray_scene, gray_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.7)

    h, w = gray_template.shape
    out = scene.copy()

    for pt in zip(*loc[::-1]):
        cv2.rectangle(out, pt, (pt[0]+w, pt[1]+h), (0,255,0), 2)

    return out

# ==============================
# MAIN
# ==============================

def main():
    # Load images dari dataset
    jalan = load_image("jalan.png")
    koin = load_image("koin.png")
    dokumen = load_image("dokumen_asli.jpg")
    target = load_image("target.png")
    template = load_image("template.png")

    # Lane detection
    edges = preprocess(jalan)
    lane_img = detect_lanes(jalan, edges)

    # Circle detection
    circle_img = detect_circles(koin)

    # Homography
    bird = get_bird_eye(dokumen)

    # Template matching
    tm_img = template_matching(target, template)

    # Display
    cv2.imshow("Lane Detection (jalan)", lane_img)
    cv2.imshow("Circle Detection (koin)", circle_img)
    cv2.imshow("Document Warp", bird)
    cv2.imshow("Template Matching", tm_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==============================
# RUN
# ==============================
if __name__ == '__main__':
    main()
