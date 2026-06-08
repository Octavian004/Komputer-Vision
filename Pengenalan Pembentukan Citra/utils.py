"""Utility functions for document scanner project.

Fungsi ini fokus pada:
- mendeteksi kontur dokumen
- mengoreksi perspektif
- auto-gamma
- undistort/koreksi lensa (jika kalibrasi tersedia)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def load_calibration(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load camera calibration (camera matrix + distortion) dari JSON."""
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data.get("camera_matrix", []), dtype=np.float64)
    dist = np.array(data.get("dist_coeff", []), dtype=np.float64)

    if K.size != 9 or dist.size == 0:
        return None

    return K.reshape(3, 3), dist.ravel()


def undistort_image(img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """Koreksi distorsi lensa menggunakan parameter kalibrasi."""
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w2, h2 = roi
    if w2 > 0 and h2 > 0:
        undistorted = undistorted[y : y + h2, x : x + w2]
    return undistorted


def auto_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Adjust gamma using lookup table. gamma=1.0 berarti no change."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def estimate_gamma(img: np.ndarray, target_mean: float = 128.0) -> float:
    """Estimate gamma value untuk mendekatkan kecerahan ke target_mean."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    if mean_val <= 0:
        return 1.0
    return np.log(target_mean / 255.0) / np.log(mean_val / 255.0)


def find_document_contour(img: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """Cari kontur dokumen terbesar dengan 4 titik."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Morphological closing untuk menutup gap pada tepi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            if cv2.contourArea(pts) > 10000:
                return order_points(pts)

    if debug:
        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """Urutkan 4 titik (top-left, top-right, bottom-right, bottom-left)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_to_a4(img: np.ndarray, pts: np.ndarray, dpi: int = 300) -> np.ndarray:
    """Warp dokumen ke ukuran A4 (portrait) dengan dpi tertentu."""
    # A4 ukuran dalam inci: 8.27 x 11.69
    a4_w = int(round(8.27 * dpi))
    a4_h = int(round(11.69 * dpi))

    dst = np.array([[0, 0], [a4_w - 1, 0], [a4_w - 1, a4_h - 1], [0, a4_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (a4_w, a4_h), flags=cv2.INTER_CUBIC)
    return warped


def make_thumbnail(img: np.ndarray, max_dim: int = 512) -> np.ndarray:
    """Buat thumbnail sebanding dengan dimensi maksimum tertentu."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
