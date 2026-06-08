"""
REAL-TIME PIPELINE ANALISIS CITRA - WEBCAM VERSION
Praktikum Komputer Vision - Modul 3 Pemrosesan Citra

Deskripsi: Pipeline real-time yang berjalan dari webcam dengan optimisasi
untuk target minimal 10 FPS menggunakan cv2.VideoCapture(0).

Kontrol:
- 'p': Toggle preprocessing
- 's': Toggle HSV segmentation
- 'e': Toggle edge detection
- 'm': Toggle morphology
- 'f': Toggle FFT
- 'r': Reset ke original
- 'q': Quit
- 'c': Capture frame ke output/

Author: Praktikum CV
Date: 2026
"""

import cv2
import numpy as np
from datetime import datetime
import os
import time
from collections import deque
from threading import Thread, Lock

# =====================================================================
# KONFIGURASI PARAMETER - REAL-TIME OPTIMIZED
# =====================================================================

# Video Capture Settings
CAMERA_ID = 0                          # Webcam ID (0 = default)
CAPTURE_WIDTH = 640                    # Reduced untuk speed
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30                       # Actual FPS dari camera

# Processing Resolution (lebih kecil untuk speed)
PROCESS_WIDTH = 320                    # Processing pada resolusi lebih rendah
PROCESS_HEIGHT = 240

# Display Resolution
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# FPS Target
TARGET_FPS = 10                        # Minimal 10 FPS

# Preprocessing (Reduced complexity)
CLAHE_CLIP_LIMIT = 1.5                # Reduced dari 2.0
CLAHE_TILE_SIZE = (4, 4)              # Smaller untuk speed
BILATERAL_D = 5                        # Reduced dari 9
BILATERAL_SIGMA_COLOR = 50            # Reduced dari 75
BILATERAL_SIGMA_SPACE = 50            # Reduced dari 75

# HSV Segmentation
HSV_RED_LOWER1 = np.array([0, 80, 80])
HSV_RED_UPPER1 = np.array([10, 255, 255])
HSV_RED_LOWER2 = np.array([170, 80, 80])
HSV_RED_UPPER2 = np.array([180, 255, 255])
HSV_BLUE_LOWER = np.array([100, 80, 80])
HSV_BLUE_UPPER = np.array([130, 255, 255])

# Edge Detection (Faster)
CANNY_THRESHOLD1 = 80                 # Reduced untuk speed
CANNY_THRESHOLD2 = 150

# Morphology
MORPH_KERNEL_SIZE = (3, 3)            # Smaller kernel
MORPH_KERNEL_TYPE = cv2.MORPH_ELLIPSE

# Connected Components
CONNECTIVITY = 8

# FFT (Process less frequently)
FFT_PROCESS_INTERVAL = 5              # Process setiap 5 frames
FILTER_D = 20

# Output
OUTPUT_DIR = "output"
CAPTURE_DIR = os.path.join(OUTPUT_DIR, "webcam_captures")

# =====================================================================
# PERFORMANCE MONITORING
# =====================================================================

class FPSCounter:
    """FPS counter untuk monitoring real-time performance."""
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def get_fps(self):
        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0


# =====================================================================
# FAST PREPROCESSING FUNCTIONS
# =====================================================================

def fast_preprocessing(img):
    """
    Fast preprocessing untuk real-time.
    Removed expensive operations, kept only essentials.
    """
    # Resize untuk speed
    img = cv2.resize(img, (PROCESS_WIDTH, PROCESS_HEIGHT))
    
    # Fast bilateral filter (reduced parameters)
    img = cv2.bilateralFilter(img, BILATERAL_D, 
                              BILATERAL_SIGMA_COLOR, 
                              BILATERAL_SIGMA_SPACE)
    
    # Fast CLAHE (simpler version)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, 
                            tileGridSize=CLAHE_TILE_SIZE)
    l_clahe = clahe.apply(l_channel)
    lab = cv2.merge([l_clahe, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img


def fast_hsv_segmentation(img):
    """Fast HSV segmentation untuk real-time."""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Merah (2 range)
    mask_red1 = cv2.inRange(img_hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
    mask_red2 = cv2.inRange(img_hsv, HSV_RED_LOWER2, HSV_RED_UPPER2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Biru
    mask_blue = cv2.inRange(img_hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    
    return mask_red, mask_blue


def fast_edge_detection(img):
    """Fast edge detection - hanya Canny (lebih cepat dari Sobel)."""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    edges = cv2.Canny(img_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    return edges


def fast_morphology(img, kernel_size=MORPH_KERNEL_SIZE):
    """Fast morphology operations."""
    kernel = cv2.getStructuringElement(MORPH_KERNEL_TYPE, kernel_size)
    
    # Combine opening + closing in one step
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed


def fast_connected_components(img):
    """Fast connected components analysis."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, CONNECTIVITY, cv2.CV_32S
    )
    return num_labels, labels, stats, centroids


def fast_fft(img):
    """Fast FFT with magnitude spectrum."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    
    # Normalize untuk display
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    magnitude = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
    
    return magnitude


# =====================================================================
# VISUALIZATION UTILITIES
# =====================================================================

def add_text_overlay(img, text, position=(10, 30), color=(0, 255, 0), fontsize=0.6):
    """Add text overlay ke image."""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                fontsize, color, 2)
    return img


def create_overlay_view(original, processed, title=""):
    """Create split-screen view (original vs processed)."""
    h, w = original.shape[:2]
    
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Resize processed untuk match
    processed = cv2.resize(processed, (w, h))
    
    # Horizontal split
    combined = np.hstack([original, processed])
    
    # Add title
    if title:
        combined = add_text_overlay(combined, title, (10, 25), (0, 255, 0), 0.7)
    
    return combined


def visualize_multiple(images_dict):
    """
    Create grid visualization dari multiple images.
    
    Args:
        images_dict: {title: image} dictionary
    
    Returns:
        Combined visualization
    """
    num_images = len(images_dict)
    if num_images == 0:
        return None
    
    # Determine grid size
    cols = 2
    rows = (num_images + cols - 1) // cols
    
    tile_h = PROCESS_HEIGHT
    tile_w = PROCESS_WIDTH
    
    grid = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
    
    for idx, (title, img) in enumerate(images_dict.items()):
        row = idx // cols
        col = idx % cols
        
        # Convert to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Resize
        img = cv2.resize(img, (tile_w, tile_h))
        
        # Place in grid
        y_start = row * tile_h
        x_start = col * tile_w
        grid[y_start:y_start+tile_h, x_start:x_start+tile_w] = img
        
        # Add title
        cv2.putText(grid, title, (x_start + 5, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return grid


# =====================================================================
# REAL-TIME PIPELINE
# =====================================================================

class RealtimePipeline:
    """Real-time image processing pipeline."""
    
    def __init__(self):
        """Initialize pipeline dan camera."""
        self.cap = cv2.VideoCapture(CAMERA_ID)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError("❌ Error: Tidak bisa buka webcam!")
        
        # FPS counter
        self.fps_counter = FPSCounter()
        
        # Processing flags
        self.process_preprocessing = True
        self.process_segmentation = False
        self.process_edges = False
        self.process_morphology = False
        self.process_fft = False
        
        # For FFT processing frequency control
        self.frame_count = 0
        
        # Lock untuk thread safety
        self.lock = Lock()
        
        # Create output directories
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        
        print("✅ Pipeline initialized successfully")
        print(f"   Camera resolution: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
        print(f"   Processing resolution: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
        print(f"   Target FPS: {TARGET_FPS}")
    
    def process_frame(self, frame):
        """Process single frame through pipeline."""
        # Store original for display
        original = frame.copy()
        original = cv2.resize(original, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        # [1] PREPROCESSING
        if self.process_preprocessing:
            processed = fast_preprocessing(frame)
        else:
            processed = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        results = {
            "01. Original": original,
            "02. Preprocessed": processed
        }
        
        # [2] HSV SEGMENTATION
        if self.process_segmentation:
            mask_red, mask_blue = fast_hsv_segmentation(processed)
            results["03. Red Mask"] = mask_red
            results["04. Blue Mask"] = mask_blue
        
        # [3] EDGE DETECTION
        if self.process_edges:
            edges = fast_edge_detection(processed)
            results["05. Edges"] = edges
        
        # [4] MORPHOLOGY
        if self.process_morphology:
            if self.process_edges:
                morph = fast_morphology(edges)
                results["06. Morphology"] = morph
            else:
                # Morphology on grayscale
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
                morph = fast_morphology(binary)
                results["06. Morphology"] = morph
        
        # [5] FFT (Process less frequently)
        if self.process_fft and self.frame_count % FFT_PROCESS_INTERVAL == 0:
            fft_result = fast_fft(processed)
            results["07. FFT Spectrum"] = fft_result
        
        self.frame_count += 1
        
        return original, results
    
    def run(self):
        """Main loop - run real-time pipeline."""
        print("\n" + "=" * 70)
        print("REAL-TIME PIPELINE - WEBCAM VERSION".center(70))
        print("=" * 70)
        print("\nContol:")
        print("  'p' - Toggle preprocessing")
        print("  's' - Toggle segmentation")
        print("  'e' - Toggle edge detection")
        print("  'm' - Toggle morphology")
        print("  'f' - Toggle FFT")
        print("  'r' - Reset to original")
        print("  'c' - Capture frame")
        print("  'q' - Quit\n")
        print("=" * 70 + "\n")
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error: Failed to read frame")
                break
            
            # Process frame
            original, results = self.process_frame(frame)
            
            # Create visualization
            viz = visualize_multiple(results)
            
            # Resize untuk display
            display = cv2.resize(viz, (DISPLAY_WIDTH * 2, DISPLAY_HEIGHT * 2))
            
            # Add FPS counter
            self.fps_counter.update()
            fps = self.fps_counter.get_fps()
            fps_text = f"FPS: {fps:.1f} (Target: {TARGET_FPS})"
            
            color = (0, 255, 0) if fps >= TARGET_FPS else (0, 165, 255)
            cv2.putText(display, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Info overlay
            overlay_text = []
            overlay_text.append(f"Preproc: {'ON' if self.process_preprocessing else 'OFF'}")
            overlay_text.append(f"Segm: {'ON' if self.process_segmentation else 'OFF'}")
            overlay_text.append(f"Edges: {'ON' if self.process_edges else 'OFF'}")
            overlay_text.append(f"Morph: {'ON' if self.process_morphology else 'OFF'}")
            overlay_text.append(f"FFT: {'ON' if self.process_fft else 'OFF'}")
            
            y_pos = 70
            for text in overlay_text:
                cv2.putText(display, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_pos += 25
            
            # Display
            cv2.imshow("Real-Time Pipeline - Press 'h' for help", display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✓ Exiting...")
                break
            elif key == ord('p'):
                self.process_preprocessing = not self.process_preprocessing
                status = "ON" if self.process_preprocessing else "OFF"
                print(f"✓ Preprocessing: {status}")
            elif key == ord('s'):
                self.process_segmentation = not self.process_segmentation
                status = "ON" if self.process_segmentation else "OFF"
                print(f"✓ Segmentation: {status}")
            elif key == ord('e'):
                self.process_edges = not self.process_edges
                status = "ON" if self.process_edges else "OFF"
                print(f"✓ Edge detection: {status}")
            elif key == ord('m'):
                self.process_morphology = not self.process_morphology
                status = "ON" if self.process_morphology else "OFF"
                print(f"✓ Morphology: {status}")
            elif key == ord('f'):
                self.process_fft = not self.process_fft
                status = "ON" if self.process_fft else "OFF"
                print(f"✓ FFT: {status}")
            elif key == ord('r'):
                self.process_preprocessing = True
                self.process_segmentation = False
                self.process_edges = False
                self.process_morphology = False
                self.process_fft = False
                print("✓ Reset to original")
            elif key == ord('c'):
                self.capture_frame(frame, results)
            elif key == ord('h'):
                self.print_help()
            
            # Frame rate control
            elapsed = time.time() - start_time
            frame_time = 1.0 / TARGET_FPS
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        self.cleanup()
    
    def capture_frame(self, frame, results):
        """Capture current frame ke file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"capture_{timestamp}_original.jpg"), frame)
        
        # Save results
        for title, img in results.items():
            safe_title = title.replace(" ", "_").replace(".", "")
            cv2.imwrite(
                os.path.join(CAPTURE_DIR, f"capture_{timestamp}_{safe_title}.jpg"),
                img
            )
        
        print(f"✓ Frame captured at: {CAPTURE_DIR}/")
    
    def print_help(self):
        """Print help information."""
        print("\n" + "=" * 50)
        print("CONTROL KEYS".center(50))
        print("=" * 50)
        print("'p' - Toggle preprocessing")
        print("'s' - Toggle segmentation")
        print("'e' - Toggle edge detection")
        print("'m' - Toggle morphology")
        print("'f' - Toggle FFT")
        print("'r' - Reset to original")
        print("'c' - Capture frame")
        print("'h' - Show this help")
        print("'q' - Quit")
        print("=" * 50 + "\n")
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    """Main entry point."""
    try:
        pipeline = RealtimePipeline()
        pipeline.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
