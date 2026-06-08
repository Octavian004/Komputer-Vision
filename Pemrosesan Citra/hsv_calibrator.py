"""
HSV COLOR RANGE DETECTOR & CALIBRATOR
Tool interaktif untuk menemukan range HSV warna spesifik dalam gambar.

Cara Penggunaan:
python hsv_calibrator.py --image input/sample.jpg

Kontrol:
- Drag trackbar untuk adjust range H, S, V
- Mask otomatis update real-time
- Tekan 'S' untuk save range
- Tekan 'ESC' untuk exit
"""

import cv2
import numpy as np
import argparse
import sys

class HSVCalibrator:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        if self.img is None:
            print(f"❌ Error: Tidak bisa baca file '{image_path}'")
            sys.exit(1)
        
        # Resize jika gambar terlalu besar
        height, width = self.img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            self.img = cv2.resize(self.img, (int(width*scale), int(height*scale)))
        
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        # Default HSV range
        self.h_min, self.h_max = 0, 10
        self.s_min, self.s_max = 100, 255
        self.v_min, self.v_max = 100, 255
        
        self.window_name = "HSV Color Range Calibrator"
        cv2.namedWindow(self.window_name)
        
        # Create trackbars
        cv2.createTrackbar("H Min", self.window_name, self.h_min, 180, self.on_h_min_change)
        cv2.createTrackbar("H Max", self.window_name, self.h_max, 180, self.on_h_max_change)
        cv2.createTrackbar("S Min", self.window_name, self.s_min, 255, self.on_s_min_change)
        cv2.createTrackbar("S Max", self.window_name, self.s_max, 255, self.on_s_max_change)
        cv2.createTrackbar("V Min", self.window_name, self.v_min, 255, self.on_v_min_change)
        cv2.createTrackbar("V Max", self.window_name, self.v_max, 255, self.on_v_max_change)
        
    def on_h_min_change(self, val):
        self.h_min = min(val, self.h_max)
        self.update_display()
    
    def on_h_max_change(self, val):
        self.h_max = max(val, self.h_min)
        self.update_display()
    
    def on_s_min_change(self, val):
        self.s_min = min(val, self.s_max)
        self.update_display()
    
    def on_s_max_change(self, val):
        self.s_max = max(val, self.s_min)
        self.update_display()
    
    def on_v_min_change(self, val):
        self.v_min = min(val, self.v_max)
        self.update_display()
    
    def on_v_max_change(self, val):
        self.v_max = max(val, self.v_min)
        self.update_display()
    
    def update_display(self):
        # Create mask
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(self.img_hsv, lower, upper)
        
        # Show both original and masked
        result = np.hstack((self.img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        
        # Add text info
        text_lines = [
            f"H: {self.h_min}-{self.h_max} | S: {self.s_min}-{self.s_max} | V: {self.v_min}-{self.v_max}",
            "Press 'S' to save range | 'ESC' to exit"
        ]
        
        for i, text in enumerate(text_lines):
            cv2.putText(result, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, result)
    
    def run(self):
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('s') or key == ord('S'):
                self.save_range()
        
        cv2.destroyAllWindows()
    
    def save_range(self):
        """Simpan range yang sudah dikalibrasi ke file teks"""
        output_file = "hsv_range_calibrated.txt"
        
        with open(output_file, 'w') as f:
            f.write("# HSV RANGE HASIL KALIBRASI\n")
            f.write("# Copy-paste kode di bawah ke main.py\n\n")
            f.write(f"HSV_LOWER = np.array([{self.h_min}, {self.s_min}, {self.v_min}])\n")
            f.write(f"HSV_UPPER = np.array([{self.h_max}, {self.s_max}, {self.v_max}])\n")
            f.write(f"\nmask = cv2.inRange(img_hsv, HSV_LOWER, HSV_UPPER)\n")
        
        print(f"\n✓ HSV Range tersimpan ke '{output_file}'")
        print(f"\nGunakan di main.py:")
        print(f"  HSV_LOWER = np.array([{self.h_min}, {self.s_min}, {self.v_min}])")
        print(f"  HSV_UPPER = np.array([{self.h_max}, {self.s_max}, {self.v_max}])")


def main():
    parser = argparse.ArgumentParser(
        description="HSV Color Range Calibrator - Interactive tool untuk mencari range warna"
    )
    parser.add_argument('--image', '-i', required=True, 
                       help='Path ke file gambar input')
    
    args = parser.parse_args()
    
    calibrator = HSVCalibrator(args.image)
    calibrator.run()


if __name__ == "__main__":
    main()
