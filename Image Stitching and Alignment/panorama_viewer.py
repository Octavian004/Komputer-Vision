"""
panorama_viewer.py
------------------
GUI sederhana berbasis Tkinter untuk melihat panorama dengan interaksi pan dan zoom.

Fitur:
- Load panorama yang dihasilkan
- Click and drag untuk panning
- Scroll wheel untuk zooming
"""

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


class PanoramaViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Panorama Viewer (Pan & Tilt)")
        self.master.geometry("1000x600")
        
        self.image_path = None
        self.cv_image = None
        self.pil_image = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Frame atas untuk kontrol
        top_frame = tk.Frame(self.master, bg="#333", height=50)
        top_frame.pack(fill=tk.X, side=tk.TOP)
        
        btn_load = tk.Button(top_frame, text="Load Panorama", command=self.load_image, bg="#555", fg="white")
        btn_load.pack(side=tk.LEFT, padx=10, pady=10)
        
        btn_reset = tk.Button(top_frame, text="Reset View", command=self.reset_view, bg="#555", fg="white")
        btn_reset.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.info_label = tk.Label(top_frame, text="Tidak ada gambar. Silakan load panorama.", bg="#333", fg="white")
        self.info_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Frame utama untuk gambar
        self.canvas = tk.Canvas(self.master, bg="#111", cursor="fleur")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Events
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux
        
        self.master.bind("<Configure>", self.on_resize)
        
    def load_image(self, path=None):
        if not path:
            path = filedialog.askopenfilename(
                title="Pilih Gambar Panorama",
                filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")),
                initialdir=os.path.join(os.getcwd(), "output")
            )
            
        if path and os.path.exists(path):
            self.image_path = path
            # Baca dengan cv2 dan konversi ke RGB
            img = cv2.imread(path)
            if img is not None:
                self.cv_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.reset_view()
            else:
                messagebox.showerror("Error", "Gagal membaca gambar.")
                
    def reset_view(self):
        if self.cv_image is not None:
            self.pan_x = 0
            self.pan_y = 0
            
            # Hitung zoom fit
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            
            if cw > 10 and ch > 10:
                h, w = self.cv_image.shape[:2]
                scale_w = cw / w
                scale_h = ch / h
                self.zoom_factor = min(scale_w, scale_h) * 0.95 # Leave a bit of margin
            else:
                self.zoom_factor = 1.0
                
            self.update_view()
            
    def update_view(self):
        if self.cv_image is None:
            return
            
        h, w = self.cv_image.shape[:2]
        
        # Hitung ukuran gambar saat ini
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        
        if new_w <= 0 or new_h <= 0:
            return
            
        # Batasi panning agar gambar tidak hilang dari layar
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        # Resize image
        img_resized = cv2.resize(self.cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.pil_image = ImageTk.PhotoImage(image=Image.fromarray(img_resized))
        
        # Hapus tampilan lama dan render yang baru
        self.canvas.delete("all")
        
        # Letakkan di tengah ditambah panning
        center_x = cw // 2 + self.pan_x
        center_y = ch // 2 + self.pan_y
        
        self.canvas.create_image(center_x, center_y, image=self.pil_image, anchor=tk.CENTER)
        
        # Update info
        self.info_label.config(text=f"Zoom: {self.zoom_factor*100:.0f}% | Pan: ({self.pan_x}, {self.pan_y})")

    def on_drag_start(self, event):
        self.is_dragging = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag_motion(self, event):
        if self.is_dragging:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            
            self.pan_x += dx
            self.pan_y += dy
            
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            
            self.update_view()

    def on_drag_release(self, event):
        self.is_dragging = False

    def on_mouse_wheel(self, event):
        if self.cv_image is None:
            return
            
        # Tentukan arah zoom
        if event.num == 5 or event.delta < 0:
            zoom_in = False
        else:
            zoom_in = True
            
        # Faktor zoom per step
        zoom_step = 1.15
        if zoom_in:
            self.zoom_factor *= zoom_step
        else:
            self.zoom_factor /= zoom_step
            
        self.update_view()

    def on_resize(self, event):
        # Update hanya kalau ukurannya berubah drastis
        if getattr(self, '_last_w', 0) != event.width or getattr(self, '_last_h', 0) != event.height:
            self._last_w = event.width
            self._last_h = event.height
            if not self.image_path:
                return
            # Tunggu rendering UI selesai
            self.master.after(100, self.update_view)


if __name__ == "__main__":
    root = tk.Tk()
    app = PanoramaViewer(root)
    
    # Auto load jika ada dari output pipeline kita
    default_path = os.path.join(os.getcwd(), "output", "panorama_room1.jpg")
    if os.path.exists(default_path):
        root.after(500, lambda: app.load_image(default_path))
        
    root.mainloop()

"""
Fix Tkinter init error due to double underscores issue.
"""
