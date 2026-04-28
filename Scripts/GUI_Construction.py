
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import os
from pathlib import Path
import numpy as np
import segmentation_models_pytorch as smp
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

BG = "#0f1117"
PANEL = "#1a1d27"
CARD = "#22263a"
ACCENT = "#00d4aa"
TEXT = "#e8eaf0"
MUTED = "#6b7280"

class ROIselector:

    def __init__(self, root):
        self.root = root
        self.root.title("ROI Selector")
        
        
        self.scale_mode = False
        self.scale_mode   = False
        self.scale_line   = None       
        self.scale_p1     = None       
        self.scale_p2     = None       
        self.px_per_um    = None
        self.seg_method = tk.StringVar(value="all")
        self.image = None
        self.tk_image = None

        self.start_x = None
        self.start_y = None
        self.rect = None

        self.current_roi= None
        self.image_path = None

        container = tk.Frame(root, bg=PANEL)
        container.pack(fill="x")
        tk.Label(container, text="ROI Analyzer",
            bg=PANEL, fg=ACCENT,
            font=("Segoe UI", 14, "bold")).pack(padx=15, pady=15, anchor="w")

        
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Canvas izquierda (imagen)
        self.canvas = tk.Canvas(main, cursor="cross", bg="black")
        self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    
        # Canvas ROI
        self.roi_canvas = tk.Canvas(main, bg="gray")
        self.roi_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Segmented Canvas 
        self.segmented_canvas = tk.Canvas(main, cursor="cross", bg="black")
        self.segmented_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(side="left",fill="x", padx=5, pady=10)

        self.btn = tk.Button(btn_frame, text="Load Image",bg=ACCENT,fg="black",font=("Segoe UI", 10, "bold" ), relief="flat",
                                padx=40, pady=5,cursor="hand2", command=self.load_image)
        self.btn.pack(side="left",pady=10, padx=40, fill="x")

        self.btn_2 = tk.Button(btn_frame, text="save Image",bg=ACCENT,fg="black",font=("Segoe UI", 10, "bold" ), relief="flat",
                                padx=40, pady=5,cursor="hand2", command=self.save_ROI)
        self.btn_2.pack(side="left",pady=10, padx=40, fill="x")


        tk.Label(btn_frame, text="Segmentation", font=("Courier New", 14, "bold"),
                     bg=PANEL, fg=ACCENT).pack(anchor="w", padx=12, pady=(14,2))
        tk.Frame(btn_frame, bg=ACCENT, height=1).pack
        

        ttk.Combobox(btn_frame, textvariable=self.seg_method,
                                   values=["all", "Xylem", "circles"],
                                   state="readonly", width=14,
                                   font=("Courier New", 9)
                                   ).pack(side="left")
        
        
        self.btn_3 = tk.Button(btn_frame, text="Segment",bg=ACCENT,fg="black",font=("Segoe UI", 10, "bold" ), relief="flat",
                                padx=40, pady=5,cursor="hand2", command=self.show_segmented)
        self.btn_3.pack(side="left",pady=10, padx=40, fill="x")
        

        self.btn_scale = tk.Button(btn_frame, text="Set Scale", bg=ACCENT, fg="black",
                           font=("Segoe UI", 10, "bold"), relief="flat",
                           padx=10, pady=5, cursor="hand2",
                           command=self.start_scale_mode)
        self.btn_scale.pack(side="left", expand=True, fill="x", padx=4)
        
        scale_input_frame = tk.Frame(btn_frame, bg=BG)
        scale_input_frame.pack(side="left", padx=4)

        tk.Label(scale_input_frame, text="um:", bg=BG, fg=TEXT,
         font=("Segoe UI", 10)).pack(side="left")

        self.um_entry = tk.Entry(scale_input_frame, width=6,
                         bg=CARD, fg=TEXT, insertbackground=TEXT,
                         font=("Segoe UI", 10), relief="flat")
        self.um_entry.pack(side="left", ipady=4, padx=(2, 0))

        self.scale_label = tk.Label(btn_frame, text="Scale: px/um",
                                    bg=BG, fg=MUTED,
                                    font=("Segoe UI", 9))
        self.scale_label.pack(side="left", padx=10)
    
        
        # Eventos
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.image = img
        self.display_image(img)
        self.image_path = path

    def display_image(self, img):
        self_canvas_w = self.canvas.winfo_width()
        self_canvas_h = self.canvas.winfo_height()
        
        h, w = img.shape[:2]

        scale = min(self_canvas_w / w, self_canvas_h / h, 1)

        new_w  = int(w * scale)
        new_h  = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.display_scale = scale

        self.img_offset_x = (self_canvas_w - new_w) // 2
        self.img_offset_y = (self_canvas_h - new_h) // 2

        pil_img = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")   # limpia antes de redibujar
        self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                  anchor="nw", image=self.tk_image)

    def on_click(self, event):
        if self.scale_mode:
            self.scale_p1 = (event.x, event.y)
            if self.scale_line:
                self.canvas.delete(self.scale_line)
            return
        self.start_x = event.x
        self.start_y = event.y

        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            self.start_x, self.start_y,
            outline="red"
        )
        

    def on_drag(self, event):
        if self.scale_mode and self.scale_p1:
            if self.scale_line:
                self.canvas.delete(self.scale_line)
            self.scale_line = self.canvas.create_line(
                self.scale_p1[0], self.scale_p1[1],
                event.x, event.y,
                fill="yellow", width=2, dash=(4, 2))
            return
      
        self.canvas.coords(
            self.rect,
            self.start_x, self.start_y,
            event.x, event.y
        )

    def on_release(self, event):
        if self.scale_mode and self.scale_p1:
            self.scale_p2 = (event.x, event.y)
            self._compute_scale()   # FIX 3: calcular escala
            return
        
        if self.image is None:
            return       
        end_x, end_y = event.x, event.y
        x1 = int((min(self.start_x, end_x) - self.img_offset_x) / self.display_scale)
        y1 = int((min(self.start_y, end_y) - self.img_offset_y) / self.display_scale)
        x2 = int((max(self.start_x, end_x) - self.img_offset_x) / self.display_scale)
        y2 = int((max(self.start_y, end_y) - self.img_offset_y) / self.display_scale)

        
        h, w = self.image.shape[:2]
        
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        roi = self.image[y1:y2, x1:x2]
        self.current_roi = roi
        self.show_roi(roi)

    def show_roi(self, roi):
        roi_canvas_w = self.roi_canvas.winfo_width()
        roi_canvas_h = self.roi_canvas.winfo_height()
        
        h, w = roi.shape[:2]

        roi_scale = min(roi_canvas_w / w, roi_canvas_h / h, 1)
        new_w = int(w * roi_scale)
        new_h = int(h * roi_scale)
        
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(resized)
        self.tk_roi = ImageTk.PhotoImage(pil_img)
        self.roi_canvas.delete("all")

        cx = roi_canvas_w // 2
        cy = roi_canvas_h // 2
        self.roi_canvas.create_image(cx, cy, anchor="center", image=self.tk_roi)

    def save_ROI(self):
        name = Path(self.image_path).stem
        directory = r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\ROIs"
        os.makedirs(directory, exist_ok=True)
        filepath = f"{name}.tiff"
        filepath = os.path.join(directory, filepath)
        cv2.imwrite(filepath, self.current_roi)
        print(f"image{name} saved")
   
    @staticmethod  
    def build_model():
        return smp.Unet(
            encoder_name    = "resnet34",
            encoder_weights = "imagenet",
            in_channels     = 3,
            classes         = 1,
            activation      = None,
        )
    def segment_all(self, model_path=r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\all_unet.pth", threshold=0.5):    
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = self.build_model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        img = self.current_roi.copy()   
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        

        h, w = img.shape[:2]

        tf  = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
        inp = tf(image=img)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(inp))[0, 0].cpu().numpy()

        # Devuelve al tamaño original
        prob_full = cv2.resize(prob, (w, h))
        mask      = (prob_full > threshold).astype(np.uint8)

        # Overlay: vasos en verde sobre la imagen
        overlay = img.copy()
        overlay[mask == 1] = [0, 220, 120]
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    #    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #    axes[0].imshow(img);         axes[0].set_title("Original");         axes[0].axis("off")
    #    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Predicted mask"); axes[1].axis("off")
    #    axes[2].imshow(blended);     axes[2].set_title("Overlay");          axes[2].axis("off")
    #    plt.tight_layout()
    #    plt.show()

        return blended, mask
    def segment_xylem(self, model_path=r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\xylem_unet.pth", threshold=0.5):    
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = self.build_model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        img = self.current_roi.copy()   
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        

        h, w = img.shape[:2]

        tf  = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
        inp = tf(image=img)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(inp))[0, 0].cpu().numpy()

        # Devuelve al tamaño original
        prob_full = cv2.resize(prob, (w, h))
        mask      = (prob_full > threshold).astype(np.uint8)

        # Overlay: vasos en verde sobre la imagen
        overlay = img.copy()
        overlay[mask == 1] = [0, 220, 120]
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    #    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #    axes[0].imshow(img);         axes[0].set_title("Original");         axes[0].axis("off")
    #    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Predicted mask"); axes[1].axis("off")
    #    axes[2].imshow(blended);     axes[2].set_title("Overlay");          axes[2].axis("off")
    #    plt.tight_layout()
    #    plt.show()

        return blended, mask
    
    def segment_circles(self, model_path=r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\circles_unet.pth", threshold=0.5):    
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = self.build_model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        img = self.current_roi.copy()   
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        

        h, w = img.shape[:2]

        tf  = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
        inp = tf(image=img)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(inp))[0, 0].cpu().numpy()

        # Devuelve al tamaño original
        prob_full = cv2.resize(prob, (w, h))
        mask      = (prob_full > threshold).astype(np.uint8)

        # Overlay: vasos en verde sobre la imagen
        overlay = img.copy()
        overlay[mask == 1] = [0, 220, 120]
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    #    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #    axes[0].imshow(img);         axes[0].set_title("Original");         axes[0].axis("off")
    #    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Predicted mask"); axes[1].axis("off")
    #    axes[2].imshow(blended);     axes[2].set_title("Overlay");          axes[2].axis("off")
    #    plt.tight_layout()
    #    plt.show()

        return blended, mask
    
    def show_segmented(self):

        if self.current_roi is None:
            messagebox.showwarning("No ROI", "Select a ROI first")
            return

        try:
            method = self.seg_method.get()
            if method == "all":
                blended, mask = self.segment_all()
            elif method == "Xylem":
                 blended, mask = self.segment_xylem()
            elif method == "circles": 
                 blended, mask = self.segment_circles()
            #elif method == "internal":
             #   blended, mask = self.segment_internal
            else:
                return
            self.root.update_idletasks()

            segmented_canvas_w = self.segmented_canvas.winfo_width()
            segmented_canvas_h = self.segmented_canvas.winfo_height()
            
            h, w = blended.shape[:2]
            segmented_scale = min(segmented_canvas_w / w, segmented_canvas_h / h, 1)
            new_w = int(w * segmented_scale)
            new_h = int(h * segmented_scale)
            
            resized = cv2.resize(blended, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if resized.dtype != np.uint8:
                resized = resized.astype(np.uint8)
            pil_img = Image.fromarray(resized)
            self.tk_segmented = ImageTk.PhotoImage(pil_img)
            self.segmented_canvas.delete("all")

            cx = segmented_canvas_w // 2
            cy = segmented_canvas_h // 2
            self.segmented_canvas.create_image(cx, cy, anchor="center", image=self.tk_segmented)
            
            area_um2 = self.calculate_area(mask)
            if area_um2 is not None:
                self.scale_label.config(
                    text=f"Scale: {self.px_per_um:.2f} px/µm  |  Area: {area_um2:,.1f} µm²",
                    fg=ACCENT)
                print(f"Segmented area: {area_um2:,.2f} µm²")

        except Exception as e:
            self.root.after(0, lambda: (
                self.progress.stop(),
                self.status_text.set(f"Error: {e}"),
                messagebox.showerror("Analysis error", str(e))
            )) 
    def start_scale_mode(self):
        messagebox.showinfo("Scale value","Please enter µm value before drawing scale")
        assert self.image is not None, f"Loa d an image first"
        self.scale_mode = True
        self.scale_p1   = None
        self.scale_p2   = None
        if self.scale_line:
            self.canvas.delete(self.scale_line)
        self.btn_scale.config(text="Drawing... (click & drag)", bg="#ff6b6b")
        self.canvas.config(cursor="tcross")

    def _compute_scale(self):
        um_text = self.um_entry.get().strip()
        if not um_text:
            messagebox.showwarning("Missing value", "Enter µm value in the entry field")
            self._reset_scale_mode()
            return
        try:
            um = float(um_text)
        except ValueError:
            messagebox.showerror("Invalid value", "µm must be a number")
            self._reset_scale_mode()
            return
        dx = self.scale_p2[0] - self.scale_p1[0]
        dy = self.scale_p2[1] - self.scale_p1[1]
        px_canvas = np.sqrt(dx**2 + dy**2)
        px_real   = px_canvas / self.display_scale
        self.px_per_um = px_real / um

        self.scale_label.config(text=f"Scale: {self.px_per_um:.2f} px/µm", fg=ACCENT)

        # Dibuja línea definitiva + etiqueta
        self.canvas.delete(self.scale_line)
        self.scale_line = self.canvas.create_line(
            self.scale_p1[0], self.scale_p1[1],
            self.scale_p2[0], self.scale_p2[1],
            fill="green", width=2)
        mx = (self.scale_p1[0] + self.scale_p2[0]) // 2
        my = (self.scale_p1[1] + self.scale_p2[1]) // 2 - 10
        self.canvas.create_text(mx, my, text=f"{um} µm",
                                 fill="yellow", font=("Segoe UI", 9, "bold"))
        self._reset_scale_mode()
        print(f"Scale set: {self.px_per_um:.4f} px/µm")
        
    def _reset_scale_mode(self):
        self.scale_mode = False
        self.btn_scale.config(text="Set Scale", bg=ACCENT)
        self.canvas.config(cursor="cross")
   
    def calculate_area(self, mask):
        if self.px_per_um is None:
            messagebox.showwarning("No scale", "Set the scale first (Set Scale button)")
            return None

        pixel_count   = int(np.sum(mask))         
        area_um2      = pixel_count / (self.px_per_um ** 2)
        return area_um2


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x660")
    app = ROIselector(root)
    root.mainloop()
