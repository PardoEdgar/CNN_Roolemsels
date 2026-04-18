
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

class ROIselector:

    def __init__(self, root):
        self.root = root
        self.root.title("ROI Selector")
        container = tk.Frame(root)
        container.pack(fill="both", expand=True)

        # Canvas izquierda (imagen)
        self.canvas = tk.Canvas(container, cursor="cross", bg="black")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Canvas i (ROI)
        self.roi_canvas = tk.Canvas(container, bg="gray")
        self.roi_canvas.pack(side="right", fill="both", expand=True)


        self.btn = tk.Button(root, text="Load Image", command=self.load_image)
        self.btn.pack(fill="x", padx=12, pady=4)

        self.image = None
        self.tk_image = None

        self.start_x = None
        self.start_y = None
        self.rect = None

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

    def display_image(self, img):
        self_canvas_w = self.canvas.winfo_width()
        self_canvas_h = self.canvas.winfo_height()
        
        h, w = img.shape[:2]

        scale = min(self_canvas_w / w, self_canvas_h / h, 1)

        resized = cv2.resize(img, (self_canvas_w, self_canvas_h), interpolation=cv2.INTER_AREA)

        self.display_scale = scale 

        pil_img = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        cx = self_canvas_w // 2
        cy = self_canvas_h // 2
        self.canvas.create_image(cx, cy, anchor="center", image=self.tk_image)
        
    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y

        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            self.start_x, self.start_y,
            outline="red"
        )

    def on_drag(self, event):
        self.canvas.coords(
            self.rect,
            self.start_x, self.start_y,
            event.x, event.y
        )

    def on_release(self, event):
        end_x, end_y = event.x, event.y
        x1 = int(min(self.start_x, end_x) / self.display_scale)
        y1 = int(min(self.start_y, end_y) / self.display_scale)
        x2 = int(max(self.start_x, end_x) / self.display_scale)
        y2 = int(max(self.start_y, end_y) / self.display_scale)
        h, w = self.image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        roi = self.image[y1:y2, x1:x2]

        self.show_roi(roi)

    def show_roi(self, roi):
        roi_canvas_w = self.roi_canvas.winfo_width()
        roi_canvas_h = self.roi_canvas.winfo_height()
        
        h, w = roi.shape[:2]

        scale = min(roi_canvas_w / w, roi_canvas_h / h, 1)

        self.display_scale = scale 
        resized = cv2.resize(roi, (roi_canvas_w, roi_canvas_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(resized)
        self.tk_roi = ImageTk.PhotoImage(pil_img)
        self.roi_canvas.delete("all")

        cx = roi_canvas_w // 2
        cy = roi_canvas_h // 2
        self.roi_canvas.create_image(cx, cy, anchor="center", image=self.tk_roi)

        

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x660")
    app = ROIselector(root)
    root.mainloop()