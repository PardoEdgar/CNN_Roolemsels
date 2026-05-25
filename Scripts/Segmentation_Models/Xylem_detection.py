"""
XylemVision - Xylem Cavity Detection & Measurement Tool
Detects and measures xylem vessel lumens from transverse sections.

Dependencies:
    pip install numpy opencv-python scikit-image matplotlib pillow pandas
    pip install torch torchvision  (for SAM2 - optional, falls back to classical)
    pip install segment-anything   (Meta SAM - lighter alternative)

Usage:
    python xylem_gui.py
"""


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
from pathlib import Path

import numpy as np
import cv2
from skimage import filters, morphology, measure, segmentation, color, exposure
from skimage.morphology import disk, remove_small_objects, binary_closing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from PIL import Image, ImageTk
import pandas as pd

# ─── Color palette ────────────────────────────────────────────────────────────
BG        = "#0f1117"
PANEL     = "#1a1d27"
CARD      = "#22263a"
ACCENT    = "#00d4aa"
ACCENT2   = "#4f8ef7"
WARN      = "#f59e42"
TEXT      = "#e8eaf0"
MUTED     = "#6b7280"
BORDER    = "#2e3347"
SUCCESS   = "#22c55e"
DANGER    = "#ef4444"

# ─── Segmentation engine ──────────────────────────────────────────────────────

class XylemSegmenter:
    """Classical CV segmentation with optional SAM fallback."""

    def __init__(self):
        self.sam_available = self._try_import_sam()

    def _try_import_sam(self):
        try:
            from segment_anything import sam_model_registry, SamPredictor
            self._sam_module = (sam_model_registry, SamPredictor)
            return True
        except ImportError:
            return False

    def preprocess(self, img_gray, clip_limit=3.0, tile_grid=8):
        """CLAHE contrast enhancement + bilateral denoise."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                 tileGridSize=(tile_grid, tile_grid))
        enhanced = clahe.apply(img_gray)
        denoised = cv2.bilateralFilter(enhanced, d=9,
                                       sigmaColor=75, sigmaSpace=75)
        return denoised

    def segment_classical(self, img_gray, min_area_px=50, max_area_px=None,
                           circularity_min=0.3, invert=True):
        """
        Multi-threshold watershed segmentation for xylem lumen detection.
        Returns labeled array and region properties.
        """
        processed = self.preprocess(img_gray)

        # Otsu threshold (lumens are bright on dark cell wall background)
        if invert:
            thresh_val = filters.threshold_otsu(processed)
            binary = processed > thresh_val
        else:
            thresh_val = filters.threshold_otsu(processed)
            binary = processed < thresh_val

        # Morphological cleanup
        binary = binary_closing(binary, disk(3))
        binary = remove_small_objects(binary, min_size=min_area_px)

        # Distance transform → watershed markers
        from scipy import ndimage as ndi
        distance = ndi.distance_transform_edt(binary)

        # Local maxima as seeds
        from skimage.feature import peak_local_max
        coords = peak_local_max(distance, min_distance=8,
                                labels=binary, exclude_border=False)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)

        labels = segmentation.watershed(-distance, markers, mask=binary)

        # Filter by area and circularity
        props = measure.regionprops(labels, intensity_image=img_gray)
        good_labels = []
        for p in props:
            area = p.area
            if area < min_area_px:
                continue
            if max_area_px and area > max_area_px:
                continue
            # Circularity = 4π·area / perimeter²
            if p.perimeter > 0:
                circ = (4 * np.pi * area) / (p.perimeter ** 2)
            else:
                circ = 0
            if circ >= circularity_min:
                good_labels.append(p.label)

        # Keep only good labels
        filtered = np.isin(labels, good_labels) * labels
        filtered_props = [p for p in props if p.label in good_labels]

        return filtered, filtered_props

    def segment_adaptive(self, img_gray, block_size=51, offset=10,
                          min_area_px=50, circularity_min=0.3):
        """Adaptive threshold — better for uneven illumination."""
        processed = self.preprocess(img_gray)
        binary = cv2.adaptiveThreshold(
            processed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, -offset
        )
        binary_bool = binary.astype(bool)
        binary_bool = binary_closing(binary_bool, disk(2))
        binary_bool = remove_small_objects(binary_bool, min_size=min_area_px)

        labels = measure.label(binary_bool)
        props = measure.regionprops(labels, intensity_image=img_gray)

        good_labels = []
        for p in props:
            if p.area < min_area_px:
                continue
            if p.perimeter > 0:
                circ = (4 * np.pi * p.area) / (p.perimeter ** 2)
            else:
                circ = 0
            if circ >= circularity_min:
                good_labels.append(p.label)

        filtered = np.isin(labels, good_labels) * labels
        filtered_props = [p for p in props if p.label in good_labels]
        return filtered, filtered_props


# ─── Measurement calculator ───────────────────────────────────────────────────

class MeasurementEngine:

    def compute(self, props, px_per_um):
        """Convert pixel measurements to µm units."""
        rows = []
        for i, p in enumerate(props):
            area_um2    = p.area / (px_per_um ** 2)
            perim_um    = p.perimeter / px_per_um
            major_um    = p.major_axis_length / px_per_um
            minor_um    = p.minor_axis_length / px_per_um
            equiv_diam  = p.equivalent_diameter / px_per_um
            if p.perimeter > 0:
                circ = (4 * np.pi * p.area) / (p.perimeter ** 2)
            else:
                circ = 0
            rows.append({
                "ID":               i + 1,
                "Area (µm²)":       round(area_um2, 2),
                "Perimeter (µm)":   round(perim_um, 2),
                "Major axis (µm)":  round(major_um, 2),
                "Minor axis (µm)":  round(minor_um, 2),
                "Equiv. diam (µm)": round(equiv_diam, 2),
                "Circularity":      round(circ, 4),
                "Centroid Y (px)":  round(p.centroid[0], 1),
                "Centroid X (px)":  round(p.centroid[1], 1),
            })
        return pd.DataFrame(rows)


# ─── Main GUI ─────────────────────────────────────────────────────────────────

class XylemVisionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("XylemVision  —  Cavity Detection & Measurement")
        self.geometry("1400x860")
        self.configure(bg=BG)
        self.minsize(1100, 700)

        # State
        self.img_path        = None
        self.img_gray        = None
        self.img_color       = None
        self.labels          = None
        self.props           = []
        self.df              = pd.DataFrame()
        self.px_per_um       = tk.DoubleVar(value=1.0)
        self.scale_bar_px    = tk.DoubleVar(value=200.0)
        self.scale_bar_um    = tk.DoubleVar(value=200.0)
        self.seg_method      = tk.StringVar(value="watershed")
        self.min_area        = tk.IntVar(value=100)
        self.max_area        = tk.IntVar(value=50000)
        self.circularity     = tk.DoubleVar(value=0.3)
        self.invert          = tk.BooleanVar(value=True)
        self.clip_limit      = tk.DoubleVar(value=3.0)
        self.show_labels     = tk.BooleanVar(value=True)
        self.show_overlay    = tk.BooleanVar(value=True)
        self.status_text     = tk.StringVar(value="Load an image to begin.")

        self.segmenter   = XylemSegmenter()
        self.measurer    = MeasurementEngine()

        self._build_ui()
        self._update_pxum()
        self.scale_bar_px.trace_add("write", lambda *a: self._update_pxum())
        self.scale_bar_um.trace_add("write", lambda *a: self._update_pxum())

    # ── Layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=PANEL, height=48)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        tk.Label(top, text="⬡ XylemVision", font=("Courier New", 15, "bold"),
                 bg=PANEL, fg=ACCENT).pack(side="left", padx=18, pady=10)
        tk.Label(top, text="Xylem Cavity Analyser  v1.0",
                 font=("Courier New", 9), bg=PANEL, fg=MUTED).pack(side="left", pady=14)

        # Status bar
        bot = tk.Frame(self, bg=PANEL, height=28)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)
        tk.Label(bot, textvariable=self.status_text, font=("Courier New", 9),
                 bg=PANEL, fg=MUTED, anchor="w").pack(side="left", padx=12, pady=6)

        # Main panes
        main = tk.PanedWindow(self, orient="horizontal",
                               bg=BG, sashwidth=5, sashrelief="flat")
        main.pack(fill="both", expand=True, padx=4, pady=4)

        left  = self._build_left_panel(main)
        right = self._build_right_panel(main)
        main.add(left,  minsize=280, width=310)
        main.add(right, minsize=600)

    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL)

        canvas_outer = tk.Canvas(frame, bg=PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical",
                                  command=canvas_outer.yview)
        canvas_outer.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_outer.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas_outer, bg=PANEL)
        win_id = canvas_outer.create_window((0, 0), window=inner, anchor="nw")

        def _on_frame_configure(e):
            canvas_outer.configure(scrollregion=canvas_outer.bbox("all"))
        def _on_canvas_configure(e):
            canvas_outer.itemconfig(win_id, width=e.width)

        inner.bind("<Configure>", _on_frame_configure)
        canvas_outer.bind("<Configure>", _on_canvas_configure)

        self._build_controls(inner)
        return frame

    def _build_controls(self, parent):
        pad = dict(padx=12, pady=4)

        def section(text):
            tk.Label(parent, text=text, font=("Courier New", 9, "bold"),
                     bg=PANEL, fg=ACCENT).pack(anchor="w", padx=12, pady=(14,2))
            tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x", padx=12, pady=(0,6))

        def row(label, widget_fn):
            f = tk.Frame(parent, bg=PANEL)
            f.pack(fill="x", padx=12, pady=2)
            tk.Label(f, text=label, font=("Courier New", 9), bg=PANEL,
                     fg=TEXT, width=18, anchor="w").pack(side="left")
            widget_fn(f)

        # — Load —
        section("📂  IMAGE")
        tk.Button(parent, text="Open Image…", command=self._load_image,
                  bg=ACCENT2, fg="white", font=("Courier New", 10, "bold"),
                  relief="flat", padx=10, pady=6, cursor="hand2"
                  ).pack(fill="x", padx=12, pady=4)

        self.img_label = tk.Label(parent, text="No image loaded",
                                   font=("Courier New", 8), bg=PANEL,
                                   fg=MUTED, wraplength=240)
        self.img_label.pack(**pad)

        # — Scale calibration —
        section("📏  SCALE CALIBRATION")
        row("Scale bar (px)",
            lambda f: tk.Entry(f, textvariable=self.scale_bar_px, width=8,
                               bg=CARD, fg=TEXT, insertbackground=TEXT,
                               font=("Courier New", 9), relief="flat"
                               ).pack(side="left"))
        row("Scale bar (µm)",
            lambda f: tk.Entry(f, textvariable=self.scale_bar_um, width=8,
                               bg=CARD, fg=TEXT, insertbackground=TEXT,
                               font=("Courier New", 9), relief="flat"
                               ).pack(side="left"))
        self.pxum_label = tk.Label(parent, text="1.00 px/µm",
                                    font=("Courier New", 8), bg=PANEL, fg=ACCENT)
        self.pxum_label.pack(**pad)

        # — Segmentation —
        section("⚙  SEGMENTATION")
        row("Method",
            lambda f: ttk.Combobox(f, textvariable=self.seg_method,
                                   values=["watershed", "adaptive"],
                                   state="readonly", width=14,
                                   font=("Courier New", 9)
                                   ).pack(side="left"))

        def slider_row(label, var, from_, to, res=1):
            f = tk.Frame(parent, bg=PANEL)
            f.pack(fill="x", padx=12, pady=2)
            tk.Label(f, text=label, font=("Courier New", 9),
                     bg=PANEL, fg=TEXT, width=18, anchor="w").pack(side="left")
            val_lbl = tk.Label(f, textvariable=var, width=6,
                               font=("Courier New", 9), bg=PANEL, fg=ACCENT)
            val_lbl.pack(side="right")
            tk.Scale(f, variable=var, from_=from_, to=to,
                     resolution=res, orient="horizontal",
                     bg=PANEL, fg=TEXT, troughcolor=CARD,
                     highlightthickness=0, sliderlength=14,
                     showvalue=False, length=100
                     ).pack(side="right", padx=4)

        slider_row("Min area (px²)", self.min_area, 10, 2000, 10)
        slider_row("Max area (px²)", self.max_area, 500, 100000, 500)
        slider_row("Circularity min", self.circularity, 0.0, 1.0, 0.05)
        slider_row("CLAHE clip limit", self.clip_limit, 1.0, 8.0, 0.5)

        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=12, pady=2)
        tk.Checkbutton(f, text="Invert (bright lumens)", variable=self.invert,
                       bg=PANEL, fg=TEXT, selectcolor=CARD,
                       activebackground=PANEL, font=("Courier New", 9)
                       ).pack(side="left")

        # — Display —
        section("🎨  DISPLAY")
        f2 = tk.Frame(parent, bg=PANEL)
        f2.pack(fill="x", padx=12, pady=2)
        tk.Checkbutton(f2, text="Show labels", variable=self.show_labels,
                       bg=PANEL, fg=TEXT, selectcolor=CARD,
                       activebackground=PANEL, font=("Courier New", 9)
                       ).pack(side="left")
        tk.Checkbutton(f2, text="Show overlay", variable=self.show_overlay,
                       bg=PANEL, fg=TEXT, selectcolor=CARD,
                       activebackground=PANEL, font=("Courier New", 9)
                       ).pack(side="left", padx=8)

        # — Run —
        section("▶  RUN")
        tk.Button(parent, text="▶  Run Analysis",
                  command=self._run_analysis,
                  bg=ACCENT, fg=BG, font=("Courier New", 11, "bold"),
                  relief="flat", padx=10, pady=8, cursor="hand2"
                  ).pack(fill="x", padx=12, pady=4)

        self.progress = ttk.Progressbar(parent, mode="indeterminate", length=240)
        self.progress.pack(padx=12, pady=4)

        # — Export —
        section("💾  EXPORT")
        f3 = tk.Frame(parent, bg=PANEL)
        f3.pack(fill="x", padx=12, pady=4)
        tk.Button(f3, text="Export CSV",
                  command=self._export_csv,
                  bg=CARD, fg=ACCENT, font=("Courier New", 9),
                  relief="flat", padx=8, pady=4, cursor="hand2"
                  ).pack(side="left", padx=(0, 6))
        tk.Button(f3, text="Save Image",
                  command=self._save_overlay,
                  bg=CARD, fg=ACCENT, font=("Courier New", 9),
                  relief="flat", padx=8, pady=4, cursor="hand2"
                  ).pack(side="left")

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=BG)

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background=CARD,
                        foreground=MUTED, font=("Courier New", 9),
                        padding=[12, 6])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", PANEL)],
                  foreground=[("selected", ACCENT)])

        self.nb = ttk.Notebook(frame, style="Dark.TNotebook")
        self.nb.pack(fill="both", expand=True)

        self._build_image_tab()
        self._build_results_tab()
        self._build_stats_tab()

        return frame

    def _build_image_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Image & Overlay  ")

        self.fig_img = Figure(figsize=(9, 6), facecolor=BG)
        self.ax_img  = self.fig_img.add_subplot(111)
        self.ax_img.set_facecolor(CARD)
        self.ax_img.tick_params(colors=MUTED)
        for spine in self.ax_img.spines.values():
            spine.set_edgecolor(BORDER)
        self.ax_img.set_title("Load an image to begin",
                               color=MUTED, fontsize=9, fontfamily="monospace")

        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=tab)
        self.canvas_img.draw()
        toolbar = NavigationToolbar2Tk(self.canvas_img, tab,
                                       pack_toolbar=False)
        toolbar.config(bg=PANEL)
        toolbar.pack(side="bottom", fill="x")
        self.canvas_img.get_tk_widget().pack(fill="both", expand=True)

    def _build_results_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Measurements  ")

        cols = ("ID", "Area (µm²)", "Perimeter (µm)", "Major axis (µm)",
                "Minor axis (µm)", "Equiv. diam (µm)", "Circularity")

        style = ttk.Style()
        style.configure("Dark.Treeview",
                         background=CARD, fieldbackground=CARD,
                         foreground=TEXT, font=("Courier New", 9),
                         rowheight=22)
        style.configure("Dark.Treeview.Heading",
                         background=PANEL, foreground=ACCENT,
                         font=("Courier New", 9, "bold"))
        style.map("Dark.Treeview", background=[("selected", ACCENT2)])

        self.tree = ttk.Treeview(tab, columns=cols, show="headings",
                                  style="Dark.Treeview")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=110, anchor="center")

        vsb = ttk.Scrollbar(tab, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tab, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        # Summary bar
        self.summary_lbl = tk.Label(tab, text="",
                                     font=("Courier New", 9),
                                     bg=PANEL, fg=ACCENT, anchor="w")
        self.summary_lbl.pack(fill="x", padx=8, pady=4)

    def _build_stats_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Statistics  ")

        self.fig_stats = Figure(figsize=(9, 6), facecolor=BG)
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=tab)
        self.canvas_stats.draw()
        self.canvas_stats.get_tk_widget().pack(fill="both", expand=True)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Open microscopy image",
            filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        self.img_path = path
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return

        # Normalise to 8-bit gray
        if img.ndim == 3:
            self.img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            self.img_gray  = img.copy()
            self.img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.img_gray.dtype != np.uint8:
            mn, mx = self.img_gray.min(), self.img_gray.max()
            self.img_gray = ((self.img_gray - mn) / (mx - mn + 1e-8) * 255
                             ).astype(np.uint8)

        fname = Path(path).name
        h, w  = self.img_gray.shape
        self.img_label.config(text=f"{fname}\n{w}×{h} px")
        self.status_text.set(f"Loaded: {fname}  ({w}×{h} px)")
        self._display_image(self.img_color, title="Original image")

    def _update_pxum(self, *_):
        try:
            ratio = self.scale_bar_px.get() / self.scale_bar_um.get()
            self.px_per_um.set(round(ratio, 4))
            self.pxum_label.config(text=f"{ratio:.4f} px/µm")
        except (ZeroDivisionError, tk.TclError):
            pass

    def _run_analysis(self):
        if self.img_gray is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        self.progress.start(10)
        self.status_text.set("Running segmentation…")
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self):
        try:
            method = self.seg_method.get()
            if method == "watershed":
                labels, props = self.segmenter.segment_classical(
                    self.img_gray,
                    min_area_px=self.min_area.get(),
                    max_area_px=self.max_area.get(),
                    circularity_min=self.circularity.get(),
                    invert=self.invert.get()
                )
            else:
                labels, props = self.segmenter.segment_adaptive(
                    self.img_gray,
                    min_area_px=self.min_area.get(),
                    circularity_min=self.circularity.get()
                )

            self.labels = labels
            self.props  = props
            self.df     = self.measurer.compute(props, self.px_per_um.get())

            self.after(0, self._update_ui_after_analysis)
        except Exception as e:
            self.after(0, lambda: (
                self.progress.stop(),
                self.status_text.set(f"Error: {e}"),
                messagebox.showerror("Analysis error", str(e))
            ))

    def _update_ui_after_analysis(self):
        self.progress.stop()
        n = len(self.props)
        self.status_text.set(
            f"Found {n} cavities  |  "
            f"px/µm = {self.px_per_um.get():.4f}  |  "
            f"method = {self.seg_method.get()}"
        )
        self._display_overlay()
        self._populate_table()
        self._draw_stats()
        self.nb.select(0)

    def _display_image(self, img_rgb, title=""):
        self.ax_img.clear()
        self.ax_img.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
        self.ax_img.set_title(title, color=TEXT, fontsize=9, fontfamily="monospace")
        self.ax_img.axis("off")
        self.fig_img.tight_layout(pad=0.5)
        self.canvas_img.draw()

    def _display_overlay(self):
        self.ax_img.clear()

        if self.show_overlay.get() and self.labels is not None:
            overlay = color.label2rgb(self.labels, image=self.img_gray,
                                       bg_label=0, alpha=0.35,
                                       bg_color=None)
            self.ax_img.imshow(overlay)
        else:
            self.ax_img.imshow(self.img_color)

        if self.show_labels.get() and self.props:
            for i, p in enumerate(self.props):
                cy, cx = p.centroid
                self.ax_img.text(cx, cy, str(i + 1),
                                  color="white", fontsize=6,
                                  ha="center", va="center",
                                  fontweight="bold",
                                  bbox=dict(boxstyle="round,pad=0.1",
                                            fc="black", alpha=0.5, lw=0))

        n = len(self.props)
        self.ax_img.set_title(
            f"{n} cavities detected  |  {self.seg_method.get()} method",
            color=ACCENT, fontsize=9, fontfamily="monospace"
        )
        self.ax_img.axis("off")
        self.fig_img.tight_layout(pad=0.5)
        self.canvas_img.draw()

    def _populate_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        cols = ("ID", "Area (µm²)", "Perimeter (µm)", "Major axis (µm)",
                "Minor axis (µm)", "Equiv. diam (µm)", "Circularity")
        for _, r in self.df.iterrows():
            self.tree.insert("", "end",
                              values=tuple(r[c] for c in cols))

        if not self.df.empty:
            mean_a = self.df["Area (µm²)"].mean()
            std_a  = self.df["Area (µm²)"].std()
            tot    = self.df["Area (µm²)"].sum()
            self.summary_lbl.config(
                text=f"  n={len(self.df)}   "
                     f"mean area={mean_a:.1f} µm²   "
                     f"SD={std_a:.1f}   "
                     f"total lumen area={tot:.1f} µm²"
            )

    def _draw_stats(self):
        if self.df.empty:
            return
        self.fig_stats.clear()

        axes = self.fig_stats.subplots(2, 2)
        self.fig_stats.patch.set_facecolor(BG)

        def style_ax(ax, title, xlabel, ylabel="Count"):
            ax.set_facecolor(CARD)
            ax.set_title(title, color=TEXT, fontsize=8, fontfamily="monospace")
            ax.set_xlabel(xlabel, color=MUTED, fontsize=7)
            ax.set_ylabel(ylabel, color=MUTED, fontsize=7)
            ax.tick_params(colors=MUTED, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        # Area distribution
        axes[0,0].hist(self.df["Area (µm²)"], bins=20,
                        color=ACCENT, edgecolor=BG, linewidth=0.5)
        style_ax(axes[0,0], "Lumen area distribution", "Area (µm²)")

        # Circularity
        axes[0,1].hist(self.df["Circularity"], bins=20,
                        color=ACCENT2, edgecolor=BG, linewidth=0.5)
        style_ax(axes[0,1], "Circularity distribution", "Circularity")

        # Area vs Circularity scatter
        axes[1,0].scatter(self.df["Area (µm²)"], self.df["Circularity"],
                           c=WARN, s=18, alpha=0.7, linewidths=0)
        style_ax(axes[1,0], "Area vs Circularity",
                 "Area (µm²)", "Circularity")

        # Equivalent diameter
        axes[1,1].hist(self.df["Equiv. diam (µm)"], bins=20,
                        color=SUCCESS, edgecolor=BG, linewidth=0.5)
        style_ax(axes[1,1], "Equivalent diameter", "Diam. (µm)")

        self.fig_stats.tight_layout(pad=1.5)
        self.canvas_stats.draw()
        self.nb.select(2)

    def _export_csv(self):
        if self.df.empty:
            messagebox.showinfo("No data", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="xylem_measurements.csv"
        )
        if path:
            self.df.to_csv(path, index=False)
            self.status_text.set(f"Saved CSV → {path}")
            messagebox.showinfo("Saved", f"Data exported to:\n{path}")

    def _save_overlay(self):
        if self.labels is None:
            messagebox.showinfo("No overlay", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif")],
            initialfile="xylem_overlay.png"
        )
        if path:
            self.fig_img.savefig(path, dpi=200, bbox_inches="tight",
                                  facecolor=BG)
            self.status_text.set(f"Saved overlay → {path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = XylemVisionApp()
    app.mainloop()