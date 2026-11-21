#!/usr/bin/env python3
"""
PEPPAH - Adulterant Detection UI
Detects papaya seeds (adulterant) in black pepper samples.
Raspberry Pi 5 + Camera Module 3 + YOLOv8 (NCNN)

Setup (one-time):
    sudo apt install -y python3-pil.imagetk
    pip install ultralytics
    python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='ncnn')"

Usage:
    python detector_ui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent
NCNN_MODEL_PATH = MODEL_DIR / "best_ncnn_model"
PT_MODEL_PATH = MODEL_DIR / "best.pt"

# ---------------------------------------------------------------------------
# Camera settings (reused from capture_image.py)
# ---------------------------------------------------------------------------
from picamera2 import Picamera2
from libcamera import controls

CAPTURE_RESOLUTION = (2304, 1296)
PREVIEW_RESOLUTION = (640, 360)

MACRO_CONTROLS = {
    "AfMode": controls.AfModeEnum.Continuous,
    "AfRange": controls.AfRangeEnum.Macro,
    "AfSpeed": controls.AfSpeedEnum.Fast,
    "Sharpness": 2.5,
    "Contrast": 1.3,
    "Saturation": 1.4,
    "Brightness": 0.05,
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
}

# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5

CLASS_COLORS = {
    "adulterant": (0, 0, 255),   # Red (BGR) - papaya seeds
    "pepper": (0, 200, 0),       # Green (BGR) - black pepper
}
# ---------------------------------------------------------------------------
# UI colors
# ---------------------------------------------------------------------------
BG_DARK = "#1e1e1e"
BG_BAR = "#2c2c2c"
FG_WHITE = "#ffffff"
FG_DIM = "#aaaaaa"
GREEN = "#4CAF50"
RED = "#f44336"
BLUE = "#2196F3"
ORANGE = "#FF9800"
INACTIVE = "#555555"


# ═══════════════════════════════════════════════════════════════════════════
# Detection Engine
# ═══════════════════════════════════════════════════════════════════════════
class DetectionEngine:
    """Loads YOLOv8 model (NCNN preferred, .pt fallback) and runs detection."""

    def __init__(self):
        self.model = None
        self.class_names = {}
        self._load_model()

    def _load_model(self):
        from ultralytics import YOLO

        if PT_MODEL_PATH.exists():
            self.model = YOLO(str(PT_MODEL_PATH))
            print(f"Loaded PyTorch model from {PT_MODEL_PATH}")
        elif NCNN_MODEL_PATH.exists():
            self.model = YOLO(str(NCNN_MODEL_PATH), task="detect")
            print(f"Loaded NCNN model from {NCNN_MODEL_PATH}")
        else:
            raise FileNotFoundError("No model found (best.pt or best_ncnn_model/)")
        self.class_names = self.model.names

    def detect(self, frame_bgr):
        """Run detection on a BGR numpy array. Returns list of detection dicts."""
        results = self.model.predict(
            source=frame_bgr,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=640,
            agnostic_nms=True,
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                detections.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "confidence": float(box.conf[0]),
                    "class_id": cls_id,
                    "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                })
        return detections


# ═══════════════════════════════════════════════════════════════════════════
# Camera Manager
# ═══════════════════════════════════════════════════════════════════════════
class CameraManager:
    """Wraps picamera2 with dual-stream macro configuration."""

    def __init__(self):
        self.picam2 = None
        self.running = False

    def start(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": CAPTURE_RESOLUTION},
            lores={"format": "YUV420", "size": PREVIEW_RESOLUTION},
            controls=MACRO_CONTROLS,
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # let AF and exposure settle
        self.running = True

    def get_frame(self):
        """Capture lores frame for detection. Returns BGR numpy array."""
        if not self.running:
            return None
        yuv = self.picam2.capture_array("lores")
        return cv2.cvtColor(yuv, cv2.COLOR_YUV420p2BGR)

    def stop(self):
        self.running = False
        if self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
            self.picam2 = None


# ═══════════════════════════════════════════════════════════════════════════
# Drawing & Stats utilities
# ═══════════════════════════════════════════════════════════════════════════
def draw_detections(frame_bgr, detections):
    """Draw colored bounding boxes on the frame (no labels). Returns frame."""
    for det in detections:
        color = CLASS_COLORS.get(det["class_name"], (255, 255, 255))
        cv2.rectangle(frame_bgr,
                      (det["x1"], det["y1"]),
                      (det["x2"], det["y2"]),
                      color, 2)
    return frame_bgr


def compute_stats(detections):
    """Compute seed counts and adulterant percentage."""
    adulterant = sum(1 for d in detections if d["class_name"] == "adulterant")
    pepper = sum(1 for d in detections if d["class_name"] == "pepper")
    total = adulterant + pepper
    pct = (adulterant / total * 100) if total > 0 else 0.0
    return {
        "adulterant_count": adulterant,
        "pepper_count": pepper,
        "total_count": total,
        "adulterant_pct": pct,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════
class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PEPPAH - Adulterant Detector")
        self.root.geometry("1024x700")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_DARK)

        # State
        self.mode = "live"
        self.running = False
        self.detection_thread = None
        self.camera = None
        self.engine = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self._photo = None
        self._last_annotated = None
        self._last_detections = None

        # Load model
        self.engine = DetectionEngine()

        # Build UI
        self._build_mode_bar()
        self._build_status_bar()
        self._build_controls_bar()
        self._build_content()

        # Start UI polling loop
        self._schedule_ui_update()

    # -------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------
    def _build_mode_bar(self):
        bar = tk.Frame(self.root, bg=BG_BAR, height=45)
        bar.pack(fill=tk.X, side=tk.TOP)
        bar.pack_propagate(False)

        self.btn_live = tk.Button(
            bar, text="Live Detection", command=self._switch_to_live,
            bg=GREEN, fg=FG_WHITE, activebackground=GREEN,
            font=("Helvetica", 12, "bold"), relief=tk.FLAT, padx=20, pady=5,
        )
        self.btn_live.pack(side=tk.LEFT, padx=(15, 5), pady=8)

        self.btn_image = tk.Button(
            bar, text="Image Detection", command=self._switch_to_image,
            bg=INACTIVE, fg=FG_WHITE, activebackground=INACTIVE,
            font=("Helvetica", 12, "bold"), relief=tk.FLAT, padx=20, pady=5,
        )
        self.btn_image.pack(side=tk.LEFT, padx=5, pady=8)

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Ready - Click Start to begin live detection")
        bar = tk.Label(
            self.root, textvariable=self.status_var,
            bg="#1a1a1a", fg=FG_DIM, font=("Helvetica", 10),
            anchor=tk.W, padx=10,
        )
        bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_controls_bar(self):
        bar = tk.Frame(self.root, bg=BG_BAR, height=50)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        self.btn_start = tk.Button(
            bar, text="Start", command=self._toggle_live,
            bg=GREEN, fg=FG_WHITE, activebackground=GREEN,
            font=("Helvetica", 11, "bold"), width=12,
        )
        self.btn_start.pack(side=tk.LEFT, padx=15, pady=8)

        self.btn_browse = tk.Button(
            bar, text="Browse Image", command=self._browse_image,
            bg=BLUE, fg=FG_WHITE, activebackground=BLUE,
            font=("Helvetica", 11), width=14, state=tk.DISABLED,
        )
        self.btn_browse.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_save = tk.Button(
            bar, text="Save Result", command=self._save_result,
            bg=ORANGE, fg=FG_WHITE, activebackground=ORANGE,
            font=("Helvetica", 11), width=12, state=tk.DISABLED,
        )
        self.btn_save.pack(side=tk.LEFT, padx=5, pady=8)

    def _build_content(self):
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: video / image canvas
        self.canvas = tk.Canvas(
            content, width=640, height=480, bg="black", highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))

        # Right: stats panel
        stats = tk.Frame(content, bg=BG_DARK, width=340)
        stats.pack(side=tk.RIGHT, fill=tk.Y)
        stats.pack_propagate(False)

        tk.Label(
            stats, text="DETECTION RESULTS",
            font=("Helvetica", 14, "bold"), bg=BG_DARK, fg=FG_WHITE,
        ).pack(pady=(20, 10))

        # Color legend
        legend = tk.Frame(stats, bg=BG_DARK)
        legend.pack(fill=tk.X, padx=20, pady=(0, 8))

        for color_hex, label_text in [("#00C800", "Black Pepper"), ("#FF4444", "Papaya Seed (Adulterant)")]:
            entry = tk.Frame(legend, bg=BG_DARK)
            entry.pack(fill=tk.X, pady=2)
            swatch = tk.Canvas(entry, width=16, height=16, bg=color_hex,
                               highlightthickness=1, highlightbackground="#555")
            swatch.pack(side=tk.LEFT, padx=(0, 8))
            tk.Label(entry, text=label_text, font=("Helvetica", 11),
                     bg=BG_DARK, fg="#cccccc").pack(side=tk.LEFT)

        ttk.Separator(stats).pack(fill=tk.X, padx=20)

        self.lbl_pepper = self._stat_row(stats, "Black Pepper:", "0", "#00C800")
        self.lbl_adulterant = self._stat_row(stats, "Papaya Seeds:", "0", "#FF4444")

        ttk.Separator(stats).pack(fill=tk.X, padx=20, pady=5)

        self.lbl_total = self._stat_row(stats, "Total Seeds:", "0")
        self.lbl_pct = self._stat_row(stats, "Adulterant %:", "0.0%", "#FF4444")

        ttk.Separator(stats).pack(fill=tk.X, padx=20, pady=5)

        self.lbl_fps = self._stat_row(stats, "FPS:", "--")
        self.lbl_inference = self._stat_row(stats, "Inference:", "--")

    def _stat_row(self, parent, label, value, fg="#FFFFFF"):
        row = tk.Frame(parent, bg=BG_DARK)
        row.pack(fill=tk.X, padx=20, pady=4)
        tk.Label(
            row, text=label, font=("Helvetica", 13),
            bg=BG_DARK, fg="#cccccc",
        ).pack(side=tk.LEFT)
        val = tk.Label(
            row, text=value, font=("Helvetica", 16, "bold"),
            bg=BG_DARK, fg=fg,
        )
        val.pack(side=tk.RIGHT)
        return val

    # -------------------------------------------------------------------
    # Mode switching
    # -------------------------------------------------------------------
    def _switch_to_live(self):
        if self.mode == "live":
            return
        if self.running:
            self._stop_live()
        self.mode = "live"
        self.btn_live.configure(bg=GREEN)
        self.btn_image.configure(bg=INACTIVE)
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_browse.configure(state=tk.DISABLED)
        self.status_var.set("Live Detection - Click Start to begin")

    def _switch_to_image(self):
        if self.running:
            self._stop_live()
        self.mode = "image"
        self.btn_image.configure(bg=GREEN)
        self.btn_live.configure(bg=INACTIVE)
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_browse.configure(state=tk.NORMAL)
        self.status_var.set("Image Detection - Browse an image to analyze")

    # -------------------------------------------------------------------
    # Live detection
    # -------------------------------------------------------------------
    def _toggle_live(self):
        if self.running:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self):
        self.status_var.set("Starting camera...")
        self.root.update()

        try:
            self.camera = CameraManager()
            self.camera.start()
        except Exception as e:
            self.status_var.set(f"Camera error: {e}")
            return

        self.stop_event.clear()
        self.running = True
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self.detection_thread.start()

        self.btn_start.configure(text="Stop", bg=RED, activebackground=RED)
        self.btn_save.configure(state=tk.NORMAL)
        self.status_var.set("Live Detection - Running")

    def _stop_live(self):
        self.stop_event.set()
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=3)
            self.detection_thread = None
        if self.camera:
            self.camera.stop()
            self.camera = None

        # drain queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.btn_start.configure(text="Start", bg=GREEN, activebackground=GREEN)
        self.status_var.set("Live Detection - Stopped")

    def _detection_loop(self):
        """Background thread: capture -> detect -> draw -> enqueue."""
        fps_count = 0
        fps_time = time.time()
        current_fps = 0.0

        while not self.stop_event.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            detections = self.engine.detect(frame)
            inference_ms = (time.time() - t0) * 1000

            annotated = draw_detections(frame.copy(), detections)

            stats = compute_stats(detections)
            stats["inference_ms"] = inference_ms

            fps_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_time = time.time()
            stats["fps"] = current_fps

            # store for save
            self._last_annotated = annotated
            self._last_detections = detections

            # enqueue, dropping oldest if full
            try:
                self.frame_queue.put_nowait((annotated, stats))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait((annotated, stats))

    # -------------------------------------------------------------------
    # Image detection
    # -------------------------------------------------------------------
    def _browse_image(self):
        filepath = filedialog.askopenfilename(
            initialdir=str(MODEL_DIR / "captures"),
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if filepath:
            self._detect_on_image(filepath)

    def _detect_on_image(self, filepath):
        self.status_var.set(f"Analyzing: {Path(filepath).name}...")
        self.root.update()

        frame_bgr = cv2.imread(filepath)
        if frame_bgr is None:
            self.status_var.set(f"Error: Could not load {filepath}")
            return

        t0 = time.time()
        detections = self.engine.detect(frame_bgr)
        inference_ms = (time.time() - t0) * 1000

        annotated = draw_detections(frame_bgr.copy(), detections)
        self._last_annotated = annotated
        self._last_detections = detections

        # resize for display preserving aspect ratio
        h, w = annotated.shape[:2]
        scale = min(640 / w, 480 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(annotated, (new_w, new_h))

        # pad to 640x480 with black
        padded = np.zeros((480, 640, 3), dtype=np.uint8)
        y_off = (480 - new_h) // 2
        x_off = (640 - new_w) // 2
        padded[y_off:y_off + new_h, x_off:x_off + new_w] = display

        self._update_canvas(padded)

        stats = compute_stats(detections)
        stats["inference_ms"] = inference_ms
        self._update_stats(stats)

        self.btn_save.configure(state=tk.NORMAL)
        self.status_var.set(
            f"Image: {Path(filepath).name} | "
            f"{stats['total_count']} seeds | {inference_ms:.0f}ms"
        )

    # -------------------------------------------------------------------
    # UI update loop (main thread)
    # -------------------------------------------------------------------
    def _schedule_ui_update(self):
        try:
            while not self.frame_queue.empty():
                frame_bgr, stats = self.frame_queue.get_nowait()
                self._update_canvas(frame_bgr)
                self._update_stats(stats)
        except queue.Empty:
            pass
        self.root.after(30, self._schedule_ui_update)

    def _update_canvas(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(640 / w, 480 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.NEAREST)
        # center on black 640x480 canvas
        self._photo = ImageTk.PhotoImage(image=pil_img)
        x_off = (640 - new_w) // 2
        y_off = (480 - new_h) // 2
        self.canvas.delete("all")
        self.canvas.create_image(x_off, y_off, anchor=tk.NW, image=self._photo)

    def _update_stats(self, stats):
        self.lbl_pepper.configure(text=str(stats["pepper_count"]))
        self.lbl_adulterant.configure(text=str(stats["adulterant_count"]))
        self.lbl_total.configure(text=str(stats["total_count"]))
        self.lbl_pct.configure(text=f"{stats['adulterant_pct']:.1f}%")

        if "fps" in stats:
            self.lbl_fps.configure(text=f"{stats['fps']:.1f}")
        if "inference_ms" in stats:
            self.lbl_inference.configure(text=f"{stats['inference_ms']:.0f}ms")

        # color-code adulterant percentage
        pct = stats["adulterant_pct"]
        if pct == 0:
            color = "#00C800"
        elif pct < 10:
            color = "#FFD700"
        else:
            color = "#FF4444"
        self.lbl_pct.configure(fg=color)

    # -------------------------------------------------------------------
    # Save result
    # -------------------------------------------------------------------
    def _save_result(self):
        if self._last_annotated is None:
            return
        captures_dir = MODEL_DIR / "captures"
        captures_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = captures_dir / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(filename), self._last_annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.status_var.set(f"Saved: {filename.name}")

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------
    def on_close(self):
        if self.running:
            self._stop_live()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    app = DetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
