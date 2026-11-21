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
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR     = Path(__file__).parent
PT_MODEL_PATH = MODEL_DIR / "best.pt"
TRAINING_DIR  = MODEL_DIR / "training_data"

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
# YOLO settings
# ---------------------------------------------------------------------------
YOLO_CONF = 0.15
YOLO_IOU  = 0.5

# ---------------------------------------------------------------------------
# Shape detection settings
# ---------------------------------------------------------------------------
MIN_CONTOUR_AREA      = 80    # px² — ignore dust/noise (small after watershed)
MAX_CONTOUR_AREA      = 5000  # px² — single-seed upper bound
AXIS_RATIO_THRESH     = 0.72  # only clearly round shapes (≥0.72) are pepper; all else adulterant
WATERSHED_DIST_THRESH = 4     # px — min dist from edge to count as seed center
COLOR_DARK_V_THRESH   = 120   # HSV value; pepper is very dark (<120); papaya seeds are lighter
FINETUNE_MIN_SAMPLES  = 5     # corrections needed before Fine-tune button enables

CLASS_COLORS = {
    "adulterant": (0, 0, 255),   # Red (BGR) - papaya seeds
    "pepper": (0, 200, 0),       # Green (BGR) - black pepper
}
CLASS_IDS = {"adulterant": 0, "pepper": 1}

# ---------------------------------------------------------------------------
# Canvas / correction constants
# ---------------------------------------------------------------------------
# Lores 640x360 is letterboxed into 640x480 canvas → 60px top/bottom offset
CANVAS_W, CANVAS_H = 640, 480
CANVAS_Y_OFFSET = (CANVAS_H - PREVIEW_RESOLUTION[1]) // 2  # = 60

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
PURPLE = "#9C27B0"
INACTIVE = "#555555"


# ═══════════════════════════════════════════════════════════════════════════
# Shape Detection Engine
# ═══════════════════════════════════════════════════════════════════════════
class ShapeDetectionEngine:
    """Classifies seeds by contour shape: round=pepper, oblong=adulterant."""

    def detect(self, frame_bgr, **_kwargs):
        gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        sure_bg = cv2.dilate(opening, kernel, iterations=2)

        # Distance transform: peaks = seed centers → 1 marker per seed
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []
        _, sure_fg = cv2.threshold(dist, WATERSHED_DIST_THRESH, 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(frame_bgr.copy(), markers)

        detections = []
        for label in range(2, int(markers.max()) + 1):
            mask = np.uint8(markers == label) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            if len(cnt) < 5:
                continue  # fitEllipse needs ≥5 points
            (_, _), (d1, d2), _ = cv2.fitEllipse(cnt)
            if max(d1, d2) == 0:
                continue
            axis_ratio = min(d1, d2) / max(d1, d2)
            x, y, w, h = cv2.boundingRect(cnt)
            cls_name = "pepper" if axis_ratio >= AXIS_RATIO_THRESH else "adulterant"
            if cls_name == "pepper" and not _is_dark_seed(frame_bgr[y:y+h, x:x+w]):
                cls_name = "adulterant"
            detections.append({
                "x1": x, "y1": y,
                "x2": x + w, "y2": y + h,
                "confidence": round(axis_ratio, 3),
                "class_id": CLASS_IDS[cls_name],
                "class_name": cls_name,
            })
        return detections


# ═══════════════════════════════════════════════════════════════════════════
# YOLO Detection Engine  (PyTorch only — thread-safe for background thread)
# ═══════════════════════════════════════════════════════════════════════════
class YOLODetectionEngine:
    """Loads best.pt and runs YOLO inference for seed localization."""

    def __init__(self):
        from ultralytics import YOLO
        if not PT_MODEL_PATH.exists():
            raise FileNotFoundError(f"No model at {PT_MODEL_PATH}")
        self.model = YOLO(str(PT_MODEL_PATH))
        print(f"[YOLO] Loaded {PT_MODEL_PATH}")
        # Warmup to eliminate first-frame latency spike
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        self.model.predict(source=dummy, conf=YOLO_CONF, iou=YOLO_IOU,
                           imgsz=640, agnostic_nms=True, verbose=False, half=True)
        print("[YOLO] Warmup complete")

    def detect(self, frame_bgr):
        results = self.model.predict(source=frame_bgr, conf=YOLO_CONF, iou=YOLO_IOU,
                                     imgsz=640, agnostic_nms=True, verbose=False, half=True)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "confidence": float(box.conf[0]),
                    "class_id": cls_id,
                    "class_name": cls_name,
                })
        return detections


def _is_dark_seed(roi_bgr):
    """Returns True if the ROI is dark (black pepper); False if lighter (papaya/adulterant)."""
    if roi_bgr is None or roi_bgr.size == 0:
        return True
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean()) < COLOR_DARK_V_THRESH


def _shape_classify_roi(roi_thresh, roi_bgr=None):
    """Classify a pre-thresholded ROI by shape + color. Returns 'pepper'|'adulterant'|None."""
    if roi_thresh.size == 0:
        return None
    cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    (_, _), (d1, d2), _ = cv2.fitEllipse(cnt)
    if max(d1, d2) == 0:
        return None
    axis_ratio = min(d1, d2) / max(d1, d2)
    cls = "pepper" if axis_ratio >= AXIS_RATIO_THRESH else "adulterant"
    if cls == "pepper" and not _is_dark_seed(roi_bgr):
        cls = "adulterant"
    return cls


def find_sample_roi(frame_bgr):
    """Auto-detect the square/rectangular sample border. Returns (x1,y1,x2,y2) or None."""
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges   = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = frame_bgr.shape[:2]
    frame_area = fw * fh

    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.08 * frame_area or area > 0.92 * frame_area:
            continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) == 4 and area > best_area:
            x, y, w, h = cv2.boundingRect(approx)
            if 0.4 < (w / h if h > 0 else 0) < 2.5:
                best_area = area
                best = (x, y, x + w, y + h)
    return best


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Detection Engine  (YOLO locates → shape confirms class + fills gaps)
# ═══════════════════════════════════════════════════════════════════════════
class HybridDetectionEngine:
    """YOLO finds seeds; shape overrides class; shape fills in seeds YOLO missed."""

    def __init__(self):
        self.yolo     = YOLODetectionEngine()
        self.shape    = ShapeDetectionEngine()
        self.last_roi = None   # exposed for drawing in detection loop

    def detect(self, frame_bgr, **_kwargs):
        # 1. Auto-detect square sample border
        roi = find_sample_roi(frame_bgr)
        self.last_roi = roi

        # 2. YOLO locates seeds
        yolo_dets = self.yolo.detect(frame_bgr)

        # 3. Build threshold image once (shared by shape steps)
        gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Shape-confirm each YOLO detection; filter to ROI
        confirmed = []
        for det in yolo_dets:
            cx = (det["x1"] + det["x2"]) // 2
            cy = (det["y1"] + det["y2"]) // 2
            if roi and not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                continue  # outside border — skip
            crop_bgr  = frame_bgr[det["y1"]:det["y2"], det["x1"]:det["x2"]]
            crop      = thresh[det["y1"]:det["y2"], det["x1"]:det["x2"]]
            shape_cls = _shape_classify_roi(crop, crop_bgr)
            cls_name  = shape_cls if shape_cls else det["class_name"]
            confirmed.append({
                **det,
                "class_name": cls_name,
                "class_id":   CLASS_IDS.get(cls_name, det["class_id"]),
            })

        # 5. Shape detection fills seeds YOLO missed
        if roi:
            rx1, ry1, rx2, ry2 = roi
            shape_raw = self.shape.detect(frame_bgr[ry1:ry2, rx1:rx2])
            shape_dets = [{**d, "x1": d["x1"]+rx1, "x2": d["x2"]+rx1,
                                 "y1": d["y1"]+ry1, "y2": d["y2"]+ry1}
                          for d in shape_raw]
        else:
            shape_dets = self.shape.detect(frame_bgr)

        for sd in shape_dets:
            cx = (sd["x1"] + sd["x2"]) // 2
            cy = (sd["y1"] + sd["y2"]) // 2
            if not any(d["x1"] <= cx <= d["x2"] and d["y1"] <= cy <= d["y2"]
                       for d in confirmed):
                confirmed.append(sd)   # gap-fill: seed YOLO missed

        return confirmed

    def reload(self):
        """Reload YOLO weights after fine-tuning."""
        self.yolo = YOLODetectionEngine()


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


def frame_to_photoimage(frame_bgr):
    """Convert BGR numpy array to Tkinter PhotoImage, letterboxed to CANVAS_W x CANVAS_H."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    scale = min(CANVAS_W / w, CANVAS_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.NEAREST)
    return ImageTk.PhotoImage(image=pil_img), new_w, new_h


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

        # Detection state
        self.mode = "live"
        self.running = False
        self.detection_thread = None
        self.camera = None
        self.engine = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self._photo = None
        self._last_annotated = None
        self._last_raw_frame = None
        self._last_detections = None

        # Correction mode state
        self._correction_mode = False
        self._correction_frame = None
        self._correction_boxes = []
        self._selected_idx = None
        self._draw_start = None
        self._current_class = "adulterant"
        self._correction_count = self._count_saved_corrections()

        # Fine-tuning state
        self._finetuning = False

        # Try hybrid (YOLO + shape), fall back to shape-only if model absent
        try:
            self.engine = HybridDetectionEngine()
            self._engine_label = "YOLO + Shape"
        except FileNotFoundError:
            self.engine = ShapeDetectionEngine()
            self._engine_label = "Shape only"

        # Build UI
        self._build_mode_bar()
        self._build_status_bar()
        self._build_controls_bar()
        self._build_content()

        # Show which engine is active
        self.status_var.set(f"Ready [{self._engine_label}] — Click Start to begin")

        # Start UI polling loop
        self._schedule_ui_update()

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _count_saved_corrections(self):
        d = TRAINING_DIR / "images" / "train"
        return len(list(d.glob("correction_*.jpg"))) if d.exists() else 0

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

        # --- Normal controls (shown by default) ---
        self._normal_controls = tk.Frame(bar, bg=BG_BAR)
        self._normal_controls.pack(fill=tk.X)

        self.btn_start = tk.Button(
            self._normal_controls, text="Start", command=self._toggle_live,
            bg=GREEN, fg=FG_WHITE, activebackground=GREEN,
            font=("Helvetica", 11, "bold"), width=12,
        )
        self.btn_start.pack(side=tk.LEFT, padx=15, pady=8)

        self.btn_browse = tk.Button(
            self._normal_controls, text="Browse Image", command=self._browse_image,
            bg=BLUE, fg=FG_WHITE, activebackground=BLUE,
            font=("Helvetica", 11), width=14, state=tk.DISABLED,
        )
        self.btn_browse.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_save = tk.Button(
            self._normal_controls, text="Save Result", command=self._save_result,
            bg=ORANGE, fg=FG_WHITE, activebackground=ORANGE,
            font=("Helvetica", 11), width=12, state=tk.DISABLED,
        )
        self.btn_save.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_correct = tk.Button(
            self._normal_controls, text="Correct", command=self._enter_correction_mode,
            bg=PURPLE, fg=FG_WHITE, activebackground=PURPLE,
            font=("Helvetica", 11), width=10, state=tk.DISABLED,
        )
        self.btn_correct.pack(side=tk.LEFT, padx=5, pady=8)

        # --- Correction controls (hidden initially) ---
        self._correction_controls = tk.Frame(bar, bg=BG_BAR)

        tk.Button(
            self._correction_controls, text="Exit", command=self._exit_correction_mode,
            bg=INACTIVE, fg=FG_WHITE, font=("Helvetica", 11), width=8,
        ).pack(side=tk.LEFT, padx=(15, 5), pady=8)

        self.btn_set_adulterant = tk.Button(
            self._correction_controls, text="→ Adulterant",
            command=lambda: self._set_correction_class("adulterant"),
            bg="#CC2222", fg=FG_WHITE, font=("Helvetica", 11), width=12,
        )
        self.btn_set_adulterant.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_set_pepper = tk.Button(
            self._correction_controls, text="→ Pepper",
            command=lambda: self._set_correction_class("pepper"),
            bg="#007700", fg=FG_WHITE, font=("Helvetica", 11), width=10,
        )
        self.btn_set_pepper.pack(side=tk.LEFT, padx=5, pady=8)

        tk.Button(
            self._correction_controls, text="Save Correction",
            command=self._save_correction,
            bg=ORANGE, fg=FG_WHITE, font=("Helvetica", 11, "bold"), width=14,
        ).pack(side=tk.LEFT, padx=5, pady=8)

        tk.Label(
            self._correction_controls,
            text="Del = remove box | Drag = new box",
            bg=BG_BAR, fg=FG_DIM, font=("Helvetica", 9),
        ).pack(side=tk.LEFT, padx=10)

    def _build_content(self):
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: video / image canvas
        self.canvas = tk.Canvas(
            content, width=CANVAS_W, height=CANVAS_H,
            bg="black", highlightthickness=0,
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
            tk.Canvas(entry, width=16, height=16, bg=color_hex,
                      highlightthickness=1, highlightbackground="#555").pack(side=tk.LEFT, padx=(0, 8))
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

        # --- Active learning section ---
        ttk.Separator(stats).pack(fill=tk.X, padx=20, pady=5)

        self.lbl_corrections = self._stat_row(
            stats, "Corrections:", str(self._correction_count), "#BB86FC"
        )

        btn_row = tk.Frame(stats, bg=BG_DARK)
        btn_row.pack(fill=tk.X, padx=20, pady=(6, 4))
        self.btn_finetune = tk.Button(
            btn_row, text="Fine-tune Model",
            command=self._start_finetune,
            bg=PURPLE, fg=FG_WHITE, activebackground=PURPLE,
            font=("Helvetica", 11, "bold"),
            state=tk.NORMAL if self._correction_count >= FINETUNE_MIN_SAMPLES else tk.DISABLED,
        )
        self.btn_finetune.pack(fill=tk.X)

        tk.Label(
            stats, text=f"(need ≥{FINETUNE_MIN_SAMPLES} corrections to enable)",
            bg=BG_DARK, fg=FG_DIM, font=("Helvetica", 9),
        ).pack()

    def _stat_row(self, parent, label, value, fg="#FFFFFF"):
        row = tk.Frame(parent, bg=BG_DARK)
        row.pack(fill=tk.X, padx=20, pady=4)
        tk.Label(row, text=label, font=("Helvetica", 13),
                 bg=BG_DARK, fg="#cccccc").pack(side=tk.LEFT)
        val = tk.Label(row, text=value, font=("Helvetica", 16, "bold"),
                       bg=BG_DARK, fg=fg)
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
        self.btn_correct.configure(state=tk.NORMAL)
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
            # Draw yellow square border if detected
            roi = getattr(self.engine, "last_roi", None)
            if roi:
                cv2.rectangle(annotated, (roi[0], roi[1]), (roi[2], roi[3]),
                              (0, 255, 255), 2)

            stats = compute_stats(detections)
            stats["inference_ms"] = inference_ms

            fps_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_time = time.time()
            stats["fps"] = current_fps

            self._last_raw_frame = frame.copy()
            self._last_annotated = annotated
            self._last_detections = detections

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
        self._last_raw_frame = frame_bgr.copy()
        self._last_annotated = annotated
        self._last_detections = detections

        h, w = annotated.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(annotated, (new_w, new_h))
        padded = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        y_off = (CANVAS_H - new_h) // 2
        x_off = (CANVAS_W - new_w) // 2
        padded[y_off:y_off + new_h, x_off:x_off + new_w] = display
        self._update_canvas(padded)

        stats = compute_stats(detections)
        stats["inference_ms"] = inference_ms
        self._update_stats(stats)

        self.btn_save.configure(state=tk.NORMAL)
        self.btn_correct.configure(state=tk.NORMAL)
        self.status_var.set(
            f"Image: {Path(filepath).name} | "
            f"{stats['total_count']} seeds | {inference_ms:.0f}ms"
        )

    # -------------------------------------------------------------------
    # UI update loop (main thread)
    # -------------------------------------------------------------------
    def _schedule_ui_update(self):
        if not self._correction_mode:
            try:
                while not self.frame_queue.empty():
                    frame_bgr, stats = self.frame_queue.get_nowait()
                    self._update_canvas(frame_bgr)
                    self._update_stats(stats)
            except queue.Empty:
                pass
        self.root.after(50, self._schedule_ui_update)

    def _update_canvas(self, frame_bgr):
        photo, new_w, new_h = frame_to_photoimage(frame_bgr)
        self._photo = photo
        x_off = (CANVAS_W - new_w) // 2
        y_off = (CANVAS_H - new_h) // 2
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

        pct = stats["adulterant_pct"]
        color = "#00C800" if pct == 0 else ("#FFD700" if pct < 10 else "#FF4444")
        self.lbl_pct.configure(fg=color)

    # -------------------------------------------------------------------
    # Correction Mode
    # -------------------------------------------------------------------
    def _enter_correction_mode(self):
        if self._last_raw_frame is None:
            self.status_var.set("No frame to correct — start live detection first")
            return

        if self.running:
            self._stop_live()

        self._correction_mode = True
        self._correction_frame = self._last_raw_frame.copy()
        self._correction_boxes = [dict(d) for d in (self._last_detections or [])]
        self._selected_idx = None
        self._draw_start = None

        # Bind canvas events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Delete>", self._on_canvas_delete)
        self.canvas.bind("<BackSpace>", self._on_canvas_delete)
        self.canvas.focus_set()

        # Swap controls bar
        self._normal_controls.pack_forget()
        self._correction_controls.pack(fill=tk.X)

        self._redraw_correction_canvas()
        self.status_var.set(
            "Correction Mode — click box to select | drag empty area to add | Del to remove"
        )

    def _exit_correction_mode(self):
        self._correction_mode = False
        self._correction_frame = None
        self._correction_boxes = []
        self._selected_idx = None
        self._draw_start = None

        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.unbind("<Delete>")
        self.canvas.unbind("<BackSpace>")

        self._correction_controls.pack_forget()
        self._normal_controls.pack(fill=tk.X)
        self.status_var.set("Ready")

    def _canvas_to_frame(self, cx, cy):
        """Convert canvas pixel coords to lores frame pixel coords."""
        fw, fh = PREVIEW_RESOLUTION
        fx = max(0, min(fw - 1, int(cx)))
        fy = max(0, min(fh - 1, int(cy - CANVAS_Y_OFFSET)))
        return fx, fy

    def _hit_test(self, cx, cy):
        """Return index of correction box that contains canvas point (cx, cy), or None."""
        for i, b in enumerate(self._correction_boxes):
            bx1 = b["x1"] - 4
            by1 = b["y1"] + CANVAS_Y_OFFSET - 4
            bx2 = b["x2"] + 4
            by2 = b["y2"] + CANVAS_Y_OFFSET + 4
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                return i
        return None

    def _on_canvas_click(self, event):
        idx = self._hit_test(event.x, event.y)
        if idx is not None:
            self._selected_idx = idx
            self._draw_start = None
        else:
            self._selected_idx = None
            self._draw_start = (event.x, event.y)
        self._redraw_correction_canvas()

    def _on_canvas_drag(self, event):
        if self._draw_start is not None:
            self._redraw_correction_canvas(drag_end=(event.x, event.y))

    def _on_canvas_release(self, event):
        if self._draw_start is not None:
            sx, sy = self._draw_start
            ex, ey = event.x, event.y
            if abs(ex - sx) > 8 and abs(ey - sy) > 8:
                fx1, fy1 = self._canvas_to_frame(min(sx, ex), min(sy, ey))
                fx2, fy2 = self._canvas_to_frame(max(sx, ex), max(sy, ey))
                cls_id = CLASS_IDS.get(self._current_class, 0)
                self._correction_boxes.append({
                    "x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2,
                    "class_id": cls_id, "class_name": self._current_class,
                })
                self._selected_idx = len(self._correction_boxes) - 1
            self._draw_start = None
            self._redraw_correction_canvas()

    def _on_canvas_delete(self, event):
        if self._selected_idx is not None:
            del self._correction_boxes[self._selected_idx]
            self._selected_idx = None
            self._redraw_correction_canvas()

    def _set_correction_class(self, cls_name):
        self._current_class = cls_name
        if self._selected_idx is not None:
            b = self._correction_boxes[self._selected_idx]
            b["class_name"] = cls_name
            b["class_id"] = CLASS_IDS.get(cls_name, 0)
            self._redraw_correction_canvas()

    def _redraw_correction_canvas(self, drag_end=None):
        frame = self._correction_frame
        if frame is None:
            return

        display = frame.copy()
        for i, b in enumerate(self._correction_boxes):
            color = CLASS_COLORS.get(b["class_name"], (255, 255, 255))
            cv2.rectangle(display, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
            if i == self._selected_idx:
                # White selection outline
                cv2.rectangle(display,
                              (b["x1"] - 2, b["y1"] - 2),
                              (b["x2"] + 2, b["y2"] + 2),
                              (255, 255, 255), 1)

        photo, new_w, new_h = frame_to_photoimage(display)
        self._photo = photo
        x_off = (CANVAS_W - new_w) // 2
        y_off = (CANVAS_H - new_h) // 2
        self.canvas.delete("all")
        self.canvas.create_image(x_off, y_off, anchor=tk.NW, image=self._photo)

        # Rubber-band rect while dragging
        if drag_end is not None and self._draw_start is not None:
            outline = "#FF4444" if self._current_class == "adulterant" else "#00C800"
            self.canvas.create_rectangle(
                self._draw_start[0], self._draw_start[1],
                drag_end[0], drag_end[1],
                outline=outline, width=2, dash=(4, 4),
            )

    # -------------------------------------------------------------------
    # Save correction
    # -------------------------------------------------------------------
    def _save_correction(self):
        if self._correction_frame is None:
            return

        self._ensure_training_structure()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        img_path = TRAINING_DIR / "images" / "train" / f"correction_{ts}.jpg"
        lbl_path = TRAINING_DIR / "labels" / "train" / f"correction_{ts}.txt"

        cv2.imwrite(str(img_path), self._correction_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        fw, fh = PREVIEW_RESOLUTION
        lines = []
        for b in self._correction_boxes:
            xc = ((b["x1"] + b["x2"]) / 2) / fw
            yc = ((b["y1"] + b["y2"]) / 2) / fh
            bw = (b["x2"] - b["x1"]) / fw
            bh = (b["y2"] - b["y1"]) / fh
            lines.append(f"{b['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        lbl_path.write_text("\n".join(lines))

        self._correction_count += 1
        self.lbl_corrections.configure(text=str(self._correction_count))
        self._update_finetune_button()
        self.status_var.set(
            f"Saved correction #{self._correction_count} ({img_path.name}) — "
            f"{len(self._correction_boxes)} boxes"
        )
        self._exit_correction_mode()

    def _ensure_training_structure(self):
        for subdir in ["images/train", "labels/train"]:
            (TRAINING_DIR / subdir).mkdir(parents=True, exist_ok=True)
        data_yaml = TRAINING_DIR / "data.yaml"
        if not data_yaml.exists():
            data_yaml.write_text(
                f"path: {TRAINING_DIR}\n"
                "train: images/train\n"
                "val: images/train\n"
                "nc: 2\n"
                "names:\n"
                "  0: adulterant\n"
                "  1: pepper\n"
            )

    def _update_finetune_button(self):
        enabled = self._correction_count >= FINETUNE_MIN_SAMPLES and not self._finetuning
        self.btn_finetune.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    # -------------------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------------------
    def _start_finetune(self):
        if self._finetuning:
            return
        if self.running:
            self._stop_live()
        self._finetuning = True
        self.btn_finetune.configure(state=tk.DISABLED, text="Fine-tuning...")
        self.status_var.set(
            f"Fine-tuning started ({self._correction_count} corrections) — ~10-15 min..."
        )
        threading.Thread(target=self._run_finetune_thread, daemon=True).start()

    def _run_finetune_thread(self):
        try:
            from ultralytics import YOLO
            model = YOLO(str(PT_MODEL_PATH))
            model.train(
                data=str(TRAINING_DIR / "data.yaml"),
                epochs=30,
                imgsz=640,
                freeze=10,
                batch=4,
                device="cpu",
                project=str(MODEL_DIR / "finetune_run"),
                name="run",
                exist_ok=True,
                verbose=False,
            )
            best = MODEL_DIR / "finetune_run" / "run" / "weights" / "best.pt"
            if best.exists():
                shutil.copy(best, PT_MODEL_PATH)
                self.root.after(0, lambda: self.status_var.set(
                    "Fine-tuning complete — reloading model..."
                ))
                if hasattr(self.engine, "reload"):
                    self.engine.reload()
                self.root.after(0, lambda: self.status_var.set(
                    "Fine-tuning complete — model updated! Restart detection to use it."
                ))
            else:
                self.root.after(0, lambda: self.status_var.set(
                    "Fine-tuning done but best.pt not found — check finetune_run/"
                ))
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: self.status_var.set(f"Fine-tuning error: {msg}"))
        finally:
            self._finetuning = False
            self.root.after(0, self._update_finetune_button)
            self.root.after(0, lambda: self.btn_finetune.configure(text="Fine-tune Model"))

    # -------------------------------------------------------------------
    # Save detection result
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
