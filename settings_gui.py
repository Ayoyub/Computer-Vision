"""
settings_gui.py — tkinter settings GUI (no new dependencies)
══════════════════════════════════════════════════════════════
Configure:
  • Camera device (enumerated, with live preview)
  • Camera frame rate
  • Display resolution
  • Detection resolution  — fixed OR auto-quality with latency budget
  • One Euro smoothing (jitter/lag tradeoff)

Saves to the same settings file that config.py reads on import,
so the rest of the app picks the values up automatically.

Dark theme, TD-style (matches the app's aesthetic).
"""
import threading
import time

import cv2
import tkinter as tk
from tkinter import ttk

import settings as st
from config import CAM, SMOOTHING

# ── Theme ─────────────────────────────────────────────────────────────────────
BG       = "#0d0f14"
PANEL    = "#14171f"
BORDER   = "#2a2f3c"
TEXT     = "#c8d0e0"
TEXT_DIM = "#6b7488"
ACCENT_D = "#2e5c3f"

# Display resolution presets: (label, w, h)
DISPLAY_PRESETS = [
    ("720p  — 1280 × 720",  1280, 720),
    ("1080p — 1920 × 1080", 1920, 1080),
    ("900p  — 1600 × 900",  1600, 900),
    ("480p  — 854 × 480",   854, 480),
]

# Detection resolution presets: (label, w, h)
DETECT_PRESETS = [
    ("360p — 640 × 360   (fast)",    640, 360),
    ("540p — 960 × 540   (mid)",    960, 540),
    ("720p — 1280 × 720  (fine)",  1280, 720),
]

CAM_FPS_CHOICES = [15, 24, 30, 60]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CAMERA DISCOVERY + PREVIEW                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def list_cameras(max_idx: int = 8) -> list:
    """Probe camera indices 0..max_idx-1. Returns [index, ...] that deliver frames."""
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()  # some devices open but can't deliver frames
        cap.release()
        time.sleep(0.05)  # let the device settle before the next probe
        if ok:
            found.append(i)
    return found


class CameraPreview:
    """Background thread feeding a downscaled frame to a tk.Canvas."""

    PREVIEW_W = 400
    PREVIEW_H = 225  # 16:9

    def __init__(self, root, canvas, photo_item):
        self._root = root
        self._canvas = canvas
        self._photo_item = photo_item
        self._thread = None
        self._running = False
        self._stop_evt = threading.Event()
        self._cap = None
        self._cap_lock = threading.Lock()

    def set_camera(self, idx, fps):
        """(Re)start the preview for camera idx. Safe to call any time."""
        self.stop_camera()
        self._stop_evt.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(idx, fps),
                                        daemon=True)
        self._thread.start()

    def stop_camera(self):
        self._running = False
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._thread = None
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    @staticmethod
    def _open_camera(idx, fps):
        """Open the camera with retries.

        On Linux/V4L2 the device may not be fully released right after the
        probe loop (or a previous capture), so the first open/read often
        fails. Retry a few times with a settle delay in between.
        Returns a working VideoCapture or None.
        """
        for _attempt in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraPreview.PREVIEW_W * 2)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraPreview.PREVIEW_H * 2)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Prime the pipeline — the first frames after open are often
                # undeliverable on V4L2 devices.
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
            if _attempt < 3:
                time.sleep(0.25)
        return None

    def _run(self, idx, fps):
        cap = self._open_camera(idx, fps)
        if cap is None:
            self._running = False
            return
        with self._cap_lock:
            self._cap = cap

        while self._running and not self._stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                if self._stop_evt.wait(0.2):
                    break
                continue
            small = cv2.resize(frame, (self.PREVIEW_W, self.PREVIEW_H),
                               interpolation=cv2.INTER_AREA)
            small = cv2.flip(small, 1)  # mirror, same as the app
            # Encode as PNG — PhotoImage(data=...) needs an image-file
            # format; raw RGB bytes are not a valid input.
            ok_enc, png = cv2.imencode(".png", small)
            if not ok_enc:
                continue
            row = png.tobytes()
            # Tk must be touched from the main thread only
            self._root.after(0, self._push, row)

        with self._cap_lock:
            cap.release()
            if self._cap is cap:
                self._cap = None

    def _push(self, png_bytes):
        img = tk.PhotoImage(data=png_bytes)
        self._canvas.itemconfig(self._photo_item, image=img)
        self._photo_ref = img  # prevent GC


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  GUI                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class SettingsGUI:
    def __init__(self, root):
        self.root = root
        self._saved = False
        root.title("Vision AI — Settings")
        root.configure(bg=BG)
        root.resizable(False, False)

        self._cam_options = []   # indices found

        self._build_styles()
        self._build_widgets()
        self.preview = CameraPreview(root, self._canvas, self._photo_item)
        self._load_values()
        self._start_preview()

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL,
                    bordercolor=BORDER, lightcolor=PANEL, darkcolor=PANEL,
                    troughcolor=PANEL)
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("Dim.TLabel", background=BG, foreground=TEXT_DIM)
        s.configure("Accent.TLabel", background=BG, foreground="#72f299",
                    font=("DejaVu Sans", 13, "bold"))
        s.configure("TCheckbutton", background=BG, foreground=TEXT)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("TRadiobutton", background=BG, foreground=TEXT)
        s.map("TRadiobutton", background=[("active", BG)])
        s.configure("TButton", background=PANEL, foreground=TEXT,
                    padding=(14, 6), font=("DejaVu Sans", 10))
        s.map("TButton", background=[("active", ACCENT_D)])
        s.configure("Accent.TButton", background=ACCENT_D, foreground="#72f299")
        s.map("Accent.TButton", background=[("active", ACCENT_D)])
        s.configure("TSeparator", background=BORDER)
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL,
                    foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)])

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_widgets(self):
        root = self.root
        pad = dict(padx=10, pady=4)

        ttk.Label(root, text="Vision AI — Settings",
                  style="Accent.TLabel").pack(pady=(14, 2))
        ttk.Label(root, text=f"saved to  {st.settings_path()}",
                  style="Dim.TLabel").pack()

        body = ttk.Frame(root)
        body.pack(fill="x", padx=10, pady=8)

        # ── Left column: camera ───────────────────────────────────────────────
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Camera", style="Dim.TLabel").pack(anchor="w", **pad)

        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(left, textvariable=self.cam_var,
                                      state="readonly", width=16)
        self.cam_combo.pack(anchor="w", **pad)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_cam_changed)

        ttk.Label(left, text="Camera frame rate", style="Dim.TLabel").pack(anchor="w", **pad)
        self.cam_fps_var = tk.StringVar()
        self.cam_fps_combo = ttk.Combobox(left, textvariable=self.cam_fps_var,
                                          values=[str(f) for f in CAM_FPS_CHOICES],
                                          state="readonly", width=16)
        self.cam_fps_combo.pack(anchor="w", **pad)
        self.cam_fps_combo.bind("<<ComboboxSelected>>", self._on_cam_changed)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(left, text="Display resolution", style="Dim.TLabel").pack(anchor="w", **pad)
        self.display_var = tk.StringVar()
        self.display_combo = ttk.Combobox(left, textvariable=self.display_var,
                                          values=[p[0] for p in DISPLAY_PRESETS],
                                          state="readonly", width=16)
        self.display_combo.pack(anchor="w", **pad)

        # ── Right column: preview + detection + smoothing ────────────────────
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))

        # Live preview canvas
        self._canvas = tk.Canvas(right, width=CameraPreview.PREVIEW_W,
                                 height=CameraPreview.PREVIEW_H,
                                 bg="#000000", highlightthickness=1,
                                 highlightbackground=BORDER)
        self._canvas.pack(anchor="w")
        self._photo_item = self._canvas.create_image(
            0, 0, anchor="nw", image=tk.PhotoImage(
                width=CameraPreview.PREVIEW_W, height=CameraPreview.PREVIEW_H))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)

        # Detection resolution
        ttk.Label(right, text="Hand detection", style="Dim.TLabel").pack(anchor="w", **pad)

        self.detect_auto_var = tk.BooleanVar()
        self.auto_chk = ttk.Checkbutton(
            right, text="Auto quality — pick finest resolution within latency budget",
            variable=self.detect_auto_var)
        self.auto_chk.pack(anchor="w")
        self.auto_chk.bind("<ButtonRelease-1>", self._on_detect_mode)

        # Fixed-resolution row
        self.fixed_frame = ttk.Frame(right)
        self.fixed_frame.pack(anchor="w")
        self.detect_res_var = tk.StringVar()
        self.detect_res_combo = ttk.Combobox(
            self.fixed_frame, textvariable=self.detect_res_var,
            values=[p[0] for p in DETECT_PRESETS], state="readonly", width=32)
        self.detect_res_combo.pack(side="left")

        # Auto-quality budget row
        self.budget_frame = ttk.Frame(right)
        self.budget_frame.pack(anchor="w")
        ttk.Label(self.budget_frame, text="latency budget  ").pack(side="left")
        self.budget_var = tk.IntVar()
        self.budget_scale = ttk.Scale(self.budget_frame, from_=10, to=60,
                                      variable=self.budget_var,
                                      orient="horizontal", length=200)
        self.budget_scale.pack(side="left")
        self.budget_lbl = ttk.Label(self.budget_frame, text="",
                                    style="Dim.TLabel", width=8)
        self.budget_lbl.pack(side="left")
        self.budget_var.trace_add("write", self._on_budget)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)

        # Smoothing
        ttk.Label(right, text="Smoothing (One Euro filter)",
                  style="Dim.TLabel").pack(anchor="w", **pad)
        row1 = self._build_slider_row(right, "jitter (min cut)",
                                      0.3, 4.0, 1.0)
        self.min_cut_var = row1
        row2 = self._build_slider_row(right, "lag (beta)", 0.0, 0.5, 0.02)
        self.beta_var = row2

        # ── Bottom bar: actions ───────────────────────────────────────────────
        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=10, pady=(0, 12))
        ttk.Button(bar, text="Reset defaults",
                   command=self._on_reset).pack(side="left")
        ttk.Button(bar, text="Cancel", command=self._on_close).pack(side="right", padx=8)
        ttk.Button(bar, text="Save", style="Accent.TButton",
                   command=self._on_save).pack(side="right")

    def _build_slider_row(self, parent, label, lo, hi, init):
        """Returns a DoubleVar; builds label + scale + live value readout."""
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=16, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=init)
        sc = ttk.Scale(f, from_=lo, to=hi, variable=var,
                       orient="horizontal", length=260)
        sc.pack(side="left", fill="x", expand=True)
        val = ttk.Label(f, text=f"{init:.2f}", style="Dim.TLabel", width=8)
        val.pack(side="left", padx=6)

        def _upd(*_a):
            v = var.get()
            val.config(text=f"{v:.3f}" if v < 1 else f"{v:.1f}")
        var.trace_add("write", _upd)
        return var

    # ── Value loading ─────────────────────────────────────────────────────────

    def _load_values(self):
        self.cam_fps_var.set(str(CAM['camera_fps']))
        self.display_var.set(self._preset_label(DISPLAY_PRESETS,
                                                CAM['display_w'], CAM['display_h']))
        self.detect_res_var.set(self._preset_label(DETECT_PRESETS,
                                                   CAM['detect_w'], CAM['detect_h']))
        self.detect_auto_var.set(bool(CAM.get('detect_auto', True)))
        self.budget_var.set(int(CAM.get('detect_latency_budget_ms', 25)))
        self._budget_lbl_set()
        self.min_cut_var.set(SMOOTHING['min_cutoff'])
        self.beta_var.set(SMOOTHING['beta'])
        self._on_detect_mode()

    @staticmethod
    def _preset_label(presets, w, h):
        for label, pw, ph in presets:
            if pw == w and ph == h:
                return label
        return presets[0][0]

    def _start_preview(self):
        found = list_cameras()
        self._cam_options = found or [0]
        self.cam_combo["values"] = [f"Camera {i}" for i in self._cam_options]
        # Select the current config's camera if available, else the first
        cur = int(CAM['source'])
        idx = self._cam_options.index(cur) if cur in self._cam_options else 0
        self.cam_combo.current(idx)
        self._update_preview()

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_cam_changed(self, _event=None):
        self._update_preview()

    def _update_preview(self):
        try:
            idx = int(self.cam_combo.get().split()[-1])
            fps = int(self.cam_fps_var.get() or 30)
        except (ValueError, IndexError):
            return
        self.preview.set_camera(idx, fps)

    def _on_detect_mode(self, _event=None):
        if self.detect_auto_var.get():
            self.detect_res_combo.state(["disabled"])
            for w in self.budget_frame.winfo_children():
                w.state(["!disabled"])
        else:
            self.detect_res_combo.state(["!disabled"])
            for w in self.budget_frame.winfo_children():
                w.state(["disabled"])

    def _on_budget(self, *args):
        self._budget_lbl_set()

    def _budget_lbl_set(self):
        self.budget_lbl.config(text=f"{int(self.budget_var.get())} ms")

    def _gather(self):
        idx = int(self.cam_combo.get().split()[-1])
        fps = int(self.cam_fps_var.get() or 30)
        dw, dh = self._preset_values(DISPLAY_PRESETS, self.display_var.get())
        if self.detect_auto_var.get():
            detect = {
                'detect_auto': True,
                'detect_latency_budget_ms': int(self.budget_var.get()),
            }
        else:
            fixed_w, fixed_h = self._preset_values(
                DETECT_PRESETS, self.detect_res_var.get())
            detect = {
                'detect_auto': False,
                'detect_w': fixed_w,
                'detect_h': fixed_h,
            }
        cam = {
            'source': idx,
            'camera_fps': fps,
            'display_w': dw,
            'display_h': dh,
            **detect,
        }
        smoothing = {
            'min_cutoff': round(self.min_cut_var.get(), 2),
            'beta': round(self.beta_var.get(), 3),
            'd_cutoff': SMOOTHING['d_cutoff'],
            'jump_ratio': SMOOTHING['jump_ratio'],
        }
        return {'cam': cam, 'smoothing': smoothing}

    @staticmethod
    def _preset_values(presets, label):
        for l, w, h in presets:
            if l == label:
                return w, h
        return presets[0][1], presets[0][2]

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_save(self):
        path = st.save(self._gather())
        self._saved = True
        print(f"[settings] saved → {path}")
        self._close()

    def _on_reset(self):
        st.reset()
        print("[settings] user settings reset — defaults restored")
        # Re-open the GUI on the pure defaults
        self._close()
        self._saved = open_settings()

    def _on_close(self):
        self._close()

    def _close(self):
        self.preview.stop_camera()
        self.root.destroy()

    @property
    def saved(self):
        return self._saved


def open_settings() -> bool:
    """Show the settings GUI. Returns True when the user wants to run the app."""
    root = tk.Tk()
    gui = SettingsGUI(root)
    root.mainloop()
    return gui.saved
