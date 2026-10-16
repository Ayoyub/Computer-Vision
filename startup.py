"""
startup.py — startup calibration and hand-tuning screen
═══════════════════════════════════════════════════════════════════════════════════
Shows a camera feed with real-time gesture feedback before the main app starts.

Controls:
  SPACE — run calibration (measure hand scale + benchmark detection)
          or launch the main app after calibration is done
  ↑/↓   — adjust fist_min_folded (2=very sensitive, 3=default, 4=less sensitive)
  ←/→   — adjust pinch ratio on/off
  ESC   — quit
"""
import math
import sys
import time

import cv2
import mediapipe as mp

from config import HAND, GESTURE, CAM
from kalman_filter import LandmarkSmoother
import hand_detect as hand_detect_mod
from hand_detect import _benchmark_latency, _landmarker


# ── Tuning params (editable with arrows) ────────────────────────────────────────
_ADJUSTABLE = {
    'fist_min_folded': {'default': 3, 'lo': 2, 'hi': 4, 'step': 1},
    'pinch_ratio_on':  {'default': 0.25, 'lo': 0.10, 'hi': 0.45, 'step': 0.01},
    'pinch_ratio_off': {'default': 0.40, 'lo': 0.25, 'hi': 0.60, 'step': 0.01},
}


class _StartupState:
    def __init__(self):
        self.fist_min_folded = HAND['fist_min_folded']
        self.pinch_ratio_on = HAND['pinch_ratio_on']
        self.pinch_ratio_off = HAND['pinch_ratio_off']
        self.calibrated = False
        self.hand_scale: float | None = None
        self.benchmark_ms: float | None = None
        self.benchmark_res: str | None = None
        self.prev_key: int | None = None  # key debounce


def _est_poing_brut(pts, fold_scale, tip_mid_ratio, min_folded):
    wrist = pts[0]
    fingers = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]
    folded = 0
    for tip, mid, base in fingers:
        d_tip_wrist = math.hypot(pts[tip][0] - wrist[0], pts[tip][1] - wrist[1])
        d_mid_wrist = math.hypot(pts[mid][0] - wrist[0], pts[mid][1] - wrist[1])
        d_tip_base = math.hypot(pts[tip][0] - pts[base][0],
                                pts[tip][1] - pts[base][1])
        if (d_tip_wrist < d_mid_wrist * tip_mid_ratio
                and d_tip_base < d_mid_wrist * fold_scale):
            folded += 1
    return folded >= min_folded


def _est_pincement_brut(pts, pinch_on, pinch_off):
    scale = max(math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]), 1.0)
    ratio = math.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1]) / scale
    if ratio < pinch_on:
        return True
    elif ratio > pinch_off:
        return False
    return False


def _draw_feedback(img, state, hand_data):
    h, w = img.shape[:2]
    font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.32

    # ── Top-left panel: status + controls ───────────────────────────────────────
    info_lines = [
        "Vision AI — Startup Calibration",
        "",
        (f"Hand scale: {state.hand_scale:.1f} px"
         if state.hand_scale else "Hand scale: —"),
        (f"Detect: {state.benchmark_res}"
         if state.benchmark_res else "Detect: —"),
        (f"Bench:  {state.benchmark_ms:.1f} ms"
         if state.benchmark_ms else "Bench:  —"),
        "",
        "[SPACE] calibrate / launch",
        "",
        (f"↑/↓ fist_min_folded:  {state.fist_min_folded}  "
         "(2=easy, 3=default, 4=strict)"),
        (f"←/→ pinch on:         {state.pinch_ratio_on:.2f}  "
         "(lower=more sensitive)"),
        (f"←/→ pinch off:        {state.pinch_ratio_off:.2f}"),
        "",
        "[ESC] quit",
    ]
    for i, line in enumerate(info_lines):
        (tw, _), _ = cv2.getTextSize(line, font, sc, 1)
        cv2.putText(img, line, (10, 16 + i * 14), font, sc,
                    (160, 160, 160), 1, cv2.LINE_AA)

    # ── Bottom-center: real-time gesture feedback ──────────────────────────────
    if hand_data:
        pts = hand_data['points']
        poing = _est_poing_brut(
            pts, fold_scale=HAND['fist_fold_scale'],
            tip_mid_ratio=HAND['fist_tip_mid_ratio'],
            min_folded=state.fist_min_folded)
        pinch = _est_pincement_brut(
            pts, state.pinch_ratio_on, state.pinch_ratio_off)

        fy = h - 80
        hints = []
        if poing:
            hints.append(("FIST ✓", (80, 255, 80)))
        else:
            hints.append(("FIST ✗", (120, 60, 60)))
        if pinch:
            hints.append(("PINCH ✓", (80, 255, 160)))
        else:
            hints.append(("PINCH ✗", (60, 120, 60)))

        total_w = (sum(cv2.getTextSize(h[0], font, 0.38, 1)[0][0] + 12
                       for h in hints) + 16)
        xs = max(8, (w - total_w) // 2)
        for label, color in hints:
            (tw, th), _ = cv2.getTextSize(label, font, 0.38, 1)
            cv2.putText(img, label, (xs, fy), font, 0.38, color, 1, cv2.LINE_AA)
            xs += tw + 16

        # Landmarks (white = fingertips, cyan = rest)
        for i, raw_pt in enumerate(pts[:21]):
            px, py = int(raw_pt[0]), int(raw_pt[1])
            c = (255, 255, 255) if i in {4, 8, 12, 16, 20} else (0, 200, 255)
            cv2.circle(img, (px, py), 3, c, -1, cv2.LINE_AA)
        for a, b in hand_detect_mod._CONNECTIONS:
            pa = pts[a]; pb = pts[b]
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     (0, 150, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(img,
                    "No hand detected — hold your hand in front of the camera",
                    (w // 2 - 200, h - 40), font, 0.36, (100, 100, 100), 1,
                    cv2.LINE_AA)

    # ── Done indicator ─────────────────────────────────────────────────────────
    if state.calibrated:
        font2 = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "✓ CALIBRATED — press SPACE to launch",
                    (w // 2 - 180, 30), font2, 0.5, (80, 255, 120), 2,
                    cv2.LINE_AA)


def run_startup() -> bool:
    """
    Show the startup calibration screen.
    Returns True if the user confirmed and wants to launch the main app.
    """
    cap = cv2.VideoCapture(CAM['source'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    DISP_W, DISP_H = 640, 360
    state = _StartupState()
    smoother = LandmarkSmoother()

    print("Startup: hold your hand in front of the camera. Press SPACE to calibrate.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (DISP_W, DISP_H))

        # Detect hands
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        result = _landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        hand_data = None
        if result.hand_landmarks:
            raw = [(lm.x * DISP_W, lm.y * DISP_H) for lm in result.hand_landmarks[0]]
            smoothed = smoother.smooth(raw, time.monotonic())
            hand_data = {'points': smoothed}

        _draw_feedback(small, state, hand_data)
        cv2.imshow("Vision AI — Calibration", small)
        key = cv2.waitKey(30) & 0xFF

        # ── ESC → quit ─────────────────────────────────────────────────────────
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return False

        # ── SPACE → calibrate (first press) or launch (second press) ───────────
        elif key == 32 and state.prev_key != 32:
            state.prev_key = 32
            if not state.calibrated:
                _run_calibration(state, cap)
            else:
                # Apply tuned params and launch
                HAND['fist_min_folded'] = state.fist_min_folded
                HAND['pinch_ratio_on'] = state.pinch_ratio_on
                HAND['pinch_ratio_off'] = state.pinch_ratio_off
                cap.release()
                cv2.destroyAllWindows()
                print("Launching main app…")
                return True

        # ── Arrow keys → adjust params ─────────────────────────────────────────
        elif key == 265 and state.prev_key != 265:   # UP
            state.prev_key = 265
            if state.fist_min_folded < _ADJUSTABLE['fist_min_folded']['hi']:
                state.fist_min_folded += 1
                print(f"fist_min_folded → {state.fist_min_folded}")
        elif key == 266 and state.prev_key != 266:   # DOWN
            state.prev_key = 266
            if state.fist_min_folded > _ADJUSTABLE['fist_min_folded']['lo']:
                state.fist_min_folded -= 1
                print(f"fist_min_folded → {state.fist_min_folded}")
        elif key == 263 and state.prev_key != 263:   # LEFT
            state.prev_key = 263
            step = _ADJUSTABLE['pinch_ratio_on']['step']
            if state.pinch_ratio_on > _ADJUSTABLE['pinch_ratio_on']['lo']:
                state.pinch_ratio_on = round(state.pinch_ratio_on - step, 2)
            if state.pinch_ratio_off > _ADJUSTABLE['pinch_ratio_off']['lo']:
                state.pinch_ratio_off = round(state.pinch_ratio_off - step, 2)
            print(f"pinch on={state.pinch_ratio_on}, off={state.pinch_ratio_off}")
        elif key == 262 and state.prev_key != 262:   # RIGHT
            state.prev_key = 262
            step = _ADJUSTABLE['pinch_ratio_on']['step']
            if state.pinch_ratio_on < _ADJUSTABLE['pinch_ratio_on']['hi']:
                state.pinch_ratio_on = round(state.pinch_ratio_on + step, 2)
            if state.pinch_ratio_off < _ADJUSTABLE['pinch_ratio_off']['hi']:
                state.pinch_ratio_off = round(state.pinch_ratio_off + step, 2)
            print(f"pinch on={state.pinch_ratio_on}, off={state.pinch_ratio_off}")

    cap.release()
    cv2.destroyAllWindows()
    return False


def _run_calibration(state, cap):
    """Run hand scale measurement and detection benchmark."""
    print("Calibrating — hold still…")

    # ── Measure hand scale ──────────────────────────────────────────────────────
    scales = []
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    for _ in range(30):
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        r = _landmarker.detect_for_video(mp_img, int(time.time() * 1000))
        if r.hand_landmarks:
            lm = r.hand_landmarks[0]
            dx = lm[9].x - lm[0].x
            dy = lm[9].y - lm[0].y
            s = math.hypot(dx, dy)
            if s > 1.0:
                scales.append(s)

    if scales:
        scales.sort()
        ref = scales[len(scales) // 2]
        hand_detect_mod.HAND_CALIB['reference_scale_px'] = ref
        state.hand_scale = ref
        print(f"  Hand scale: {ref:.1f} px")
    else:
        print("  No hand detected during calibration")

    # ── Benchmark detection ─────────────────────────────────────────────────────
    try:
        hi_ms = _benchmark_latency(_landmarker, 1280, 720, CAM['source'], samples=4)
        base_ms = _benchmark_latency(_landmarker, 640, 360, CAM['source'], samples=4)
        budget = CAM.get('detect_latency_budget_ms', 25)
        if hi_ms <= budget:
            state.benchmark_res = f"1280x720 ({hi_ms:.1f}ms)"
            state.benchmark_ms = hi_ms
        else:
            state.benchmark_res = f"640x360 ({base_ms:.1f}ms)"
            state.benchmark_ms = base_ms
        print(f"  Benchmark: hi={hi_ms:.1f}ms base={base_ms:.1f}ms budget={budget}ms")
    except Exception as e:
        state.benchmark_res = "error"
        state.benchmark_ms = None
        print(f"  Benchmark failed: {e}")

    state.calibrated = True
    print("Calibration done. Adjust params with arrows, then SPACE to launch.")
