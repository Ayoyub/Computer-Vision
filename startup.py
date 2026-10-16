"""
startup.py — startup calibration and hand-tuning screen
═══════════════════════════════════════════════════════════════════════════════════
Uses the exact same TD pipeline as the main app:
  • camera dim + background tint
  • emission skeleton with Gaussian bloom
  • fingertip state indicators
  • semi-transparent HUD panel (top-left)
  • gesture hints (bottom-center)
  • large readable text

Controls:
  SPACE — run calibration / launch after calibration
  ↑/↓   — adjust fist_min_folded
  ←/→   — adjust pinch ratio on/off
  ESC   — quit
"""
import math
import sys
import time

import cv2
import numpy as np
import mediapipe as mp

from config import HAND, GESTURE, CAM
from kalman_filter import LandmarkSmoother
import hand_detect as hand_detect_mod
from hand_detect import _benchmark_latency, _landmarker
import td_render as td

# Import the same skeleton topology + palm set that hand_detect uses
_CONNECTIONS = hand_detect_mod._CONNECTIONS
_FINGERTIPS  = {4, 8, 12, 16, 20}
_PALM_JOINTS = {0, 1, 5, 9, 13, 17}


# ── Tuning params ───────────────────────────────────────────────────────────────
_ADJUSTABLE = {
    'fist_min_folded': {'lo': 2, 'hi': 4},
    'pinch_ratio_on':  {'lo': 0.10, 'hi': 0.45, 'step': 0.01},
    'pinch_ratio_off': {'lo': 0.25, 'hi': 0.60, 'step': 0.01},
}


class _StartupState:
    def __init__(self):
        self.fist_min_folded = HAND['fist_min_folded']
        self.pinch_ratio_on  = HAND['pinch_ratio_on']
        self.pinch_ratio_off = HAND['pinch_ratio_off']
        self.calibrated: bool = False
        self.hand_scale: float | None = None
        self.benchmark_ms: float | None = None
        self.benchmark_res: str | None = None
        self.prev_key: int | None = None


# ── Gesture estimation (same logic as hand_detect) ─────────────────────────────

def _est_poing(pts, fold_scale, tip_mid_ratio, min_folded) -> bool:
    wrist = pts[0]
    folded = 0
    for tip, mid, base in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
        d_tw = math.hypot(pts[tip][0] - wrist[0], pts[tip][1] - wrist[1])
        d_mw = math.hypot(pts[mid][0] - wrist[0], pts[mid][1] - wrist[1])
        d_tb = math.hypot(pts[tip][0] - pts[base][0], pts[tip][1] - pts[base][1])
        if d_tw < d_mw * tip_mid_ratio and d_tb < d_mw * fold_scale:
            folded += 1
    return folded >= min_folded


def _est_pinch(pts, pinch_on, pinch_off) -> bool:
    scale = max(math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]), 1.0)
    ratio = math.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1]) / scale
    if ratio < pinch_on:
        return True
    elif ratio > pinch_off:
        return False
    return False


# ── Full TD pipeline (identical to hand_detect._draw_hand_td + main.py HUD) ────

def _draw_feedback(img, state, pts, poing, pinch, hand_count, fps):
    """
    Apply the complete pipeline:
      1. Dim + tint camera
      2. Draw skeleton on float32 emission layer (TD style)
      3. Gaussian bloom
      4. Additive composite
      5. Draw HUD overlay (top-left) + gesture hints (bottom-center)
    """
    h, w = img.shape[:2]

    # 1. Dim camera (same as hand_detect)
    img_f = img.astype(np.float32) / 255.0
    img_f = img_f * td.CAM_DIM + td.BG_TINT
    img[:] = np.clip(img_f * 255, 0, 255).astype(np.uint8)

    # 2. Emission layer for hand skeleton
    em = td.make_emission(h, w)
    if pts is not None:
        pts_int = [(int(round(x)), int(round(y))) for x, y in pts]

        # Skeleton lines (same logic as _draw_hand_td in hand_detect)
        for s, e in _CONNECTIONS:
            is_palm = (s in _PALM_JOINTS and e in _PALM_JOINTS)
            color = td.TD_BLUE if (is_palm or poing) else td.TD_CYAN
            td.draw_line_em(em, pts_int[s], pts_int[e],
                            color=(color * (0.5 if poing else 0.8)))

        # Joints
        for i, pt in enumerate(pts_int):
            if i in _FINGERTIPS:
                td.draw_node_em(em, pt, r=3,
                                core=td.TD_WHITE * (0.3 if poing else 1.0),
                                rim=td.TD_CYAN * (0.3 if poing else 0.9))
            elif i in _PALM_JOINTS:
                td.draw_node_em(em, pt, r=2,
                                core=td.TD_WHITE * 0.4,
                                rim=td.TD_BLUE)
            else:
                td.draw_node_em(em, pt, r=1,
                                core=td.TD_WHITE * 0.3,
                                rim=td.TD_BLUE * 0.6)

        # Fingertip state indicator (at index tip, same as hand_detect)
        td.draw_fingertip_em(em, pts_int[8], pinch=pinch, fist=poing)

    # 3. Bloom + additive composite (same as hand_detect)
    bloomed = td.bloom(em, kernel=15, sigma=6.0, strength=0.9)
    td.add_emission(img, bloomed)

    # 4. HUD — top-left panel (same style as _draw_fps in main.py)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lh = 26
    pad = 12
    nw, nh = 480, lh * 12 + pad * 2 + 20

    ov = img.copy()
    cv2.rectangle(ov, (pad, pad), (pad + nw, pad + nh), (10, 10, 10), -1)
    cv2.rectangle(ov, (pad, pad), (pad + nw, pad + nh), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)

    # Title
    cv2.putText(img, "VISION AI — CALIBRATION",
                (pad + 10, pad + 28), font, 0.50, (160, 180, 255), 1, cv2.LINE_AA)

    # Separator
    sep_y = pad + 44
    cv2.line(img, (pad + 8, sep_y), (pad + nw - 8, sep_y), (60, 60, 60), 1, cv2.LINE_AA)
    ty = sep_y + lh

    # Hand status
    hand_col = (100, 255, 100) if hand_count > 0 else (180, 80, 80)
    cv2.putText(img, f"hands  :  {hand_count}", (pad + 10, ty),
                font, 0.48, hand_col, 1, cv2.LINE_AA)
    ty += lh

    # Fist status
    fist_col = (80, 255, 80) if poing else (120, 120, 120)
    fist_label = "  ✓ detected" if poing else "  ✗ not detected"
    cv2.putText(img, f"fist   :{fist_label}", (pad + 10, ty),
                font, 0.48, fist_col, 1, cv2.LINE_AA)
    ty += lh

    # Pinch status
    pinch_col = (80, 255, 160) if pinch else (120, 120, 120)
    pinch_label = "  ✓ detected" if pinch else "  ✗ not detected"
    cv2.putText(img, f"pinch  :{pinch_label}", (pad + 10, ty),
                font, 0.48, pinch_col, 1, cv2.LINE_AA)
    ty += lh

    # Calibration status
    if state.hand_scale:
        cv2.putText(img, f"scale  :  {state.hand_scale:.1f} px",
                    (pad + 10, ty), font, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "scale  :  —",
                    (pad + 10, ty), font, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    ty += lh

    if state.benchmark_ms:
        cv2.putText(img, f"bench  :  {state.benchmark_res}",
                    (pad + 10, ty), font, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "bench  :  —",
                    (pad + 10, ty), font, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    ty += lh

    # Separator
    cv2.line(img, (pad + 8, ty), (pad + nw - 8, ty), (60, 60, 60), 1, cv2.LINE_AA)
    ty += lh

    # Tunable params
    cv2.putText(img, f"↑/↓  fist_min_folded = {state.fist_min_folded}",
                (pad + 10, ty), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    ty += lh
    cv2.putText(img, f"←/→  pinch_on  = {state.pinch_ratio_on:.2f}",
                (pad + 10, ty), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    ty += lh
    cv2.putText(img, f"←/→  pinch_off = {state.pinch_ratio_off:.2f}",
                (pad + 10, ty), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    ty += lh

    # Separator
    cv2.line(img, (pad + 8, ty), (pad + nw - 8, ty), (60, 60, 60), 1, cv2.LINE_AA)
    ty += lh

    # ── Action hint ─────────────────────────────────────────────────────────────────────
    if state.calibrated:
        cv2.putText(img, "✓ READY — SPACE to launch",
                    (pad + 10, ty), font, 0.50, (80, 255, 120), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "SPACE to calibrate",
                    (pad + 10, ty), font, 0.50, (140, 160, 220), 1, cv2.LINE_AA)

    # ESC hint in panel corner
    cv2.putText(img, "ESC quit", (pad + nw - 80, pad + 28),
                font, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    # 5. Gesture hints — bottom-center (same style as _draw_gesture_hints in main)
    if pts is not None:
        hints = []
        if poing:
            hints.append(("FIST", (80, 255, 80)))
        if pinch:
            hints.append(("PINCH", (80, 255, 160)))
        if hints:
            hint_font = cv2.FONT_HERSHEY_SIMPLEX
            hint_scale = 0.55
            y_base = h - 50
            total_w = sum(cv2.getTextSize(h[0], hint_font, hint_scale, 1)[0][0] + 20
                          for h in hints) + 24
            x_start = max(12, (w - total_w) // 2)
            for label, color in hints:
                (tw, th), _ = cv2.getTextSize(label, hint_font, hint_scale, 1)
                cv2.rectangle(img, (x_start - 10, y_base - th - 8),
                              (x_start + tw + 10, y_base + 10), (12, 12, 12), -1)
                cv2.putText(img, label, (x_start, y_base),
                            hint_font, hint_scale, color, 1, cv2.LINE_AA)
                x_start += tw + 20

    # 6. FPS in top-right (same style as main.py HUD)
    fps_color = (0, 255, 0) if fps >= 30 else (255, 255, 0) if fps >= 15 else (0, 100, 255)
    cv2.putText(img, f"{fps:.0f} fps", (w - 100, pad + 28),
                font, 0.48, fps_color, 1, cv2.LINE_AA)

    # 7. No hand detected message
    if pts is None:
        no_hand_msg = "no hand detected — show your hand to camera"
        (tw, th), _ = cv2.getTextSize(no_hand_msg, font, 0.45, 1)
        mx, my = w // 2 - tw // 2, h // 2
        cv2.rectangle(img, (mx - 12, my - th - 10), (mx + tw + 12, my + 10),
                      (12, 12, 12), -1)
        cv2.putText(img, no_hand_msg, (mx, my), font, 0.45,
                    (100, 100, 100), 1, cv2.LINE_AA)


# ── Calibration ─────────────────────────────────────────────────────────────────

# Qt key codes as returned by cv2.waitKey on Linux (Qt/Wayland backend)
_K_UP    = 0x010000EA   # 16777258
_K_DOWN  = 0x010000EC   # 16777260
_K_LEFT  = 0x010000E9   # 16777257
_K_RIGHT = 0x010000EB   # 16777259


def _probe_cameras(max_idx: int = 8) -> list:
    """Return list of camera indices that deliver frames."""
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        time.sleep(0.05)
        if ok:
            found.append(i)
    return found


def _open_camera(idx, width, height, fps):
    """Open camera with retry (same pattern as settings_gui)."""
    for _attempt in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            # Prime the pipeline — first frames are often undeliverable on V4L2
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        if _attempt < 3:
            time.sleep(0.25)
    return None


def _run_calibration(state, cap):
    """Measure hand scale + benchmark detection."""
    print("Calibrating — hold still …")

    # Hand scale
    scales = []
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
        print(f"  Hand scale : {ref:.1f} px")
    else:
        print("  No hand detected during calibration")

    # Benchmark (opens separate handles, releases them)
    try:
        hi_ms   = _benchmark_latency(_landmarker, 1280, 720, CAM['source'], samples=4)
        base_ms = _benchmark_latency(_landmarker, 640, 360, CAM['source'], samples=4)
        budget  = CAM.get('detect_latency_budget_ms', 25)
        if hi_ms <= budget:
            state.benchmark_ms = hi_ms
            state.benchmark_res = f"1280×720  {hi_ms:.1f}ms"
        else:
            state.benchmark_ms = base_ms
            state.benchmark_res = f"640×360   {base_ms:.1f}ms"
        print(f"  Benchmark  : hi={hi_ms:.1f}ms  base={base_ms:.1f}ms  budget={budget}ms")
    except Exception as e:
        state.benchmark_ms = None
        state.benchmark_res = "error"
        print(f"  Benchmark failed: {e}")

    state.calibrated = True
    print("  Calibration done. Adjust params with arrows, then SPACE to launch.")


# ── Main loop ───────────────────────────────────────────────────────────────────

def run_startup() -> bool:
    """
    Show the startup calibration screen.
    Returns True if the user confirmed and wants to launch the main app.
    """
    cap = cv2.VideoCapture(CAM['source'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM['display_w'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM['display_h'])
    cap.set(cv2.CAP_PROP_FPS, CAM.get('camera_fps', 60))

    state    = _StartupState()
    smoother = LandmarkSmoother()
    fps_frames, fps_time, current_fps = 0, time.monotonic(), 0.0

    # Probe for available cameras
    available = _probe_cameras()
    if not available:
        print("ERROR: no camera detected. Use --settings to pick a source.")
        return False
    print(f"Detected cameras: {available}")

    # Open camera with retry
    cap = _open_camera(CAM['source'], CAM['display_w'], CAM['display_h'],
                       CAM.get('camera_fps', 60))
    if cap is None:
        # Fall back to first available
        cap = _open_camera(available[0], CAM['display_w'], CAM['display_h'],
                           CAM.get('camera_fps', 60))
    if cap is None:
        print("ERROR: could not open camera.")
        return False

    print("Startup: hold your hand in front of the camera. Press SPACE to calibrate.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # Detect hands (same as main: detect on full frame)
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = _landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        pts = None
        if result.hand_landmarks:
            h, w = frame.shape[:2]
            raw = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
            pts = smoother.smooth(raw, time.monotonic())

        # Estimate gestures
        poing = _est_poing(pts, HAND['fist_fold_scale'],
                           HAND['fist_tip_mid_ratio'],
                           state.fist_min_folded) if pts else False
        pinch = _est_pinch(pts, state.pinch_ratio_on,
                           state.pinch_ratio_off) if pts else False

        # FPS
        fps_frames += 1
        now = time.monotonic()
        if now - fps_time >= 0.5:
            current_fps = fps_frames / (now - fps_time)
            fps_frames = 0
            fps_time = now

        # Full TD pipeline
        hand_count = 1 if pts is not None else 0
        _draw_feedback(frame, state, pts, poing, pinch, hand_count, current_fps)

        cv2.imshow("Vision AI — Calibration", frame)
        key = cv2.waitKey(16) & 0xFF

        # ── ESC → quit ───────────────────────────────────────────────────────
        if key == 27:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            return False

        # ── SPACE → calibrate (first press) or launch (second press) ───────────
        elif key == 32 and state.prev_key != 32:
            state.prev_key = 32
            if not state.calibrated:
                _run_calibration(state, cap)
                # The benchmark opens a second V4L2 handle on the same camera,
                # which can seize the device on Linux — re-acquire the capture.
                if cap:
                    cap.release()
                    cap = None
                cap = _open_camera(CAM['source'], CAM['display_w'], CAM['display_h'],
                                   CAM.get('camera_fps', 60))
                # Flush any stale key events queued during calibration
                state.prev_key = None
                while True:
                    k = cv2.pollKey()
                    if k == -1:
                        break
            else:
                # Apply tuned params
                HAND['fist_min_folded'] = state.fist_min_folded
                HAND['pinch_ratio_on']  = state.pinch_ratio_on
                HAND['pinch_ratio_off'] = state.pinch_ratio_off
                if cap:
                    cap.release()
                cv2.destroyAllWindows()
                print("Launching main app …")
                return True

        # ── Arrow keys → adjust params ───────────────────────────────────────
        elif key == _K_UP and state.prev_key != _K_UP:   # UP
            state.prev_key = _K_UP
            if state.fist_min_folded < _ADJUSTABLE['fist_min_folded']['hi']:
                state.fist_min_folded += 1
                print(f"fist_min_folded → {state.fist_min_folded}")
        elif key == _K_DOWN and state.prev_key != _K_DOWN:   # DOWN
            state.prev_key = _K_DOWN
            if state.fist_min_folded > _ADJUSTABLE['fist_min_folded']['lo']:
                state.fist_min_folded -= 1
                print(f"fist_min_folded → {state.fist_min_folded}")
        elif key == _K_LEFT and state.prev_key != _K_LEFT:   # LEFT
            state.prev_key = _K_LEFT
            step = _ADJUSTABLE['pinch_ratio_on']['step']
            if state.pinch_ratio_on > _ADJUSTABLE['pinch_ratio_on']['lo']:
                state.pinch_ratio_on = round(state.pinch_ratio_on - step, 2)
            if state.pinch_ratio_off > _ADJUSTABLE['pinch_ratio_off']['lo']:
                state.pinch_ratio_off = round(state.pinch_ratio_off - step, 2)
            print(f"pinch on={state.pinch_ratio_on}, off={state.pinch_ratio_off}")
        elif key == _K_RIGHT and state.prev_key != _K_RIGHT:   # RIGHT
            state.prev_key = _K_RIGHT
            step = _ADJUSTABLE['pinch_ratio_on']['step']
            if state.pinch_ratio_on < _ADJUSTABLE['pinch_ratio_on']['hi']:
                state.pinch_ratio_on = round(state.pinch_ratio_on + step, 2)
            if state.pinch_ratio_off < _ADJUSTABLE['pinch_ratio_off']['hi']:
                state.pinch_ratio_off = round(state.pinch_ratio_off + step, 2)
            print(f"pinch on={state.pinch_ratio_on}, off={state.pinch_ratio_off}")

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    return False
