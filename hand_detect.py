"""
hand_detect.py — MediaPipe hand detection + gesture estimation
══════════════════════════════════════════════════════════════
Pipeline per frame:
  1. MediaPipe HandLandmarker (VIDEO mode) → 21 landmarks + handedness
  2. Stable identity assignment (match by wrist proximity across frames)
  3. One Euro filter smoothing (jitter-free, low-lag)
  4. Gesture estimation: fist (debounced), pinch (Schmitt hysteresis + debounce)
  5. TD-style skeleton render on the dimmed frame
"""
import os
import time
import math
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import td_render as td
from config import HAND, DETECTION, SMOOTHING
from kalman_filter import LandmarkSmoother

# ── Model ────────────────────────────────────────────────────────────────────────
MODEL_PATH = DETECTION['model_path']
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


_ensure_model()

_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=DETECTION['num_hands'],
        min_hand_detection_confidence=DETECTION['min_hand_detection_confidence'],
        min_hand_presence_confidence=DETECTION['min_hand_presence_confidence'],
        min_tracking_confidence=DETECTION['min_tracking_confidence'],
    )
)

# ── Skeleton topology ────────────────────────────────────────────────────────────
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
_FINGERTIPS  = {4, 8, 12, 16, 20}
_PALM_JOINTS = {0, 1, 5, 9, 13, 17}

# ── Gesture parameters (from config) ─────────────────────────────────────────────
_PINCH_ON  = HAND['pinch_ratio_on']
_PINCH_OFF = HAND['pinch_ratio_off']
_FIST_TIP_MID = HAND['fist_tip_mid_ratio']
_FIST_FOLD_SCALE = HAND['fist_fold_scale']
_FIST_MIN_FOLDED  = HAND['fist_min_folded']
_DEB_ON  = HAND['debounce_on']
_DEB_OFF = HAND['debounce_off']
_WRIST_SEP = HAND['min_wrist_separation']

# ── Identity matching ────────────────────────────────────────────────────────────
# Max wrist distance (px in detect frame) to consider the same hand.
_MATCH_DIST = 120
# Keep state this long (s) after a hand disappears before clearing.
_STATE_TIMEOUT = 0.5


class _HandState:
    """Per-hand tracking state, keyed by a stable id."""
    __slots__ = ('id', 'smoother', 'last_seen', 'pinch', 'poing',
                 'poing_hist', 'pinch_hist', '_last_wrist')

    def __init__(self, hid: int):
        self.id         = hid
        self.smoother   = LandmarkSmoother()
        self.last_seen  = 0.0
        self.pinch      = False
        self.poing      = False
        self.poing_hist = deque(maxlen=max(_DEB_ON, _DEB_OFF))
        self.pinch_hist = deque(maxlen=max(_DEB_ON, _DEB_OFF))
        self._last_wrist = None


def _hand_scale(pts) -> float:
    dx = pts[9][0] - pts[0][0]
    dy = pts[9][1] - pts[0][1]
    s = math.hypot(dx, dy)
    return s if s > 1.0 else 1.0


def _est_poing_brut(pts) -> bool:
    """
    Raw fist check: a finger is folded when its tip is closer to the wrist
    than 90% of its middle phalanx is, AND the tip is near its base.
    Fist = at least 3 of 4 fingers folded.
    """
    wrist = pts[0]
    scale = _hand_scale(pts)
    fingers = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]
    folded = 0
    for tip, mid, base in fingers:
        d_tip_wrist = math.hypot(pts[tip][0] - wrist[0], pts[tip][1] - wrist[1])
        d_mid_wrist = math.hypot(pts[mid][0] - wrist[0], pts[mid][1] - wrist[1])
        d_tip_base  = math.hypot(pts[tip][0] - pts[base][0],
                                 pts[tip][1] - pts[base][1])
        if d_tip_wrist < d_mid_wrist * _FIST_TIP_MID and d_tip_base < scale * _FIST_FOLD_SCALE:
            folded += 1
    return folded >= _FIST_MIN_FOLDED


def _est_pincement_brut(pts, state) -> bool:
    """
    Raw pinch check with Schmitt-trigger hysteresis:
    engages when ratio < _PINCH_ON, releases when ratio > _PINCH_OFF.
    """
    scale = _hand_scale(pts)
    ratio = math.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1]) / scale
    if ratio < _PINCH_ON:
        state.pinch = True
    elif ratio > _PINCH_OFF:
        state.pinch = False
    return state.pinch


def _debounce(history, raw, on_thresh, off_thresh, current) -> bool:
    """
    Temporal debounce: requires on_thresh consecutive True samples to switch
    on, and off_thresh consecutive False samples to switch off.
    """
    history.append(raw)
    if not current and len(history) >= on_thresh:
        if all(list(history)[-on_thresh:]):
            return True
    if current and len(history) >= off_thresh:
        if not any(list(history)[-off_thresh:]):
            return False
    return current


def _match_ids(raw_landmarks, states: dict, w: int, h: int) -> list:
    """
    Assign stable ids to current-frame hands by matching wrist positions
    against previously seen hands. Unmatched hands get fresh ids and are
    registered in `states` immediately (so their One Euro state survives
    from the next frame onward).
    Returns a list of _HandState in the same order as raw_landmarks.
    """
    cur_wrist = [(raw_lm.landmarks[0].x * w, raw_lm.landmarks[0].y * h) for raw_lm in raw_landmarks]
    prev = [(sid, st._last_wrist) for sid, st in states.items()]

    matched = []
    used = set()
    for cw in cur_wrist:
        best_sid, best_d = None, _MATCH_DIST
        for j, (sid, pw) in enumerate(prev):
            if j in used or pw is None:
                continue
            d = math.hypot(cw[0] - pw[0], cw[1] - pw[1])
            if d < best_d:
                best_d, best_sid = d, sid
        if best_sid is not None:
            used.add(prev.index((best_sid, states[best_sid]._last_wrist)))
            matched.append(states[best_sid])
        else:
            st = _HandState(hid=_next_id())
            states[st.id] = st
            matched.append(st)
    # ── Set _last_wrist for ALL matched hands (including newly created ones) ────
    for state, raw_lm in zip(matched, raw_landmarks):
        state._last_wrist = (raw_lm.landmarks[0].x * w, raw_lm.landmarks[0].y * h)
    return matched


_next_id_counter = 0


def _next_id() -> int:
    global _next_id_counter
    _next_id_counter += 1
    return _next_id_counter


def rotation_3d(raw_landmarks) -> dict:
    """
    Estimate rough hand orientation from three landmarks.
    'inclinaison': tilt from vertical (deg), 'rotation': yaw (deg), 'roulis': roll (deg)
    """
    p0, p5, p17 = raw_landmarks[0], raw_landmarks[5], raw_landmarks[17]
    dx = p5.x - p0.x; dy = p5.y - p0.y; dz = p5.z - p0.z
    lx = p17.x - p5.x; ly = p17.y - p5.y
    return {
        'inclinaison': round(math.degrees(math.atan2(dy, math.hypot(dx, dz))), 1),
        'rotation'   : round(math.degrees(math.atan2(dz, dx)), 1),
        'roulis'     : round(math.degrees(math.atan2(ly, lx)), 1),
    }


# ── TouchDesigner-style hand renderer ────────────────────────────────────────────

def _draw_hand_td(img, pts, gestes):
    """
    Renders the hand skeleton in TouchDesigner aesthetic:
    - Hair-thin lines on a float32 emission layer
    - Gaussian bloom pass
    - Additive composite onto the (already-dimmed) frame
    """
    h, w = img.shape[:2]
    poing = gestes['poing']
    pinch = gestes['pincement']

    em = td.make_emission(h, w)

    # Skeleton lines — palm slightly dimmer than finger bones
    for s, e in _CONNECTIONS:
        is_palm = (s in _PALM_JOINTS and e in _PALM_JOINTS)
        color = td.TD_BLUE if (is_palm or poing) else td.TD_CYAN
        td.draw_line_em(em, pts[s], pts[e], color=(color * (0.5 if poing else 0.8)))

    # Joints
    for i, pt in enumerate(pts):
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

    # Fingertip state indicator
    td.draw_fingertip_em(em, pts[8], pinch=pinch, fist=poing)

    # Bloom & additive composite
    bloomed = td.bloom(em, kernel=15, sigma=6.0, strength=0.9)
    td.add_emission(img, bloomed)


# ── Module state ─────────────────────────────────────────────────────────────────
_states: dict = {}   # id → _HandState


def hand_detect(img):
    """
    Detect hands on the original bright frame, then apply TD dimming + skeleton
    only to the display copy.

    Returns (annotated_frame, hand_data) where hand_data is a list of:
        {'id': int, 'points': [(x,y)...21], 'gestes': {'poing': b, 'pincement': b},
         'rotation': {...}}
    """
    global _states
    h, w = img.shape[:2]
    now = time.monotonic()

    # ── Detection on the ORIGINAL frame (full brightness) ────────────────────────
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    result = _landmarker.detect_for_video(mp_img, int(time.time() * 1000))

    # ── Dim the frame for display AFTER detection ───────────────────────────────
    img_f = img.astype(np.float32) / 255.0
    img_f = img_f * td.CAM_DIM + td.BG_TINT
    img[:] = np.clip(img_f * 255, 0, 255).astype(np.uint8)

    if not result.hand_landmarks:
        # Expire stale state (timeout instead of instant clear → no jumps on
        # brief occlusion).
        for sid in [s for s, st in _states.items() if now - st.last_seen > _STATE_TIMEOUT]:
            del _states[sid]
        return img, []

    raw_landmarks = result.hand_landmarks

    # ── Ghost-hand guard: two "hands" whose wrists are too close are one hand ────
    if len(raw_landmarks) == 2:
        w0 = (raw_landmarks[0][0].x * w, raw_landmarks[0][0].y * h)
        w1 = (raw_landmarks[1][0].x * w, raw_landmarks[1][0].y * h)
        if math.hypot(w0[0] - w1[0], w0[1] - w1[1]) < _WRIST_SEP:
            # Keep the first (MediaPipe orders by detection confidence)
            raw_landmarks = raw_landmarks[:1]

    # ── Stable identity ─────────────────────────────────────────────────────────
    # st._last_wrist persists from the previous frame's call (set after smoothing),
    # so matching always sees the real previous wrist.
    matched = _match_ids(raw_landmarks, _states, w, h)

    # ── Prune unmatched states that timed out ───────────────────────────────────
    matched_ids = {m.id for m in matched}
    for sid in list(_states):
        if sid not in matched_ids and now - _states[sid].last_seen > _STATE_TIMEOUT:
            del _states[sid]

    hand_data = []
    for state, raw_lm in zip(matched, raw_landmarks):
        raw_float = [(lm.x * w, lm.y * h) for lm in raw_lm]

        # Smooth landmarks with the One Euro filter (time-based)
        pts_f = state.smoother.smooth(raw_float, now)
        pts = [(int(round(x)), int(round(y))) for x, y in pts_f]
        state._last_wrist = pts[0]
        state.last_seen = now

        # Gestures (computed on smoothed points for stability)
        state.poing = _debounce(state.poing_hist, _est_poing_brut(pts),
                                _DEB_ON, _DEB_OFF, state.poing)
        state.pinch = _debounce(state.pinch_hist, _est_pincement_brut(pts, state),
                                _DEB_ON, _DEB_OFF, state.pinch)

        gestes = {'poing': state.poing, 'pincement': state.pinch}
        hand_data.append({
            'id'      : state.id,
            'points'  : pts,
            'gestes'  : gestes,
            'rotation': rotation_3d(raw_lm),
        })
        _draw_hand_td(img, pts, gestes)

        # Register in state dict under stable id
        _states[state.id] = state

    return img, hand_data


def release():
    """Close the MediaPipe landmarker (frees model memory)."""
    _landmarker.close()
