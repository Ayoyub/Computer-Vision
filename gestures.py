"""
gestures.py — shared gesture helpers
Used by main.py, virt_mouse.py, glob_metal.py
"""
import cv2
import math

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DOUBLE-FIST → BACK TO MENU                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_DOUBLE_FIST_FRAMES_NEEDED = 20


def check_double_fist(hand_data, counter: list) -> bool:
    """
    Returns True (once) when both hands are fisted for ~0.33s consecutively.
    counter must be a mutable [int].
    """
    if (len(hand_data) >= 2
            and hand_data[0]['gestes']['poing']
            and hand_data[1]['gestes']['poing']):
        counter[0] += 1
    else:
        counter[0] = 0
    if counter[0] >= _DOUBLE_FIST_FRAMES_NEEDED:
        counter[0] = 0
        return True
    return False


def draw_double_fist_hint(display, counter: list):
    """Progress arc shown while the double-fist gesture is building up."""
    if counter[0] == 0:
        return
    h, w     = display.shape[:2]
    cx, cy   = w // 2, 40
    progress = counter[0] / _DOUBLE_FIST_FRAMES_NEEDED
    angle    = int(360 * progress)
    cv2.ellipse(display, (cx, cy), (18, 18), -90, 0, 360,
                (40, 40, 40), 2, cv2.LINE_AA)
    cv2.ellipse(display, (cx, cy), (18, 18), -90, 0, angle,
                (180, 220, 140), 2, cv2.LINE_AA)
    font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.34
    label    = "both fists — menu"
    (tw, _), _ = cv2.getTextSize(label, font, sc, 1)
    cv2.putText(display, label, (cx - tw//2, cy + 30),
                font, sc, (90, 110, 80), 1, cv2.LINE_AA)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TWO-HAND SCISSOR SPREAD → ROTATION SPEED                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Activation conditions (all must be true simultaneously):
#   1. Exactly 2 hands detected, neither fisting
#   2. On each hand: index (lm 8) and middle (lm 12) fingertips are JOINED
#      — their distance < FINGER_JOIN_RATIO × hand_scale
#   3. Both hands are NEAR each other
#      — wrist distance < HAND_PROXIMITY_PX (display px)
#
# Controlled value:
#   Distance between the midpoint of each hand's joined pair, mapped linearly
#   from [SPREAD_PX_MIN … SPREAD_PX_MAX] → [SPEED_MULT_MIN … SPEED_MULT_MAX].
#   Applied to the last-touched shape (tracked externally).

FINGER_JOIN_RATIO  = 0.20    # index+middle must be within 20% of hand scale
HAND_PROXIMITY_PX  = 280     # wrists must be within this many display pixels
SPREAD_PX_MIN      = 10      # minimum pair-to-pair distance (≈ touching)
SPREAD_PX_MAX      = 220     # maximum pair-to-pair distance (arms wide open)
SPEED_MULT_MIN     = 0.10    # nearly stopped
SPEED_MULT_MAX     = 5.0     # spinning fast


def _hand_scale(pts) -> float:
    return max(math.hypot(pts[9][0]-pts[0][0], pts[9][1]-pts[0][1]), 1.0)


def _fingers_joined(hand) -> tuple:
    """
    Returns (joined: bool, pair_midpoint: (float,float)) for one hand.
    Checks that index (8) and middle (12) tips are close enough.
    """
    pts   = hand['points']
    scale = _hand_scale(pts)
    d     = math.hypot(pts[8][0]-pts[12][0], pts[8][1]-pts[12][1])
    joined = d / scale < FINGER_JOIN_RATIO
    mid_x  = (pts[8][0] + pts[12][0]) / 2.0
    mid_y  = (pts[8][1] + pts[12][1]) / 2.0
    return joined, (mid_x, mid_y)


def read_two_hand_spread(hand_data, SX: float, SY: float):
    """
    Evaluates the two-hand scissor spread gesture.

    Returns:
        active      : bool   — gesture is currently valid
        speed_mult  : float  — rotation speed multiplier [SPEED_MULT_MIN … MAX]
        mid0_disp   : (int,int) — display-space midpoint of hand 0's pair
        mid1_disp   : (int,int) — display-space midpoint of hand 1's pair
    """
    if len(hand_data) < 2:
        return False, 1.0, None, None

    h0, h1 = hand_data[0], hand_data[1]

    # Neither hand may be fisting (conflicts with grab gesture)
    if h0['gestes']['poing'] or h1['gestes']['poing']:
        return False, 1.0, None, None

    # Both hands must have their index+middle joined
    joined0, mid0 = _fingers_joined(h0)
    joined1, mid1 = _fingers_joined(h1)
    if not (joined0 and joined1):
        return False, 1.0, None, None

    # Wrists must be near each other (display coordinates)
    w0x = h0['points'][0][0] * SX;  w0y = h0['points'][0][1] * SY
    w1x = h1['points'][0][0] * SX;  w1y = h1['points'][0][1] * SY
    wrist_dist = math.hypot(w0x-w1x, w0y-w1y)
    if wrist_dist > HAND_PROXIMITY_PX:
        return False, 1.0, None, None

    # Pair midpoints in display space
    m0d = (mid0[0]*SX, mid0[1]*SY)
    m1d = (mid1[0]*SX, mid1[1]*SY)

    spread_px  = math.hypot(m0d[0]-m1d[0], m0d[1]-m1d[1])
    t          = (spread_px - SPREAD_PX_MIN) / max(SPREAD_PX_MAX - SPREAD_PX_MIN, 1.0)
    t          = max(0.0, min(1.0, t))
    speed_mult = SPEED_MULT_MIN + t * (SPEED_MULT_MAX - SPEED_MULT_MIN)

    mid0_disp  = (int(m0d[0]), int(m0d[1]))
    mid1_disp  = (int(m1d[0]), int(m1d[1]))

    return True, speed_mult, mid0_disp, mid1_disp


def draw_spread_indicator(display, mid0, mid1, speed_mult, active):
    """
    TD-style indicator: a line between the two finger-pair midpoints,
    brightness proportional to speed, with a readout at the centre.
    """
    if not active or mid0 is None or mid1 is None:
        return

    t      = (speed_mult - SPEED_MULT_MIN) / max(SPEED_MULT_MAX - SPEED_MULT_MIN, 1e-6)
    bright = int(55 + t * 170)
    color  = (bright // 2, bright, int(bright * 0.4))   # dim cyan-green

    cv2.line(display, mid0, mid1, color, 1, cv2.LINE_AA)

    # Small dot at each midpoint
    cv2.circle(display, mid0, 4, color, -1, cv2.LINE_AA)
    cv2.circle(display, mid1, 4, color, -1, cv2.LINE_AA)

    # Label at the centre of the line
    cx = (mid0[0] + mid1[0]) // 2
    cy = (mid0[1] + mid1[1]) // 2 - 16
    label    = f"x{speed_mult:.1f}"
    font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.38
    (tw, _), _ = cv2.getTextSize(label, font, sc, 1)
    cv2.putText(display, label, (cx - tw//2, cy), font, sc, color, 1, cv2.LINE_AA)