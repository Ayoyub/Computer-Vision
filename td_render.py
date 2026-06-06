"""
td_render.py — TouchDesigner-style bloom render pipeline
═════════════════════════════════════════════════════════

Shared utilities used by hand_detect.py, main.py (shapes), and glob_metal.py.

Core idea:
  Everything is drawn onto a float32 black emission layer.
  A gaussian blur of that layer creates the glow/bloom.
  The two are composited additively onto a darkened camera feed.

  result = cam * CAM_DIM + emission + blur(emission) * BLOOM_STRENGTH

No new dependencies — pure numpy + opencv.
"""

import cv2
import numpy as np

# ── Global style constants ────────────────────────────────────────────────────────
CAM_DIM          = 0.18    # how much the camera feed is dimmed (0=black, 1=full)
BLOOM_STRENGTH   = 0.75    # glow layer intensity multiplier
BLOOM_KERNEL     = 31      # gaussian blur kernel size for bloom (must be odd)
BLOOM_SIGMA      = 12.0    # gaussian sigma

# Palette — one accent hue, varying luminosity (TD style: near-white → cyan → deep indigo)
# All in BGR float (0-1)
TD_WHITE   = np.array([1.00, 1.00, 1.00], dtype=np.float32)  # core catchlight
TD_CYAN    = np.array([0.85, 0.95, 0.60], dtype=np.float32)  # mid glow  (B,G,R)
TD_BLUE    = np.array([0.55, 0.40, 0.15], dtype=np.float32)  # outer glow
TD_DIM     = np.array([0.12, 0.10, 0.04], dtype=np.float32)  # dim interior

# Background tint for the camera (very dark blue-grey)
BG_TINT    = np.array([0.06, 0.05, 0.03], dtype=np.float32)  # additive tint on cam


def dim_camera(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Darken the camera frame and add a subtle blue-grey tint.
    Returns float32 (0-1) BGR array.
    """
    f = frame_bgr.astype(np.float32) / 255.0
    f = f * CAM_DIM + BG_TINT
    return np.clip(f, 0, 1)


def bloom(emission: np.ndarray,
          kernel: int = BLOOM_KERNEL,
          sigma:  float = BLOOM_SIGMA,
          strength: float = BLOOM_STRENGTH) -> np.ndarray:
    """
    Given a float32 emission layer (0-1), return emission + glow.
    """
    glow = cv2.GaussianBlur(emission, (kernel, kernel), sigma)
    return np.clip(emission + glow * strength, 0, 1)


def composite(cam_dim: np.ndarray, emission_bloomed: np.ndarray) -> np.ndarray:
    """
    Additively composite bloomed emission onto the dimmed camera.
    Returns uint8 BGR.
    """
    result = np.clip(cam_dim + emission_bloomed, 0, 1)
    return (result * 255).astype(np.uint8)


def make_emission(h: int, w: int) -> np.ndarray:
    """Create a zeroed float32 emission layer (H, W, 3)."""
    return np.zeros((h, w, 3), dtype=np.float32)


def draw_line_em(em: np.ndarray, p1, p2,
                 color=TD_CYAN, thickness: int = 1):
    """Draw a line onto an emission layer (color is float32 BGR 0-1)."""
    c = (float(color[0]), float(color[1]), float(color[2]))
    cv2.line(em, p1, p2, c, thickness, cv2.LINE_AA)


def draw_node_em(em: np.ndarray, pt, r: int = 2,
                 core=TD_WHITE, rim=TD_CYAN):
    """
    Draw a TD-style node: tiny dark-punched circle with a bright rim.
    core  — inner catchlight color (float32 BGR)
    rim   — outer ring color (float32 BGR)
    """
    cx, cy = int(pt[0]), int(pt[1])
    # Rim
    cv2.circle(em, (cx, cy), r,
               (float(rim[0]), float(rim[1]), float(rim[2])),
               1, cv2.LINE_AA)
    # Core punch (black sink — draws attention to the node)
    cv2.circle(em, (cx, cy), max(1, r-1), (0, 0, 0), -1, cv2.LINE_AA)
    # Catchlight
    if r >= 2:
        cv2.circle(em, (cx-1, cy-1), max(1, r//2),
                   (float(core[0]), float(core[1]), float(core[2])),
                   -1, cv2.LINE_AA)


def draw_fingertip_em(em: np.ndarray, pt, pinch: bool = False, fist: bool = False):
    """
    Fingertip indicator in TD style.
    pinch → bright white bloom ring
    fist  → dim cross hairline
    """
    cx, cy = int(pt[0]), int(pt[1])
    if pinch:
        cv2.circle(em, (cx, cy), 10,
                   (float(TD_WHITE[0]), float(TD_WHITE[1]), float(TD_WHITE[2])),
                   1, cv2.LINE_AA)
        cv2.circle(em, (cx, cy), 14,
                   (float(TD_CYAN[0]*0.5), float(TD_CYAN[1]*0.5), float(TD_CYAN[2]*0.5)),
                   1, cv2.LINE_AA)
    elif fist:
        dim = 0.25
        cv2.line(em, (cx-6, cy-6), (cx+6, cy+6),
                 (dim, dim, dim), 1, cv2.LINE_AA)
        cv2.line(em, (cx+6, cy-6), (cx-6, cy+6),
                 (dim, dim, dim), 1, cv2.LINE_AA)
