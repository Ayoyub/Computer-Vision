"""
glob_metal.py — TouchDesigner-style Liquid Metal Metaball Glob
══════════════════════════════════════════════════════════════
Pure signal, pure bloom. No chrome LUT, no decoration.
The isosurface rim IS the light source — everything else is dark.

Pipeline:
  field   = Σ r²/d²  at 1/4 resolution
  rim     = |∇field| masked near the isosurface  → this is the emission
  interior = field >> threshold → very dim fill
  bloomed = gaussian_blur(emission) * strength
  display = cam * CAM_DIM  +  bloomed   (additive)

Zero new dependencies — numpy + opencv only.
"""

import cv2
import math
import numpy as np
import time
from config import CAM
import td_render as td

# ── Resolution ────────────────────────────────────────────────────────────────────
SIM_SCALE  = 4
DISPLAY_W  = CAM['display_w']
DISPLAY_H  = CAM['display_h']
SIM_W      = DISPLAY_W // SIM_SCALE
SIM_H      = DISPLAY_H // SIM_SCALE
CAM_W      = CAM['detect_w']
CAM_H      = CAM['detect_h']

# ── Metaball parameters ───────────────────────────────────────────────────────────
N_BLOBS          = 6
BLOB_R_MIN       = 26
BLOB_R_MAX       = 52
META_THRESHOLD   = 1.0
RIM_BAND         = 0.35    # field window around threshold that glows (TD: tight rim)

# ── Physics ───────────────────────────────────────────────────────────────────────
FRICTION         = 0.986
WALL_BOUNCE      = 0.60
HAND_PUSH_RADIUS = 75
HAND_PUSH_FORCE  = 2.0
HAND_PULL_FORCE  = 1.5
BLOB_REPULSION   = 800
BLOB_ATTRACT     = 25
SPEED_MAX        = 5.5
GRAVITY          = 0.055

# ── TD colour palette for the glob ───────────────────────────────────────────────
# One accent: cold cyan-white. Single hue, all luminosity.
GLOB_RIM_COLOR  = np.array([0.85, 0.95, 0.60], np.float32)  # BGR cyan-white (rim)
GLOB_FILL_COLOR = np.array([0.04, 0.06, 0.02], np.float32)  # near-black interior
GLOB_SIGIL_COL  = np.array([0.30, 0.40, 0.15], np.float32)  # dim connecting lines

# Pre-computed coordinate grids
_YY, _XX = np.mgrid[0:SIM_H, 0:SIM_W].astype(np.float32)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOB PHYSICS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Blob:
    def __init__(self, x=None, y=None, r=None):
        self.x  = float(x if x is not None else np.random.uniform(SIM_W*.2, SIM_W*.8))
        self.y  = float(y if y is not None else np.random.uniform(SIM_H*.2, SIM_H*.8))
        self.r  = float(r if r is not None else np.random.uniform(BLOB_R_MIN, BLOB_R_MAX))
        a       = np.random.uniform(0, 2*math.pi)
        s       = np.random.uniform(0.2, 1.0)
        self.vx = math.cos(a) * s
        self.vy = math.sin(a) * s
        self.base_r      = self.r
        self.pulse_phase = np.random.uniform(0, 2*math.pi)
        self.pulse_speed = np.random.uniform(0.018, 0.04)

    def update(self, push_pts, pull_pts):
        self.pulse_phase += self.pulse_speed
        self.r = self.base_r * (1.0 + 0.05 * math.sin(self.pulse_phase))

        self.vy += GRAVITY

        for hx, hy in push_pts:
            dx = self.x - hx;  dy = self.y - hy
            d  = math.hypot(dx, dy)
            if 0 < d < HAND_PUSH_RADIUS:
                f = HAND_PUSH_FORCE * (1.0 - d/HAND_PUSH_RADIUS)**2
                self.vx += dx/d * f;  self.vy += dy/d * f

        for fx, fy in pull_pts:
            dx = fx - self.x;  dy = fy - self.y
            d  = math.hypot(dx, dy)
            if 0 < d < HAND_PUSH_RADIUS * 1.5:
                f = HAND_PULL_FORCE * (1.0 - d/(HAND_PUSH_RADIUS*1.5))**2
                self.vx += dx/d * f;  self.vy += dy/d * f

        spd = math.hypot(self.vx, self.vy)
        if spd > SPEED_MAX:
            self.vx = self.vx/spd * SPEED_MAX
            self.vy = self.vy/spd * SPEED_MAX

        self.vx *= FRICTION;  self.vy *= FRICTION
        self.x  += self.vx;   self.y  += self.vy

        m = self.r * 0.3
        if self.x < m:             self.x = m;            self.vx =  abs(self.vx)*WALL_BOUNCE
        elif self.x > SIM_W - m:   self.x = SIM_W - m;   self.vx = -abs(self.vx)*WALL_BOUNCE
        if self.y < m:             self.y = m;            self.vy =  abs(self.vy)*WALL_BOUNCE
        elif self.y > SIM_H - m:   self.y = SIM_H - m;   self.vy = -abs(self.vy)*WALL_BOUNCE


def _repulsion(blobs):
    n = len(blobs)
    for i in range(n):
        for j in range(i+1, n):
            dx = blobs[i].x - blobs[j].x
            dy = blobs[i].y - blobs[j].y
            d  = max(math.hypot(dx, dy), 1.0)
            cr = blobs[i].r + blobs[j].r
            if d < cr * 0.8:
                f = BLOB_REPULSION / (d*d)
                nx, ny = dx/d, dy/d
                blobs[i].vx += nx*f*.5;  blobs[i].vy += ny*f*.5
                blobs[j].vx -= nx*f*.5;  blobs[j].vy -= ny*f*.5
            elif d < cr * 3.0:
                f = BLOB_ATTRACT / (d*d)
                nx, ny = dx/d, dy/d
                blobs[i].vx -= nx*f;  blobs[i].vy -= ny*f
                blobs[j].vx += nx*f;  blobs[j].vy += ny*f


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TD METABALL RENDERER                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _compute_field(blobs):
    field = np.zeros((SIM_H, SIM_W), np.float32)
    for b in blobs:
        dx = _XX - b.x;  dy = _YY - b.y
        field += (b.r * b.r) / (dx*dx + dy*dy + 0.1)
    return field


def _render_td_glob(field):
    """
    TouchDesigner-style metaball render.

    Emission sources:
      1. Rim band  — narrow strip where field ≈ META_THRESHOLD
                     This is the brightest part (where the surface curvature is)
      2. Interior  — field > threshold, very dim fill (the void absorbs light)

    Both are drawn onto a float32 emission layer, then bloomed.
    """
    # ── Rim mask: field within [threshold - band, threshold + band/4] ───────────
    # Tight on the outer side (more TD-like: thin luminous edge)
    rim_lo = META_THRESHOLD - RIM_BAND
    rim_hi = META_THRESHOLD + RIM_BAND * 0.25
    rim_t  = np.clip((field - rim_lo) / (rim_hi - rim_lo), 0, 1)   # 0→1 ramp in
    rim_t *= np.clip((rim_hi - field) / (rim_hi - rim_lo), 0, 1)   # 1→0 ramp out
    # rim_t peaks at threshold, falls off symmetrically → thin glowing rim

    # ── Interior mask ────────────────────────────────────────────────────────────
    interior = np.clip((field - META_THRESHOLD) / (META_THRESHOLD * 0.8), 0, 1)

    # ── Build emission at SIM resolution ────────────────────────────────────────
    em_sim = np.zeros((SIM_H, SIM_W, 3), np.float32)

    # Interior: near-black fill
    fill = interior[:, :, np.newaxis] * GLOB_FILL_COLOR[np.newaxis, np.newaxis, :]
    em_sim += fill

    # Rim: bright cyan-white
    rim_strength = rim_t[:, :, np.newaxis] * GLOB_RIM_COLOR[np.newaxis, np.newaxis, :]
    em_sim += rim_strength

    em_sim = np.clip(em_sim, 0, 1)

    # ── Upscale to display resolution ───────────────────────────────────────────
    em_big = cv2.resize(em_sim, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)

    # ── Bloom ────────────────────────────────────────────────────────────────────
    bloomed = td.bloom(em_big, kernel=35, sigma=14.0, strength=1.3)

    return bloomed


def _draw_sigil_lines_td(em, blobs, t):
    """
    Dim connecting lines between nearby blobs — pure emission, no color.
    TD aesthetic: data lines, not decoration.
    """
    n = len(blobs)
    for i in range(n):
        for j in range(i+1, n):
            bx1 = int(blobs[i].x * SIM_SCALE)
            by1 = int(blobs[i].y * SIM_SCALE)
            bx2 = int(blobs[j].x * SIM_SCALE)
            by2 = int(blobs[j].y * SIM_SCALE)
            d   = math.hypot(bx2-bx1, by2-by1)
            max_d = (blobs[i].r + blobs[j].r) * SIM_SCALE * 2.8
            if d < max_d:
                alpha = (1.0 - d/max_d) ** 2
                pulse = 0.5 + 0.5 * math.sin(t*1.8 + i*1.1 + j*0.9)
                col   = GLOB_SIGIL_COL * alpha * pulse
                td.draw_line_em(em, (bx1,by1), (bx2,by2), color=col)

    # Node at each blob centre
    for b in blobs:
        cx = int(b.x * SIM_SCALE);  cy = int(b.y * SIM_SCALE)
        pulse = 0.5 + 0.5 * math.sin(t*2.5 + b.pulse_phase)
        rim   = GLOB_RIM_COLOR * 0.6 * pulse
        td.draw_node_em(em, (cx, cy), r=3, core=td.TD_WHITE*0.5, rim=rim)


def _split(blobs, pinch_sim):
    px, py = pinch_sim
    for b in blobs[:]:
        if len(blobs) >= 10:
            break
        if math.hypot(px-b.x, py-b.y) < b.r*0.9 and b.base_r > BLOB_R_MIN*1.4:
            nr = b.base_r * 0.65
            a  = math.atan2(py-b.y, px-b.x)
            p  = a + math.pi/2
            b1 = Blob(b.x + math.cos(p)*nr*.4, b.y + math.sin(p)*nr*.4, nr)
            b1.vx = b.vx + math.cos(p)*1.5;  b1.vy = b.vy + math.sin(p)*1.5
            b2 = Blob(b.x - math.cos(p)*nr*.4, b.y - math.sin(p)*nr*.4, nr)
            b2.vx = b.vx - math.cos(p)*1.5;  b2.vy = b.vy - math.sin(p)*1.5
            blobs.remove(b);  blobs.extend([b1, b2])
            return True
    return False


def _draw_hud(display, n_blobs):
    h, w = display.shape[:2]
    lines = [
        "open hand — push",
        "fist — attract",
        "pinch — split",
        "ESC — menu",
    ]
    font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.32
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, sc, 1)
        x = w - tw - 14;  y = h - 12 - i*16
        cv2.putText(display, line, (x, y), font, sc, (50, 70, 55), 1, cv2.LINE_AA)
    cv2.putText(display, f"blobs  {n_blobs}", (14, 22),
                font, sc, (50, 70, 45), 1, cv2.LINE_AA)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_glob_mode(cam, det):
    SX_S = SIM_W / CAM_W;  SY_S = SIM_H / CAM_H
    SX_D = DISPLAY_W / CAM_W;  SY_D = DISPLAY_H / CAM_H

    blobs        = [Blob() for _ in range(N_BLOBS)]
    split_cd     = 0
    SPLIT_CD     = 18
    t            = 0.0

    print("Glob Metal | ESC = back to menu")

    while True:
        frame = cam.read()
        if frame is None:
            continue

        det.submit(cv2.resize(frame, (CAM_W, CAM_H)))
        res_frame, hand_data = det.get()

        # ── Hand data ────────────────────────────────────────────────────────────
        push_pts  = []
        pull_pts  = []
        pinch_sim = None

        if hand_data:
            for hand in hand_data:
                pts      = hand['points']
                is_fist  = hand['gestes']['poing']
                is_pinch = hand['gestes']['pincement']
                if is_fist:
                    pull_pts.append((pts[0][0]*SX_S, pts[0][1]*SY_S))
                else:
                    push_pts.append((pts[8][0]*SX_S, pts[8][1]*SY_S))
                if is_pinch:
                    pinch_sim = ((pts[4][0]+pts[8][0])/2*SX_S,
                                 (pts[4][1]+pts[8][1])/2*SY_S)

        # ── Split ────────────────────────────────────────────────────────────────
        if split_cd > 0:
            split_cd -= 1
        if pinch_sim and split_cd == 0:
            if _split(blobs, pinch_sim):
                split_cd = SPLIT_CD

        # ── Physics ──────────────────────────────────────────────────────────────
        _repulsion(blobs)
        for b in blobs:
            b.update(push_pts, pull_pts)

        # ── Render ───────────────────────────────────────────────────────────────
        # 1. Dim camera
        base = cv2.resize(res_frame, (DISPLAY_W, DISPLAY_H)) \
               if res_frame is not None else frame.copy()
        cam_dim = td.dim_camera(base)   # float32 0-1

        # 2. Metaball bloom layer
        field   = _compute_field(blobs)
        bloomed = _render_td_glob(field)   # float32 0-1, display size

        # 3. Sigil lines on a separate emission layer
        em_sigil = td.make_emission(DISPLAY_H, DISPLAY_W)
        _draw_sigil_lines_td(em_sigil, blobs, t)
        sigil_bloomed = td.bloom(em_sigil, kernel=15, sigma=5.0, strength=0.7)

        # 4. Hand indicator (very subtle — just a dim circle)
        em_hand = td.make_emission(DISPLAY_H, DISPLAY_W)
        if hand_data:
            for hand in hand_data:
                pts     = hand['points']
                is_fist = hand['gestes']['poing']
                ix_d    = int(pts[8][0]*SX_D);  iy_d = int(pts[8][1]*SY_D)
                r_vis   = int(HAND_PUSH_RADIUS * SIM_SCALE * (1.4 if is_fist else 1.0))
                col     = td.TD_CYAN * 0.18
                cv2.circle(em_hand, (ix_d, iy_d), r_vis,
                           (float(col[0]), float(col[1]), float(col[2])),
                           1, cv2.LINE_AA)

        # 5. Composite: cam + blobs + sigil + hand indicator
        result = np.clip(cam_dim + bloomed + sigil_bloomed + em_hand, 0, 1)
        display = (result * 255).astype(np.uint8)

        _draw_hud(display, len(blobs))

        t += 0.033
        cv2.imshow("Vision AI", display)
        if cv2.waitKey(1) == 27:
            break