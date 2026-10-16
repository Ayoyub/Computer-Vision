"""
glob_metal.py — TouchDesigner-style Liquid Metal Metaball Glob
══════════════════════════════════════════════════════════════
Pure signal, pure bloom. No chrome LUT, no decoration.
The isosurface rim IS the light source — everything else is dark.

Pipeline:
  field   = Σ r²/d²  at 1/4 resolution
  grad    = ∇field  → surface normals + signed distance
  body    = subsurface depth cue  (thick = warm fill)
  spec    = directional glint     (fixed screen-space light)
  rim     = |∇field| band × curvature response  → this is the emission
  interior = field >> threshold → very dim fill
  bloomed = gaussian_blur(emission, sim-res) * strength  → upscale
  display = cam * CAM_DIM  +  bloomed   (additive)

Zero new dependencies — numpy + opencv only.
"""

import math

import cv2
import numpy as np

from config import CAM, GLOB
import td_render as td
import gestures as gest

# ── Resolution ────────────────────────────────────────────────────────────────────
SIM_SCALE  = GLOB['sim_scale']
DISPLAY_W  = CAM['display_w']
DISPLAY_H  = CAM['display_h']
SIM_W      = DISPLAY_W // SIM_SCALE
SIM_H      = DISPLAY_H // SIM_SCALE
CAM_W      = CAM['detect_w']
CAM_H      = CAM['detect_h']

# ── Metaball parameters (from config) ─────────────────────────────────────────────
N_BLOBS          = GLOB['n_blobs']
BLOB_R_MIN       = GLOB['blob_r_min']
BLOB_R_MAX       = GLOB['blob_r_max']
META_THRESHOLD   = GLOB['meta_threshold']
RIM_BAND         = GLOB['rim_band']
BLOB_MAX_COUNT   = GLOB['blob_max_count']
SPLIT_CD         = GLOB['split_cooldown']

# ── Physics (from config) ─────────────────────────────────────────────────────────
FRICTION         = GLOB['friction']
WALL_BOUNCE      = GLOB['wall_bounce']
HAND_PUSH_RADIUS = GLOB['hand_push_radius']
HAND_PUSH_FORCE  = GLOB['hand_push_force']
HAND_PULL_FORCE  = GLOB['hand_pull_force']
BLOB_REPULSION   = GLOB['blob_repulsion']
BLOB_ATTRACT     = GLOB['blob_attract']
SPEED_MAX        = GLOB['speed_max']
GRAVITY          = GLOB['gravity']

# ── TD colour palette for the glob (from config) ──────────────────────────────────
GLOB_RIM_COLOR  = np.array(GLOB['rim_color'],   np.float32)
GLOB_FILL_COLOR = np.array(GLOB['fill_color'],  np.float32)
GLOB_SIGIL_COL  = np.array(GLOB['sigil_color'], np.float32)

# ── Shader constants (from config) ───────────────────────────────────────────────
SPEC_SHININESS = GLOB['spec_shininess']
SPEC_STRENGTH  = GLOB['spec_strength']
BODY_DEPTH     = GLOB['body_depth']
BODY_STRENGTH  = GLOB['body_strength']
CURV_SCALE     = GLOB['curv_scale']
CURV_MIN       = GLOB['curv_min']
CURV_MAX       = GLOB['curv_max']

# Fixed screen-space light direction (top-left, y-down)
_LIGHT = np.array((-0.45, -0.89), np.float32)

# Pre-computed coordinate grids
_YY, _XX = np.mgrid[0:SIM_H, 0:SIM_W].astype(np.float32)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BLOB PHYSICS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Blob:
    def __init__(self, x=None, y=None, r=None):
        self.x = float(x if x is not None else np.random.uniform(SIM_W * .2, SIM_W * .8))
        self.y = float(y if y is not None else np.random.uniform(SIM_H * .2, SIM_H * .8))
        self.r = float(r if r is not None else np.random.uniform(BLOB_R_MIN, BLOB_R_MAX))
        a = np.random.uniform(0, 2 * math.pi)
        s = np.random.uniform(0.2, 1.0)
        self.vx = math.cos(a) * s
        self.vy = math.sin(a) * s
        self.base_r = self.r
        self.pulse_phase = np.random.uniform(0, 2 * math.pi)
        self.pulse_speed = np.random.uniform(0.018, 0.04)

    def update(self, push_pts, pull_pts):
        self.pulse_phase += self.pulse_speed
        self.r = self.base_r * (1.0 + 0.05 * math.sin(self.pulse_phase))

        self.vy += GRAVITY

        for hx, hy in push_pts:
            dx = self.x - hx;  dy = self.y - hy
            d = math.hypot(dx, dy)
            if 0 < d < HAND_PUSH_RADIUS:
                f = HAND_PUSH_FORCE * (1.0 - d / HAND_PUSH_RADIUS) ** 2
                self.vx += dx / d * f;  self.vy += dy / d * f

        for fx, fy in pull_pts:
            dx = fx - self.x;  dy = fy - self.y
            d = math.hypot(dx, dy)
            if 0 < d < HAND_PUSH_RADIUS * 1.5:
                f = HAND_PULL_FORCE * (1.0 - d / (HAND_PUSH_RADIUS * 1.5)) ** 2
                self.vx += dx / d * f;  self.vy += dy / d * f

        spd = math.hypot(self.vx, self.vy)
        if spd > SPEED_MAX:
            self.vx = self.vx / spd * SPEED_MAX
            self.vy = self.vy / spd * SPEED_MAX

        self.vx *= FRICTION;  self.vy *= FRICTION
        self.x += self.vx;    self.y += self.vy

        m = self.r * 0.3
        if self.x < m:             self.x = m;            self.vx =  abs(self.vx) * WALL_BOUNCE
        elif self.x > SIM_W - m:   self.x = SIM_W - m;   self.vx = -abs(self.vx) * WALL_BOUNCE
        if self.y < m:             self.y = m;            self.vy =  abs(self.vy) * WALL_BOUNCE
        elif self.y > SIM_H - m:   self.y = SIM_H - m;   self.vy = -abs(self.vy) * WALL_BOUNCE


def _repulsion(blobs):
    n = len(blobs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = blobs[i].x - blobs[j].x
            dy = blobs[i].y - blobs[j].y
            d = max(math.hypot(dx, dy), 1.0)
            cr = blobs[i].r + blobs[j].r
            if d < cr * 0.8:
                f = BLOB_REPULSION / (d * d)
                nx, ny = dx / d, dy / d
                blobs[i].vx += nx * f * .5;  blobs[i].vy += ny * f * .5
                blobs[j].vx -= nx * f * .5;  blobs[j].vy -= ny * f * .5
            elif d < cr * 3.0:
                f = BLOB_ATTRACT / (d * d)
                nx, ny = dx / d, dy / d
                blobs[i].vx -= nx * f;  blobs[i].vy -= ny * f
                blobs[j].vx += nx * f;  blobs[j].vy += ny * f


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TD METABALL RENDERER                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _compute_field(blobs):
    field = np.zeros((SIM_H, SIM_W), np.float32)
    for b in blobs:
        dx = _XX - b.x;  dy = _YY - b.y
        field += (b.r * b.r) / (dx * dx + dy * dy + 0.1)
    return field


def _render_td_glob(field):
    """
    TouchDesigner-style liquid-metal metaball render.

    Emission sources (all additive, pure emission):
      1. Body      — subsurface depth cue: thicker regions glow warmer
      2. Specular  — fixed screen-space glint (metallic reflection)
      3. Rim       — narrow strip where field ≈ threshold, shaped by curvature
                        (convex bright, concave dim — liquid tension look)

    Bloom is applied at sim resolution (scaled sigma/kernel) then upsampled,
    giving the same visual result as display-res bloom at ~10× less cost.
    """
    # ── Surface normals & signed distance ──────────────────────────────────────
    gy, gx = np.gradient(field)
    gm = np.hypot(gx, gy).astype(np.float32)
    gm_s = np.maximum(gm, 1e-4)
    sd = ((field - META_THRESHOLD) / gm_s).astype(np.float32)   # sim units
    inside = field > META_THRESHOLD

    # Unit normals in screen space
    nx = (gx / gm_s).astype(np.float32)
    ny = (gy / gm_s).astype(np.float32)

    # Curvature = divergence of normal field (convex +, concave −)
    cnx = np.gradient(nx, axis=1)
    cny = np.gradient(ny, axis=0)
    curv = (cnx + cny).astype(np.float32)

    # ── Body glow (subsurface depth cue) ───────────────────────────────────────
    # sd blows up at blob centers (grad→0), so we use field value directly:
    # thicker regions (higher field) glow warmer. 0 at surface, ~1 deep inside.
    body = np.clip((field - META_THRESHOLD) / (META_THRESHOLD * BODY_DEPTH), 0.0, 1.0).astype(np.float32)

    # ── Specular glint (fixed screen-space light) ──────────────────────────────
    ndotl = (nx * _LIGHT[0] + ny * _LIGHT[1]).astype(np.float32)
    spec = np.where(inside, np.clip(ndotl, 0.0, 1.0) ** SPEC_SHININESS, 0.0).astype(np.float32)

    # ── Rim band with curvature response ───────────────────────────────────────
    rim_lo = META_THRESHOLD - RIM_BAND
    rim_hi = META_THRESHOLD + RIM_BAND * 0.25
    rim_t = np.clip((field - rim_lo) / (rim_hi - rim_lo), 0, 1)
    rim_t *= np.clip((rim_hi - field) / (rim_hi - rim_lo), 0, 1)
    # Convex (curv>0) → bright, concave (curv<0, e.g. neck between merging blobs) → dim
    curv_resp = np.clip(1.0 + curv * CURV_SCALE, CURV_MIN, CURV_MAX).astype(np.float32)

    # ── Composite emission at sim resolution ───────────────────────────────────
    em = np.zeros((SIM_H, SIM_W, 3), np.float32)
    em += body[:, :, np.newaxis] * BODY_STRENGTH * GLOB_FILL_COLOR[np.newaxis, np.newaxis, :]
    em += (rim_t * curv_resp)[:, :, np.newaxis] * GLOB_RIM_COLOR[np.newaxis, np.newaxis, :]
    em += spec[:, :, np.newaxis] * SPEC_STRENGTH
    em = np.clip(em, 0, 1.5)

    # ── Bloom at sim resolution (scaled params), then upscale ──────────────────
    # Kernel/sigma scaled down so the effective bloom at display size matches
    # the original display-res bloom (kernel=35, sigma=14) visually.
    _BK = max(3, 35 // SIM_SCALE)
    _BS = 14.0 / SIM_SCALE
    bloomed_sim = td.bloom(em, kernel=_BK, sigma=_BS, strength=1.3)
    bloomed = cv2.resize(bloomed_sim, (DISPLAY_W, DISPLAY_H),
                         interpolation=cv2.INTER_LANCZOS4)

    return bloomed


def _draw_sigil_lines_td(em, blobs, t):
    """
    Dim connecting lines between nearby blobs — pure emission, no color.
    TD aesthetic: data lines, not decoration.
    """
    n = len(blobs)
    for i in range(n):
        for j in range(i + 1, n):
            bx1 = int(blobs[i].x * SIM_SCALE)
            by1 = int(blobs[i].y * SIM_SCALE)
            bx2 = int(blobs[j].x * SIM_SCALE)
            by2 = int(blobs[j].y * SIM_SCALE)
            d = math.hypot(bx2 - bx1, by2 - by1)
            max_d = (blobs[i].r + blobs[j].r) * SIM_SCALE * 2.8
            if d < max_d:
                alpha = (1.0 - d / max_d) ** 2
                pulse = 0.5 + 0.5 * math.sin(t * 1.8 + i * 1.1 + j * 0.9)
                col = GLOB_SIGIL_COL * alpha * pulse
                td.draw_line_em(em, (bx1, by1), (bx2, by2), color=col)

    # Node at each blob centre
    for b in blobs:
        cx = int(b.x * SIM_SCALE);  cy = int(b.y * SIM_SCALE)
        pulse = 0.5 + 0.5 * math.sin(t * 2.5 + b.pulse_phase)
        rim = GLOB_RIM_COLOR * 0.6 * pulse
        td.draw_node_em(em, (cx, cy), r=3, core=td.TD_WHITE * 0.5, rim=rim)


def _split(blobs, pinch_sim):
    px, py = pinch_sim
    for b in blobs[:]:
        if len(blobs) >= BLOB_MAX_COUNT:
            break
        if math.hypot(px - b.x, py - b.y) < b.r * 0.9 and b.base_r > BLOB_R_MIN * 1.4:
            nr = b.base_r * 0.65
            a = math.atan2(py - b.y, px - b.x)
            p = a + math.pi / 2
            b1 = Blob(b.x + math.cos(p) * nr * .4, b.y + math.sin(p) * nr * .4, nr)
            b1.vx = b.vx + math.cos(p) * 1.5;  b1.vy = b.vy + math.sin(p) * 1.5
            b2 = Blob(b.x - math.cos(p) * nr * .4, b.y - math.sin(p) * nr * .4, nr)
            b2.vx = b.vx - math.cos(p) * 1.5;  b2.vy = b.vy - math.sin(p) * 1.5
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
        x = w - tw - 14;  y = h - 12 - i * 16
        cv2.putText(display, line, (x, y), font, sc, (50, 70, 55), 1, cv2.LINE_AA)
    cv2.putText(display, f"blobs  {n_blobs}", (14, 22),
                font, sc, (50, 70, 45), 1, cv2.LINE_AA)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_glob_mode(cam, det):
    SX_S = SIM_W / CAM_W;  SY_S = SIM_H / CAM_H
    SX_D = DISPLAY_W / CAM_W;  SY_D = DISPLAY_H / CAM_H

    blobs = [Blob() for _ in range(N_BLOBS)]
    split_cd = 0
    t = 0.0
    double_fist_ctr = [0]

    print("Glob Metal | both fists = menu | ESC = menu")

    while True:
        frame = cam.read()
        if frame is None:
            continue

        det.submit(cv2.resize(frame, (CAM_W, CAM_H)))
        res_frame, hand_data = det.get()

        # ── Hand data ────────────────────────────────────────────────────────────
        push_pts = []
        pull_pts = []
        pinch_sim = None

        if hand_data:
            for hand in hand_data:
                pts = hand['points']
                is_fist = hand['gestes']['poing']
                is_pinch = hand['gestes']['pincement']
                if is_fist:
                    pull_pts.append((pts[0][0] * SX_S, pts[0][1] * SY_S))
                else:
                    push_pts.append((pts[8][0] * SX_S, pts[8][1] * SY_S))
                if is_pinch:
                    pinch_sim = ((pts[4][0] + pts[8][0]) / 2 * SX_S,
                                 (pts[4][1] + pts[8][1]) / 2 * SY_S)

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

        # 2. Metaball bloom layer (gradient-based liquid metal shader)
        field = _compute_field(blobs)
        bloomed = _render_td_glob(field)   # float32 0-1, display size

        # 3. Sigil lines on a separate emission layer
        em_sigil = td.make_emission(DISPLAY_H, DISPLAY_W)
        _draw_sigil_lines_td(em_sigil, blobs, t)
        sigil_bloomed = td.bloom(em_sigil, kernel=15, sigma=5.0, strength=0.7)

        # 4. Hand indicator (very subtle — just a dim circle)
        em_hand = td.make_emission(DISPLAY_H, DISPLAY_W)
        if hand_data:
            for hand in hand_data:
                pts = hand['points']
                is_fist = hand['gestes']['poing']
                ix_d = int(pts[8][0] * SX_D);  iy_d = int(pts[8][1] * SY_D)
                r_vis = int(HAND_PUSH_RADIUS * SIM_SCALE * (1.4 if is_fist else 1.0))
                col = td.TD_CYAN * 0.18
                cv2.circle(em_hand, (ix_d, iy_d), r_vis,
                           (float(col[0]), float(col[1]), float(col[2])),
                           1, cv2.LINE_AA)

        # 5. Composite: cam + blobs + sigil + hand indicator
        result = np.clip(cam_dim + bloomed + sigil_bloomed + em_hand, 0, 1)
        display = (result * 255).astype(np.uint8)

        _draw_hud(display, len(blobs))

        # ── Double-fist → back to menu ────────────────────────────────────────────
        if gest.check_double_fist(hand_data, double_fist_ctr):
            break
        gest.draw_double_fist_hint(display, double_fist_ctr)

        t += 0.033
        cv2.imshow("Vision AI", display)
        if cv2.waitKey(1) == 27:
            break
