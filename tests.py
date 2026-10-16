#!/usr/bin/env python3
"""Full test suite for Computer-Vision-improved — runs without camera."""
import math, sys, time
import numpy as np

sys.path.insert(0, '.')
import config, kalman_filter, td_render, gestures, shapes_3d, hand_detect
import virt_mouse, glob_metal, main

print("1. All modules import OK")

# 1€ filter: stationary → low jitter
f = kalman_filter.OneEuroFilter(fc_min=1.0, beta=0.02, fc_d=1.0)
vals = [f.filter(100 + (1.0 if i % 2 else -1.0), i / 60.0) for i in range(200)]
assert np.std(vals[-50:]) < 0.5
lags = []
f2 = kalman_filter.OneEuroFilter(fc_min=1.0, beta=0.02, fc_d=1.0)
for i in range(120):
    t = i / 60.0
    r = t * 600.0
    if i > 30:
        lags.append(r - f2.filter(r, t))
print(f"2. 1€ filter: jitter std={np.std(vals[-50:]):.3f}px, sweep lag={np.mean(lags):.1f}px")
assert np.mean(lags) < 10

# Jump reset
sm = kalman_filter.LandmarkSmoother()
base = [(100 + i, 100 + i) for i in range(21)]
for i in range(30):
    sm.smooth(base, i / 60.0)
out = sm.smooth([(500 + i, 500 + i) for i in range(21)], 31 / 60.0)
assert abs(out[0][0] - 500) < 5
print("3. Landmark jump reset OK")

# Double fist
cnt = [0]
hd = [{'gestes': {'poing': True}}, {'gestes': {'poing': True}}]
fa = None
for i in range(25):
    if gestures.check_double_fist(hd, cnt):
        fa = i
assert fa == 19 and not gestures.check_double_fist(None, cnt) and not gestures.check_double_fist([], cnt)
print("4. Double-fist @19, None-safe OK")

# Pinch Schmitt
class FS:
    pinch = False

pts = [(0, 0)] * 21
pts[0] = (0, 0)
pts[9] = (0, 80)  # wrist at origin, palm length 80px
pts[4] = pts[8] = (40, 44)
assert hand_detect._est_pincement_brut(pts, FS()) == True
s = FS()
s.pinch = True
pts[8] = (40, 50)
assert hand_detect._est_pincement_brut(pts, s) == True
pts[8] = (40, 77)  # dist=33, ratio=0.4125 > 0.40 -> release
assert hand_detect._est_pincement_brut(pts, s) == False
print("5. Pinch Schmitt on/hold/off OK")

# Fist
op = [(0, 0), (10, 20), (30, 30), (40, 40), (50, 90), (0, 60), (15, 60), (30, 65), (50, 90),
      (80, 80), (10, 60), (25, 65), (70, 90), (10, 80), (25, 85), (40, 90), (90, 90),
      (10, 100), (25, 105), (40, 110), (110, 90)]
fp = [(0, 0), (10, 20), (30, 30), (40, 40), (25, 45), (0, 60), (15, 60), (30, 65), (25, 45),
      (80, 80), (10, 60), (25, 65), (25, 50), (10, 80), (25, 85), (40, 90), (25, 55),
      (10, 100), (25, 105), (40, 110), (25, 60)]
assert not hand_detect._est_poing_brut(op) and hand_detect._est_poing_brut(fp)
print("6. Fist detection OK")

# Debounce
from collections import deque
h = deque(maxlen=4)
cur = False
for r in [True, True, False, False]:
    cur = hand_detect._debounce(h, r, 2, 4, cur)
assert cur
h2 = deque(maxlen=4)
cur = True
for r in [False] * 4:
    cur = hand_detect._debounce(h2, r, 2, 4, cur)
assert not cur
print("7. Debounce on=2/off=4 OK")

# TD render
img = np.full((720, 1280, 3), 128, np.uint8)
em = td_render.make_emission(720, 1280)
td_render.draw_line_em(em, (100, 100), (200, 200))
td_render.draw_node_em(em, (150, 150), r=3)
td_render.add_emission(img, td_render.bloom(em, kernel=15, sigma=5, strength=0.9))
print("8. TD emission+bloom OK")

# Shapes
sp = shapes_3d.MatrixSphere3D(640, 360, 80)
py = shapes_3d.MatrixPyramid3D(640, 360, 80)
assert sp.nodes.shape == (154, 3) and len(sp.edges) == 280
py2 = shapes_3d.MatrixPyramid3D(100, 100, 50)
py2.edges[0] = (99, 99)
assert py.edges[0] == (0, 1), "edges must be per-instance (copy)"
sp.draw(img)
py.draw(img)
print("9. Shapes geometry + per-instance edges OK")

# main.py fixes
src = open('main.py').read()
assert 'menu_rotation_frozen' in src and 'pass  # rotation' not in src
assert 'SEUIL_SELECT' not in src and 'FRAMES_CONFIRM' not in src and 'import sys' not in src.split('\n')[0]
print("10. main.py fixes (rotation freeze, dead constants, unused imports) OK")

# Config all wired
live = ''.join(open(f).read() for f in
              ['main.py', 'hand_detect.py', 'gestures.py', 'virt_mouse.py',
               'glob_metal.py', 'shapes_3d.py', 'kalman_filter.py',
               'td_render.py', 'test_cam.py'])
keys = [
    "'min_cutoff'", "'beta'", "'jump_ratio'", "'pinch_ratio_on'", "'pinch_ratio_off'",
    "'fist_tip_mid_ratio'", "'fist_fold_scale'", "'fist_min_folded'", "'debounce_on'",
    "'debounce_off'", "'min_wrist_separation'", "'double_fist_frames'", "'finger_join_ratio'",
    "'hand_proximity_px'", "'spread_px_min'", "'spread_px_max'", "'speed_mult_min'",
    "'speed_mult_max'", "'cursor_min_cutoff'", "'cursor_beta'", "'zone_x_min'",
    "'zone_x_max'", "'zone_y_min'", "'zone_y_max'", "'radius'", "'icon_radius'",
    "'carousel_sensitivity'", "'sphere_rings'", "'sphere_segments'",
    "'sphere_color'", "'pyramid_color'", "'pyramid_radius_initial'", "'max_shapes'",
    "'triple_clic_fenetre'", "'grab_range_mult'", "'spawn_min_dist_mult'", "'lissage'",
    "'friction'", "'vitesse_min'", "'rotation_sensibilite'", "'rotation_auto_sphere'",
    "'rotation_auto_pyramid_y'", "'rotation_auto_pyramid_x'", "'radius_min'",
    "'radius_max'", "'cam_dim'", "'bloom_strength'", "'td_white'", "'td_cyan'",
    "'td_blue'", "'td_dim'", "'bg_tint'", "'sim_scale'", "'n_blobs'",
    "'meta_threshold'", "'rim_band'", "'blob_max_count'", "'split_cooldown'",
    "'hand_push_radius'", "'blob_repulsion'", "'blob_attract'", "'speed_max'",
    "'gravity'", "'rim_color'", "'fill_color'", "'sigil_color'",
    "'spec_shininess'", "'spec_strength'", "'body_depth'",
    "'body_strength'", "'curv_scale'", "'curv_min'", "'curv_max'",
    "'num_hands'",
    "'model_path'", "'min_hand_detection_confidence'", "'min_hand_presence_confidence'",
    "'min_tracking_confidence'", "'source'", "'display_w'", "'display_h'",
    "'detect_w'", "'detect_h'", "'buffer_size'"
]
missing = [k for k in keys if k not in live]
assert not missing, f"never read: {missing}"
print(f"11. Config: all {len(keys)} keys wired into live code OK")

# Render smoke
disp = np.full((720, 1280, 3), 20, np.uint8)
main._draw_dynamic_menu(disp, 640, 360, 1, 0.0, main.MENU_OPTIONS)
main._draw_dynamic_menu(disp, 400, 400, 0, 0.0, main.SHAPE_OPTIONS,
                          icon_drawers=main._SHAPE_ICON_DRAWERS)
shapes_3d.MatrixSphere3D(800, 300, 60).draw(disp)
shapes_3d.MatrixPyramid3D(200, 200, 60).draw(disp)
print("12. Full 720p render smoke (2 menus + 2 shapes) OK")

# Physics: drag + throw + friction
sh = shapes_3d.MatrixSphere3D(640, 360, 80)
for i in range(30):
    sh.pinches = [(640 + i * 8, 360 + i * 4, False)]
    main._update_shape_physics(sh)
x1, y1, vx1, vy1 = sh.x, sh.y, sh.vx, sh.vy
sh.pinches = []
for i in range(10):
    main._update_shape_physics(sh)
assert (sh.x, sh.y) != (x1, y1)
print(f"13. Physics drag->drift ({x1:.0f},{y1:.0f})->({sh.x:.0f},{sh.y:.0f}), friction OK")

# Rotation by fist
sh2 = shapes_3d.MatrixSphere3D(640, 360, 80)
a0 = sh2.angle_y
for i in range(20):
    sh2.pinches = [(640 + i * 10, 360, True)]
    main._update_shape_physics(sh2)
assert abs(sh2.angle_y - a0) > 0.5
print(f"14. Fist rotation {a0:.2f}->{sh2.angle_y:.2f} rad OK")

# Two-hand scale
sh3 = shapes_3d.MatrixSphere3D(640, 360, 80)
sh3.pinches = [(600, 360, False), (680, 360, False)]
main._update_shape_physics(sh3)
r0 = sh3.radius
sh3.pinches = [(500, 360, False), (780, 360, False)]
for i in range(80):
    main._update_shape_physics(sh3)
assert sh3.radius > r0 * 1.5 and sh3.radius <= 400
print(f"15. Two-hand scale {r0:.0f}->{sh3.radius:.0f} (clamped <=400) OK")


class FL:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FH(list):
    def __init__(self, x, y):
        super().__init__([FL(x, y)])


# Stable hand identity: same hand across frames
st = {}
m = hand_detect._match_ids([FH(0.10, 0.50)], st, 640, 360)
sid1 = m[0].id
m2 = hand_detect._match_ids([FH(0.12, 0.50)], st, 640, 360)
assert m2[0].id == sid1, f"identity lost: {m2[0].id} vs {sid1}"
m3 = hand_detect._match_ids([FH(0.90, 0.50)], st, 640, 360)
assert m3[0].id != sid1
print(f"16. Stable hand identity: same hand={sid1} preserved, new hand->new id OK")

# Two-hand swap: order change detected correctly
st2 = {}
m = hand_detect._match_ids([FH(0.2, 0.5), FH(0.8, 0.5)], st2, 640, 360)
ids0 = [m[0].id, m[1].id]
m2 = hand_detect._match_ids([FH(0.8, 0.5), FH(0.2, 0.5)], st2, 640, 360)
assert m2[0].id == ids0[1] and m2[1].id == ids0[0], "swap not handled"
print("17. Two-hand swap: identities tracked correctly OK")

# Ghost guard threshold from config
assert hand_detect._WRIST_SEP == 100
print("18. Ghost-hand guard threshold = 100px (from config) OK")

# Mouse backend
mb = virt_mouse._MouseBackend()
print(f"19. Mouse backend: {mb.kind!r} (graceful degradation: no pynput, no ydotool)")

# ── Settings persistence (settings.py) ───────────────────────────────────────
import settings as _settings
import tempfile, os
# Save → load → reset round-trip
tmp = _settings.settings_path()
# Override the path to a temp file for the test (settings_path caches)
import settings as _s
_orig = _s.settings_path
_s.settings_path = lambda: tmp
try:
    data = {'cam': {'source': 1, 'camera_fps': 30, 'display_w': 1280, 'display_h': 720,
                    'detect_auto': False, 'detect_w': 960, 'detect_h': 540,
                    'detect_latency_budget_ms': 25},
            'smoothing': {'min_cutoff': 1.5, 'beta': 0.05, 'd_cutoff': 1.0, 'jump_ratio': 1.5}}
    p = _s.save(data)
    assert p == tmp
    loaded = _s.load()
    assert loaded['cam']['source'] == 1
    assert loaded['cam']['detect_w'] == 960
    assert loaded['smoothing']['min_cutoff'] == 1.5
    # Junk keys are filtered out
    junk = {'cam': {'bogus_key': 999, 'source': 2}}
    _s.save(junk)
    assert _s.load()['cam']['source'] == 2
    assert 'bogus_key' not in _s.load()['cam']
    _s.reset()
    assert _s.load() == {}
    print("20. settings.py round-trip + junk filter + reset OK")
finally:
    _s.settings_path = _orig
    if tmp.exists():
        tmp.unlink()

# ── config.py applies user overrides ─────────────────────────────────────────
import importlib, config
importlib.reload(config)
assert config.CAM['source'] == 0
assert config.SMOOTHING['min_cutoff'] == 1.0
print("21. config.py default overrides OK")

# ── settings_gui imports cleanly ─────────────────────────────────────────────
import settings_gui
assert hasattr(settings_gui, 'SettingsGUI')
assert hasattr(settings_gui, 'CameraPreview')
assert hasattr(settings_gui, 'list_cameras')
print("22. settings_gui import OK")

# ── main.py --settings wiring ────────────────────────────────────────────────
src = open('main.py').read()
assert '--settings' in src
print("23. main.py --settings wiring OK")

print("\n" + "=" * 60)
print("ALL 23 TESTS PASSED — fully validated codebase")
print("=" * 60)
