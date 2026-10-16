# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG.PY — single source of truth for the project                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CAM = {
    'source': 0,
    'display_w': 1280,
    'display_h': 720,

    # ── Détection ──────────────────────────────────────────────────────────────
    # Résolution de détection de base (640x360).
    # Si detect_auto=True, un benchmark au démarrage teste la résolution
    # supérieure (detect_hi_w/h) et choisit la plus fine que la machine
    # peut tenir sous le budget de latence (detect_latency_budget_ms).
    'detect_w': 640,
    'detect_h': 360,
    'detect_auto': True,
    'detect_hi_w': 1280,
    'detect_hi_h': 720,
    'detect_latency_budget_ms': 25,

    # Demander 60 fps à la caméra (silencieusement ignoré si non supporté)
    'camera_fps': 60,
    'buffer_size': 1,
}

DETECTION = {
    'num_hands': 2,
    'model_path': 'hand_landmarker.task',
    'min_hand_detection_confidence': 0.6,
    'min_hand_presence_confidence':  0.6,
    'min_tracking_confidence':       0.5,
}

SMOOTHING = {
    'min_cutoff': 1.0,
    'beta': 0.02,
    'd_cutoff': 1.0,
    'jump_ratio': 1.5,
}

HAND = {
    'pinch_ratio_on':  0.25,
    'pinch_ratio_off': 0.40,
    'fist_tip_mid_ratio': 0.90,
    'fist_fold_scale': 0.55,
    'fist_min_folded': 3,
    'debounce_on':  2,
    'debounce_off': 4,
    'min_wrist_separation': 100,
}

# ── Hand calibration (measured at startup) ─────────────────────────────────────
# reference_hand_scale is computed once during startup and used to normalise
# gesture thresholds. The raw _hand_scale() is still used as a fallback when
# calibration has not yet run (returns 1.0).
HAND_CALIB = {
    'reference_scale_px': 0.0,   # set at runtime by calibrate_hand()
}

GESTURE = {
    'double_fist_frames': 20,
    'finger_join_ratio': 0.20,
    'hand_proximity_px': 280,
    'spread_px_min': 10,
    'spread_px_max': 220,
    'speed_mult_min': 0.10,
    'speed_mult_max': 5.0,
}

MOUSE = {
    'cursor_min_cutoff': 1.0,
    'cursor_beta': 0.04,
    'cursor_d_cutoff': 1.0,
    'zone_x_min': 0.15,
    'zone_x_max': 0.85,
    'zone_y_min': 0.10,
    'zone_y_max': 0.90,
}

MENU = {
    'radius': 130,
    'icon_radius': 38,
    'carousel_sensitivity': 1.2,
}

SHAPES = {
    'sphere_rings': 10,
    'sphere_segments': 14,
    'sphere_color': (106, 50, 159),
    'pyramid_color': (0, 150, 255),
    'pyramid_radius_initial': 80,
    'max_shapes': 8,
    'triple_clic_fenetre': 0.75,
    'grab_range_mult': 1.5,
    'spawn_min_dist_mult': 2.0,
}

PHYSICS = {
    'lissage': 3.0,
    'friction': 0.85,
    'vitesse_min': 0.1,
    'rotation_sensibilite': 0.015,
    'rotation_auto_sphere': 0.003,
    'rotation_auto_pyramid_y': 0.04,
    'rotation_auto_pyramid_x': 0.02,
    'radius_min': 30,
    'radius_max': 400,
}

VISUALS = {
    'cam_dim': 0.18,
    'bg_tint': (0.06, 0.05, 0.03),
    'bloom_strength': 0.75,
    'bloom_kernel': 31,
    'bloom_sigma': 12.0,
    'td_white': (1.00, 1.00, 1.00),
    'td_cyan': (0.85, 0.95, 0.60),
    'td_blue': (0.55, 0.40, 0.15),
    'td_dim': (0.12, 0.10, 0.04),
}

GLOB = {
    'sim_scale': 4,
    'n_blobs': 6,
    'blob_r_min': 26,
    'blob_r_max': 52,
    'meta_threshold': 1.0,
    'rim_band': 0.35,
    'blob_max_count': 10,
    'split_cooldown': 18,
    'friction': 0.986,
    'wall_bounce': 0.60,
    'hand_push_radius': 75,
    'hand_push_force': 2.0,
    'hand_pull_force': 1.5,
    'blob_repulsion': 800,
    'blob_attract': 25,
    'speed_max': 5.5,
    'gravity': 0.055,
    'rim_color': (0.85, 0.95, 0.60),
    'fill_color': (0.04, 0.06, 0.02),
    'sigil_color': (0.30, 0.40, 0.15),
    # ── Liquid-metal shader (gradient-based) ────────────────────────────────────
    'spec_shininess': 20,   # specular highlight sharpness (pow exponent)
    'spec_strength':  0.85, # specular intensity
    'body_depth':     5.0,    # body glow falloff depth (field/ratio → 0..1)
    'body_strength':  1.20, # body fill brightness
    'curv_scale':     80.0, # curvature response multiplier
    'curv_min':       0.20, # min curvature clamp (concave neck darkening)
    'curv_max':       1.80, # max curvature clamp (convex edge brightening)
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  USER OVERRIDES (settings.py → settings.json)                              ║
# ║  Applied last: whatever the user saved in the settings GUI wins.           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _apply_user_overrides():
    import settings as _settings
    user = _settings.load()
    cam = user.get('cam')
    if cam:
        CAM.update(cam)
    sm = user.get('smoothing')
    if sm:
        SMOOTHING.update(sm)

_apply_user_overrides()
