"""
settings.py — user settings persistence
════════════════════════════════════════
Stores user overrides (camera, fps, detection, smoothing) in a JSON file.
config.py applies them on top of its defaults, so every module that reads
config picks up the user's choices with zero other changes.

File location:
  $XDG_CONFIG_HOME/computer-vision/settings.json  (default ~/.config/...)
  Falls back to <project_dir>/user_settings.json if the config dir
  is not writable (e.g. read-only HOME).
"""
import json
import os
import sys
from pathlib import Path

# Sections and their valid keys (used to filter junk on load)
SCHEMA = {
    'cam': {
        'source', 'display_w', 'display_h',
        'detect_w', 'detect_h', 'detect_auto',
        'detect_hi_w', 'detect_hi_h', 'detect_latency_budget_ms',
        'camera_fps', 'buffer_size',
    },
    'smoothing': {
        'min_cutoff', 'beta', 'd_cutoff', 'jump_ratio',
    },
}

DEFAULT_FILE = None  # resolved lazily (import must stay side-effect-free)


def settings_path() -> Path:
    global DEFAULT_FILE
    if DEFAULT_FILE is not None:
        return DEFAULT_FILE
    try:
        base = Path(os.environ.get(
            'XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'computer-vision'
        base.mkdir(parents=True, exist_ok=True)
        # Probe writability
        probe = base / '.probe'
        probe.write_text('ok')
        probe.unlink()
        DEFAULT_FILE = base / 'settings.json'
    except OSError:
        DEFAULT_FILE = Path(__file__).parent / 'user_settings.json'
    return DEFAULT_FILE


def load() -> dict:
    """Load user overrides. Returns {} if the file is missing or invalid."""
    p = settings_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[settings] could not read {p}: {e} — using defaults")
        return {}
    if not isinstance(raw, dict):
        return {}
    # Filter to known keys only
    clean = {}
    for section, keys in SCHEMA.items():
        if section in raw and isinstance(raw[section], dict):
            clean[section] = {k: v for k, v in raw[section].items()
                              if k in keys and v is not None}
    return clean


def save(data: dict) -> Path:
    """Write user overrides. Returns the path written."""
    clean = {}
    for section, keys in SCHEMA.items():
        if section in data and isinstance(data[section], dict):
            clean[section] = {k: v for k, v in data[section].items()
                              if k in keys and v is not None}
    p = settings_path()
    p.write_text(json.dumps(clean, indent=2))
    return p


def reset() -> Path:
    """Delete the user settings file (back to pure defaults)."""
    p = settings_path()
    if p.exists():
        p.unlink()
    return p
