"""
setup_check.py — pre-flight dependency & environment check
══════════════════════════════════════════════════════════════
Runs before main.py starts. Detects platform, checks dependencies,
and shows clear install instructions if anything is missing.
Also offers auto-install on Windows (where it's safe).
"""
import sys
import shutil
import subprocess
import platform

WARN = "\033[93m"  # yellow
OK   = "\033[92m"  # green
ERR  = "\033[91m"  # red
RESET = "\033[0m"


def _check(cmd, name):
    return shutil.which(cmd) is not None


def _has_module(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _try_install_windows(mod, pkg):
    """Attempt automatic install on Windows (safe, user has pip)."""
    if sys.platform != 'win32':
        return False
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _has_module(mod)
    except Exception:
        return False


def check():
    issues = []
    fixes = []
    tips = []
    plat = platform.system()  # Windows, Linux, Darwin

    print(f"\n{'='*50}")
    print(f"  Computer-Vision — Platform: {plat} {'(' + sys.platform + ')'}")
    print(f"{'='*50}\n")

    # ── Core Python deps ────────────────────────────────────────────────────────
    core = [
        ('cv2',      'opencv-contrib-python',  'OpenCV (camera + rendering)'),
        ('mediapipe','mediapipe',              'MediaPipe (hand detection)'),
        ('numpy',    'numpy',                  'NumPy (math / rendering)'),
    ]
    for mod, pkg, desc in core:
        if _has_module(mod):
            print(f"  {OK}✓{RESET} {desc}")
        else:
            issues.append((pkg, desc))
            print(f"  {ERR}✗{RESET} {desc} — missing '{mod}'")

    # ── Mouse input backend ─────────────────────────────────────────────────────
    mouse_backend = None
    if plat == 'Windows':
        if _has_module('pynput'):
            mouse_backend = 'pynput'
            print(f"  {OK}✓{RESET} Mouse backend: pynput (Windows native)")
        else:
            issues.append(('pynput>=1.7', 'Mouse input (pynput on Windows)'))
            print(f"  {ERR}✗{RESET} Mouse backend: pynput missing")
            if _try_install_windows('pynput', 'pynput>=1.7'):
                fixes.append('pynput')
                print(f"  {OK}→{RESET} Auto-installed pynput")
            else:
                print(f"  {WARN}  Install: python -m pip install pynput{RESET}")

    elif plat == 'Linux':
        # Try ydotool first (Wayland + X11)
        if _check('ydotool', 'ydotool'):
            mouse_backend = 'ydotool'
            # Check ydotoold daemon
            try:
                subprocess.run(['systemctl', '--user', 'is-active', 'ydotoold'],
                               capture_output=True, timeout=2)
                print(f"  {OK}✓{RESET} Mouse backend: ydotool + ydotoold (Wayland/X11)")
            except Exception:
                tips.append("ydotool found but ydotoold not running — start it with: systemctl --user enable --now ydotoold")
                print(f"  {WARN}⚠{RESET} Mouse backend: ydotool found but ydotoold may not be running")
        elif _has_module('pynput'):
            mouse_backend = 'pynput'
            print(f"  {OK}✓{RESET} Mouse backend: pynput (X11)")
        else:
            issues.append(('ydotool or pynput', 'Mouse input'))
            print(f"  {ERR}✗{RESET} Mouse backend: none found")
            print(f"  {WARN}  For Wayland (KDE/GNOME): sudo dnf install ydotool")
            print(f"  {WARN}  For X11:            python -m pip install pynput{RESET}")

    elif plat == 'Darwin':  # macOS
        if _has_module('pynput'):
            mouse_backend = 'pynput'
            print(f"  {OK}✓{RESET} Mouse backend: pynput (macOS native)")
        else:
            issues.append(('pynput>=1.7', 'Mouse input (macOS)'))
            print(f"  {ERR}✗{RESET} Mouse backend: pynput missing")
            print(f"  {WARN}  Install: python -m pip install pynput{RESET}")
    else:
        tips.append(f"Unknown platform '{plat}' — install pynput manually if mouse mode is needed")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print()
    if not issues:
        print(f"  {OK}All checks passed — ready to run!{RESET}\n")
        if fixes:
            print(f"  {OK}Auto-installed: {', '.join(fixes)}{RESET}")
        if tips:
            for t in tips:
                print(f"  {WARN}ℹ{RESET} {t}")
        return True
    else:
        print(f"  {ERR}Issues found:{RESET}")
        for pkg, desc in issues:
            print(f"    • {desc} — need '{pkg}'")
        if tips:
            for t in tips:
                print(f"  {WARN}ℹ{RESET} {t}")
        print()
        print("  Run again after installing missing deps.\n")
        return False


if __name__ == '__main__':
    ok = check()
    sys.exit(0 if ok else 1)
