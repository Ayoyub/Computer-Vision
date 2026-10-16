# Computer-Vision

Hand-controlled virtual reality on a plain webcam — built for HackClub StarDance Challenge 2026.

## Modes
| Menu option | What it does |
|---|---|
| **Shapes** | Spawn & throw 3D wireframe shapes (sphere / pyramid). Pinch = grab, fist = rotate, 2 hands = scale & move, triple pinch = delete, 2-hand scissor spread = rotation speed. |
| **Mouse** | Your index finger drives the system cursor; pinch = click. `C` = recalibrate the active zone. |
| **Glob** | Liquid-metal metaball blob you can push (open hand), attract (fist) and split (pinch). |
| **Quit** | Exit. |

**Double fist = back to the radial menu** in every mode. `ESC` = menu / quit.

### Settings GUI
```bash
uv run python main.py --settings
```
Opens a tkinter configurator to pick your camera, frame rate, display +
detection resolution, and smoothing. Saved to
`~/.config/computer-vision/settings.json` and applied on every launch.

## Run
```bash
# Clone & enter
git clone https://github.com/Ayoyub/Computer-Vision
cd Computer-Vision

# Install deps once
uv sync

# Configure (optional): pick camera, frame rate, display + detection
# resolution, and smoothing. Saves to ~/.config/computer-vision/settings.json
uv run python main.py --settings

# Run
uv run python main.py
```

The MediaPipe model (`hand_landmarker.task`, ~7.5 MB) is downloaded automatically on first launch.

On Linux, `pynput`'s `evdev` backend must compile against Python's headers. If you get a build error about `Python.h`, install the development package and retry:
```bash
# Fedora / Nobara
sudo dnf install python3-devel
uv sync
```

### Mouse mode on Wayland
For native Wayland input, use ydotool instead of pynput:
```bash
# Fedora / Nobara
sudo -S -p '' dnf install ydotool
sudo -S -p '' udevadm control --reload-rules && sudo -S -p '' udevadm trigger
sudo -S -p '' bash -c 'echo "KERNEL==\"uinput\", MODE=\"0666\"" > /etc/udev/rules.d/99-uinput.rules'
# reboot (or re-login), then:
systemctl --user enable --now ydotoold
```

The app auto-detects: ydotool → pynput → none (mouse mode disabled gracefully).

### Windows
```cmd
git clone https://github.com/Ayoyub/Computer-Vision
cd Computer-Vision
uv sync
uv run python main.py
```
Mouse input uses pynput natively on Windows (no extra setup).

## Project layout
```
main.py          — radial menu + state machine + shapes mode
shapes_3d.py     — interactive 3D wireframe shapes (TouchDesigner bloom)
hand_detect.py   — MediaPipe detection, stable hand IDs, gestures
kalman_filter.py — One Euro filter (best-in-class jitter/lag tradeoff)
gestures.py      — multi-hand gestures (double-fist, scissor spread)
virt_mouse.py    — cursor control (ydotool / pynput backend)
glob_metal.py    — liquid-metal metaball simulation
td_render.py     — shared bloom / emission rendering pipeline
config.py        — every tunable constant lives here
settings.py      — user settings persistence (JSON, applied on import)
settings_gui.py  — tkinter configurator (--settings flag)
```

## Why these algorithms?
- **MediaPipe HandLandmarker (VIDEO mode)** — benchmarked 2.35× faster on CPU
  than YOLOv11-Pose for hand landmarks (IJSET 2025 dual-pipeline study), and
  far more stable in low light.
- **One Euro filter** (Casiez et al., CHI 2012) — the standard for smoothing
  gesture-control signals: zero-lag at high speed, jitter-free at rest.
  It replaces the fixed-gain Kalman filter and gives strictly better feel.
- **Schmitt-trigger pinch + temporal debounce** — the standard hysteresis
  approach for click gestures (used in production touchless pointer projects)
  to prevent click chatter.
- **Stable hand identity** — hands are matched across frames by wrist
  proximity, so grabbing a shape keeps following the *same* hand even when
  MediaPipe swaps its internal index order.
