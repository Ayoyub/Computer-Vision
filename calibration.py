import cv2
import math
import json
from hand_detect import hand_detect, release
from config import CAM

CAM_W, CAM_H = CAM['detect_w'],  CAM['detect_h']
DW,   DH     = CAM['display_w'], CAM['display_h']
SX,   SY     = DW / CAM_W, DH / CAM_H

OUTPUT_FILE = 'calibration.json'

# Gap between the two measured midpoints:
#   seuil_paume = midpoint * (1 + MARGIN)  → needs to be clearly open
#   seuil_poing = midpoint * (1 - MARGIN)  → needs to be clearly closed
MARGIN = 0.15


def _dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def _d4_20(pts):
    """Thumb (4) ↔ pinky (20) distance — main open/close metric."""
    return _dist(pts[4], pts[20])


def _d_tips_wrist(pts):
    """Mean distance from the 4 fingertips to the wrist."""
    return sum(_dist(pts[i], pts[0]) for i in [8, 12, 16, 20]) / 4


def _draw_ui(img, title, hint, progress, n, target):
    """Render the calibration overlay: title, hint, progress bar."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent header band
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 110), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.70, img, 0.30, 0, img)

    cv2.putText(img, title, (20, 36), font, 0.80, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, hint,  (20, 68), font, 0.46, (160, 160, 160), 1, cv2.LINE_AA)

    # Progress bar
    bx, by, bw, bh = 20, 82, w-40, 10
    ov = img.copy()
    cv2.rectangle(ov, (bx, by), (bx+bw, by+bh), (40, 40, 40), -1)
    cv2.addWeighted(ov, 0.70, img, 0.30, 0, img)

    fill = int(bw * min(progress, 1.0))
    if fill > 0:
        ov = img.copy()
        cv2.rectangle(ov, (bx, by), (bx+fill, by+bh), (200, 200, 200), -1)
        cv2.addWeighted(ov, 0.80, img, 0.20, 0, img)

    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (100, 100, 100), 1)

    counter = f"{n}/{target}"
    (tw, _), _ = cv2.getTextSize(counter, font, 0.40, 1)
    cv2.putText(img, counter, (w-tw-20, by+bh+14), font, 0.40, (120, 120, 120), 1, cv2.LINE_AA)


def _draw_metrics(img, pts, d420, dmoy, label):
    """Show key landmarks and live measurements at the bottom of the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]

    for idx, color in {4:(255,200,100), 8:(100,255,100), 20:(100,200,255), 0:(200,200,200)}.items():
        cv2.circle(img, pts[idx], 7, color, -1, cv2.LINE_AA)
    cv2.line(img, pts[4], pts[20], (180, 180, 180), 1, cv2.LINE_AA)

    for i, text in enumerate([
        f"thumb <-> pinky   : {d420:.0f} px",
        f"tips  -> wrist avg: {dmoy:.0f} px",
        f"state             : {label}",
    ]):
        y = h - 60 + i*18
        (tw, th), _ = cv2.getTextSize(text, font, 0.38, 1)
        ov = img.copy()
        cv2.rectangle(ov, (10, y-th-2), (10+tw+6, y+4), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        cv2.putText(img, text, (13, y), font, 0.38, (170, 170, 170), 1, cv2.LINE_AA)


def _collect(cap, label, hint, n_target=60):
    """
    Collect n_target stable samples for a given pose.
    Skips samples that jump more than 40 px from the previous one.
    Returns (mean_d420, mean_dtips, std_d420) or None if cancelled.
    """
    s420, stips = [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame   = cv2.flip(frame, 1)
        small   = cv2.resize(frame, (CAM_W, CAM_H))
        display, hand_data = hand_detect(small)
        display = cv2.resize(display, (DW, DH))

        n = len(s420)

        if hand_data:
            pts_raw = hand_data[0]['points']
            pts     = [(int(x*SX), int(y*SY)) for x, y in pts_raw]
            d420    = _d4_20(pts)
            dtips   = _d_tips_wrist(pts)

            # Accept sample only if it is stable (no sudden jump)
            if n == 0 or abs(d420 - s420[-1]) < 40:
                s420.append(d420)
                stips.append(dtips)

            _draw_metrics(display, pts, d420, dtips, label)
        else:
            dummy = {k: (DW//2, DH//2) for k in [0, 4, 8, 20]}
            _draw_metrics(display, dummy, 0, 0, "no hand detected")

        _draw_ui(display, f"Step: {label}", hint, n/n_target, n, n_target)

        if n >= n_target:
            mean = sum(s420) / n
            std  = (sum((x-mean)**2 for x in s420) / n) ** 0.5
            mean_tips = sum(stips) / n

            # Green flash — pose recorded
            ov = display.copy()
            cv2.rectangle(ov, (0, 0), (DW, DH), (100, 200, 100), -1)
            cv2.addWeighted(ov, 0.15, display, 0.85, 0, display)

            font = cv2.FONT_HERSHEY_SIMPLEX
            msg  = f"OK — mean: {mean:.0f}px  std: {std:.0f}px"
            (tw, _), _ = cv2.getTextSize(msg, font, 0.48, 1)
            cv2.putText(display, msg,
                        (DW//2-tw//2, DH//2), font, 0.48, (200,240,200), 1, cv2.LINE_AA)
            cv2.putText(display, "Press SPACE to continue",
                        (DW//2-110, DH//2+30), font, 0.42, (150,150,150), 1, cv2.LINE_AA)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(0)
            if key == 32:   # Space
                return mean, mean_tips, std
            elif key == 27: # Esc
                return None

        cv2.imshow("Calibration", display)
        if cv2.waitKey(1) == 27:
            return None


def run():
    cap = cv2.VideoCapture(CAM['source'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DH)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("── Gesture Calibration ──────────────────────────")
    print("  Two poses, ~2 seconds each.")
    print("  Results saved to calibration.json")
    print("  Esc to cancel at any time.")
    print("─────────────────────────────────────────────────")

    # Step 1 — open palm
    r_palm = _collect(cap,
        label    = "OPEN PALM",
        hint     = "Spread fingers, palm facing camera — hold still",
        n_target = 60,
    )
    if r_palm is None:
        print("Cancelled."); cap.release(); release(); cv2.destroyAllWindows(); return
    mean_palm, _, _ = r_palm

    # Step 2 — closed fist
    r_fist = _collect(cap,
        label    = "CLOSED FIST",
        hint     = "Close fist tightly, thumb to the side — hold still",
        n_target = 60,
    )
    if r_fist is None:
        print("Cancelled."); cap.release(); release(); cv2.destroyAllWindows(); return
    mean_fist, _, _ = r_fist

    # Compute thresholds from the midpoint with a safety margin
    midpoint    = (mean_palm + mean_fist) / 2
    seuil_paume = midpoint * (1 + MARGIN)
    seuil_poing = midpoint * (1 - MARGIN)

    # Safety: ensure the two thresholds don't cross
    if seuil_paume <= seuil_poing:
        seuil_paume = mean_palm * 0.75
        seuil_poing = mean_fist * 1.25

    data = {
        '_comment'   : "Auto-generated by calibration.py",
        'mean_palm'  : round(mean_palm, 1),
        'mean_fist'  : round(mean_fist, 1),
        'midpoint'   : round(midpoint,  1),
        'seuil_paume': round(seuil_paume, 1),  # d_4_20 > this → open palm
        'seuil_poing': round(seuil_poing, 1),  # d_4_20 < this → closed fist
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    # Confirmation screen
    ok, frame = cap.read()
    display = cv2.resize(cv2.flip(frame, 1), (DW, DH)) if ok else \
              __import__('numpy').zeros((DH, DW, 3), dtype='uint8')

    ov = display.copy()
    cv2.rectangle(ov, (0, 0), (DW, DH), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.75, display, 0.25, 0, display)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (text, scale, color) in enumerate([
        ("Calibration complete!", 0.65, (220,220,220)),
        ("", 0.44, (140,140,140)),
        (f"palm threshold  : d > {seuil_paume:.0f} px", 0.44, (140,140,140)),
        (f"fist threshold  : d < {seuil_poing:.0f} px", 0.44, (140,140,140)),
        (f"midpoint        : {midpoint:.0f} px",         0.44, (140,140,140)),
        ("", 0.44, (140,140,140)),
        (f"Saved to: {OUTPUT_FILE}",                     0.44, (140,140,140)),
        ("", 0.44, (140,140,140)),
        ("Press any key to exit",                        0.44, (140,140,140)),
    ]):
        y = DH//2 - 110 + i*28
        (tw, _), _ = cv2.getTextSize(text, font, scale, 1)
        cv2.putText(display, text, (DW//2-tw//2, y), font, scale, color, 1, cv2.LINE_AA)

    cv2.imshow("Calibration", display)
    cv2.waitKey(0)

    print(f"\npalm threshold : {seuil_paume:.0f} px (d_4_20 > this)")
    print(f"fist threshold : {seuil_poing:.0f} px (d_4_20 < this)")
    print(f"Saved to       : {OUTPUT_FILE}")

    cap.release()
    release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()