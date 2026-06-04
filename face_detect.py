import cv2
import os

# Load Haar cascade — prefer local file, fall back to OpenCV's built-in path
_cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(_cascade_path):
    _cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

_cascade = cv2.CascadeClassifier(_cascade_path)


def _draw_face(img, x, y, w, h):
    """Render a liquid-glass frame around a detected face."""

    # Frosted fill — barely visible white tint
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x+w, y+h), (240, 240, 240), -1)
    cv2.addWeighted(ov, 0.10, img, 0.90, 0, img)

    # Pearlescent border
    cv2.rectangle(img, (x, y), (x+w, y+h), (200, 200, 200), 1, cv2.LINE_AA)

    # Simulated rounded corners
    r = 8
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(img, (cx, cy), r, (200, 200, 200), 1, cv2.LINE_AA)

    # Top highlight — simulates glass reflection
    ov = img.copy()
    cv2.line(ov, (x+r, y+1), (x+w-r, y+1), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)

    # Floating label
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    label = "face"
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    lx, ly, pad = x+6, y-6, 4

    ov = img.copy()
    cv2.rectangle(ov, (lx-pad, ly-th-pad), (lx+tw+pad, ly+pad), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.40, img, 0.60, 0, img)
    cv2.putText(img, label, (lx, ly), font, scale, (230, 230, 230), thick, cv2.LINE_AA)


def face_detect(img):
    """
    Detect faces in a BGR frame and draw liquid-glass overlays.

    Returns:
        img   : annotated frame
        faces : list of (x, y, w, h) bounding boxes
    """
    try:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            _draw_face(img, x, y, w, h)
        return img, faces
    except Exception as e:
        print(f"[face_detect] error: {e}")
        return img, []