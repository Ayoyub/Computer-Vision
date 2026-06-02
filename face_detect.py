import cv2
import os

# Define constants for the face detection and image processing
FACE_COLOR = (230, 210, 255)
FACE_COLOR_INNER = (200, 170, 255)
FACE_TEXT_COLOR = (255, 255, 255)
FACE_SHADOW = (120, 80, 160)

_cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(_cascade_path):
    _cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(_cascade_path)

def _draw_liquid_face(img, x, y, w, h):
    """
    Draw a liquid glass effect around the detected face.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (240, 240, 240), -1)
    cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)

    # Draw the contour of the face
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1, cv2.LINE_AA)

    # Draw the simulated rounded corners
    r = 8
    corners = [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]
    for cx, cy in corners:
        cv2.circle(img, (cx, cy), r, (200, 200, 200), 1, cv2.LINE_AA)

    # Draw the reflection at the top
    reflet = img.copy()
    cv2.line(reflet, (x + r, y + 1), (x + w - r, y + 1), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(reflet, 0.55, img, 0.45, 0, img)

    # Draw the floating label
    label = "face"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    lx = x + 6
    ly = y - 6
    pad = 4
    label_ov = img.copy()
    cv2.rectangle(label_ov, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad), (20, 20, 20), -1)
    cv2.addWeighted(label_ov, 0.40, img, 0.60, 0, img)
    cv2.putText(img, label, (lx, ly), font, scale, (230, 230, 230), thick, cv2.LINE_AA)

def face_detect(img):
    """
    Detect faces in the image and return the image with the detected faces and the coordinates of the faces.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            _draw_liquid_face(img, x, y, w, h)
        return img, faces
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None