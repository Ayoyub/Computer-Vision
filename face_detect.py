import cv2
import os

_cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(_cascade_path):
    _cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(_cascade_path)


def _draw_liquid_face(img, x, y, w, h):
    # Remplissage frosted blanc très doux
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (240, 240, 240), -1)
    cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)

    # Contour fin gris nacré
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1, cv2.LINE_AA)

    # Coins arrondis simulés
    r = 8
    for cx, cy in [(x + r, y + r), (x + w - r, y + r),
                   (x + r, y + h - r), (x + w - r, y + h - r)]:
        cv2.circle(img, (cx, cy), r, (200, 200, 200), 1, cv2.LINE_AA)

    # Reflet horizontal en haut
    reflet = img.copy()
    cv2.line(reflet, (x + r, y + 1), (x + w - r, y + 1), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(reflet, 0.55, img, 0.45, 0, img)

    # Label flottant minimaliste
    label = "face"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.42, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    lx, ly, pad = x + 6, y - 6, 4

    label_ov = img.copy()
    cv2.rectangle(label_ov,
                  (lx - pad, ly - th - pad),
                  (lx + tw + pad, ly + pad),
                  (20, 20, 20), -1)
    cv2.addWeighted(label_ov, 0.40, img, 0.60, 0, img)
    cv2.putText(img, label, (lx, ly), font, scale, (230, 230, 230), thick, cv2.LINE_AA)


def face_detect(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            _draw_liquid_face(img, x, y, w, h)
        return img, faces
    except Exception as e:
        print(f"face_detect error: {e}")
        return img, []