import cv2
import os

_cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(_cascade_path):
    _cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(_cascade_path)

# Liquid glass palette
FACE_COLOR       = (230, 210, 255)   # lavande clair (BGR)
FACE_COLOR_INNER = (200, 170, 255)
FACE_TEXT_COLOR  = (255, 255, 255)
FACE_SHADOW      = (120, 80, 160)


def _draw_liquid_face(img, x, y, w, h):    
   overlay = img.copy()
   cv2.rectangle(overlay, (x, y), (x + w, y + h), (240, 240, 240), -1)
   cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)
 
    # ── Contour principal ──
   cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1, cv2.LINE_AA)
 
    # ── Coins arrondis simulés ──
   r = 8
   corners = [(x + r, y + r), (x + w - r, y + r),
               (x + r, y + h - r), (x + w - r, y + h - r)]
   for cx, cy in corners:
      cv2.circle(img, (cx, cy), r, (200, 200, 200), 1, cv2.LINE_AA)
 
    # ── Reflet en haut (ligne fine brillante) ──
   reflet = img.copy()
   cv2.line(reflet, (x + r, y + 1), (x + w - r, y + 1), (255, 255, 255), 1, cv2.LINE_AA)
   cv2.addWeighted(reflet, 0.55, img, 0.45, 0, img)
 
    # ── Label flottant ──
   label = "face"
   font  = cv2.FONT_HERSHEY_SIMPLEX
   scale = 0.42
   thick = 1
   (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
 
   lx = x + 6
   ly = y - 6
   pad = 4
 
    # Fond du label semi-opaque
   label_ov = img.copy()
   cv2.rectangle(label_ov,
                 (lx - pad, ly - th - pad),
                 (lx + tw + pad, ly + pad),
                 (20, 20, 20), -1)
   cv2.addWeighted(label_ov, 0.40, img, 0.60, 0, img)
 
    # Texte blanc fin
   cv2.putText(img, label, (lx, ly), font, scale,
                (230, 230, 230), thick, cv2.LINE_AA)


def face_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        _draw_liquid_face(img, x, y, w, h)

    return img, faces