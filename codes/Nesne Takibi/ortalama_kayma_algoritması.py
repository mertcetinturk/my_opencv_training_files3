import cv2
import numpy as np

# kamera aç
cap = cv2.VideoCapture(0)

# bir tane frame oku
ret, frame = cap.read()
if not ret:
    print('Uyarı!')

# detection
cascade = r'..\..\HaarCascade\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade)
face_rect = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rect[0])
track_window = (face_x, face_y, w, h)  # meanshif algoritması girdisi

# ROI belirlenmesi
roi = frame[face_y:face_y+h, face_x:face_x+w]  # roi = face

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])  # takip için histogram gerekli
cv2.normalize(roi_hist, roi_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# takip için gerekli durdurma kriterleri
# count = hesaplanacak öge sayısı
# epsilon = değişiklik
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # histogramı bir görüntüde bulmak için kullanıyoruz
        # piksel karşılaştırma
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], scale=1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        (x, y, w, h) = track_window

        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)

        cv2.imshow('Takip', img2)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
