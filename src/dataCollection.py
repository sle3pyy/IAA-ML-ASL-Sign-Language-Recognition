import cv2
from HandTrackingModule import HandDetector
import numpy as np
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(num_hands=1)
imgSize = 300

folder = 'Data/collected'

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Detect and draw hands
    hands, img, bbox = detector.findHands(img)

    if bbox:
        x,y,w,h = bbox

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)

        imgCrop = img[y:y+h, x:x+w]
        # cv2.imshow("ImageCrop", imgCrop)
        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
        cv2.imshow("ImageResize", imgResize)
        

    cv2.imshow("Hand Tracking", img)

    # Send to pipeline (later)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgResize)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()