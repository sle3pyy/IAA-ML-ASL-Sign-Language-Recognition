import cv2
from HandTrackingModule import HandDetector
import numpy as np
import time
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--visuals', action='store_true', default=False, help='Display hand visuals (landmarks and bounding box)')
args = parser.parse_args()

cap = cv2.VideoCapture(0)
detector = HandDetector(num_hands=1)
imgSize = 299
show_visuals = args.visuals

folder = 'Data/collected'


model = load_model("model_v2.h5", custom_objects={'KerasLayer': hub.KerasLayer})


while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Detect and draw hands
    hands, img, bbox = detector.findHands(img, draw=show_visuals)

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
        #cv2.imwrite(f'{folder}/C/Image_{time.time()}.jpg', imgResize)

        imgResize = np.expand_dims(imgResize, axis=0)
        imgResize = imgResize / 255

        prediction = model.predict(imgResize)
        print(prediction)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()