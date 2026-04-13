from sklearn.model_selection import train_test_split
import numpy as np
import cv2

import PIL.Image as Image
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

import pathlib


classifier = tf.keras.applications.InceptionV3(
    weights='imagenet', 
    input_shape=(299, 299, 3)
)

img = Image.open("./Data/mushroom.jpg").convert('RGB')
img = img.resize((299, 299))
img = np.array(img) / 255.0  
img = np.expand_dims(img, axis=0)

predictions = classifier.predict(img)

# To see the actual labels instead of raw numbers:
decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=5)
print(decoded_predictions)



