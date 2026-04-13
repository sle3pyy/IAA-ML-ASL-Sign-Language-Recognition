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
    input_shape=(299, 299, 3),
    include_top=False,
)

classifier.trainable = False

train_ds = tf.keras.utils.image_dataset_from_directory(
    "./Data/collected",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(299, 299),
    batch_size=16  # Smaller batch size to save memory
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "./Data/collected",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(299, 299),
    batch_size=16
)

# Get number of classes
n_letters = len(train_ds.class_names)


model = tf.keras.Sequential([
    classifier,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax') 
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, epochs=10, validation_data=val_ds)

model.save('model_v2.h5')