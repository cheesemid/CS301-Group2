#!/usr/bin/env python3


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import random 
from cv2 import cv2
import imutils
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import utils # from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation, Flatten, Dense,MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization




model = None
train = []
validation = []

dataset_path = "dataset"



def init_model():
    global model

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape=(32,32,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(35, activation='softmax'))

    # model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
    # history = model.fit(train_X,train_Y, epochs=50, batch_size=32, validation_data = (val_X, val_Y),  verbose=1)

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

# def get_letters(img):
#     letters = []
#     image = cv2.imread(img)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
#     dilated = cv2.dilate(thresh1, None, iterations=2)

#     cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     cnts = sort_contours(cnts, method="left-to-right")[0]
#     # loop over the contours
#     for c in cnts:
#         if cv2.contourArea(c) > 10:
#             (x, y, w, h) = cv2.boundingRect(c)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         roi = gray[y:y + h, x:x + w]
#         thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#         thresh = cv2.resize(thresh, (32, 32),  interpolation = cv2.INTER_CUBIC)
#         thresh = thresh.astype("float32") / 255.0
#         thresh = np.expand_dims(thresh, axis=-1)
#         thresh = thresh.reshape(1,32,32,1)
#         ypred = model.predict(thresh)
#         ypred = LB.inverse_transform(ypred)
#         [x] = ypred
#         letters.append(x)
#     return letters, image

# def get_word(letter):
#     word = "".join(letter)
#     return word
    
# letter,image = get_letters("path of the directory")
# word = get_word(letter)
# print(word)

def import_dataset():
    # Load Train Dataset
    for letter in os.listdir(os.path.join(dataset_path, "Train")):
        print(f"Train Dataset: Loading Letter {letter}")
        for image_path in os.listdir(os.path.join(dataset_path, "Train", letter)):
            img = cv2.imread(os.path.join(dataset_path, "Train", letter, image_path), 0)
            img = cv2.resize(img,(32,32))
            train.append((img, letter))

    # Load Validation Dataset
    for letter in os.listdir(os.path.join(dataset_path, "Validation")):
        print(f"Validation Dataset: Loading Letter {letter}")
        for image_path in os.listdir(os.path.join(dataset_path, "Validation", letter)):
            img = cv2.imread(os.path.join(dataset_path, "Validation", letter, image_path), 0)
            img = cv2.resize(img,(32,32))
            validation.append((img, letter))

    print("Shuffling")
    random.shuffle(train)
    random.shuffle(validation)

    print(train[:5])

def save_model(path):
    tf.keras.models.save_model(model, path, overwrite=True, include_optimizer=True, save_format='tf')

def load_model(path):
    global model
    model = tf.keras.models.load_model(path, custom_objects=None, compile=True)


def train_model():
    init_model()
    import_dataset()
    #train
    pass

def evaluate():
    pass

def main():
    if len(sys.argv) < 2:
        print("Use command line argument")
        exit(0)

    if sys.argv[1].lower() == "train":
        train_model()
    elif sys.argv[1].lower() == "eval":
        evaluate()
    else:
        print("Command not recognized")
        exit(0)

if __name__ == "__main__":
    main()
