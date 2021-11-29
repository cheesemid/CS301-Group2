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
dataset_path = "mntram/dataset"
train = []
validation = []

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
    model.add(Dense(39, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])

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

def save_model(path):
    # tf.keras.models.save_model(model, path, overwrite=True, include_optimizer=True, save_format='h5py')
    model.save(path)

def load_model(path):
    global model
    # model = tf.keras.models.load_model(path, custom_objects=None, compile=True)
    model = tf.keras.models.load_model(path)

def train_model(epochs, div_dataset = 1):
    if model is None:
        init_model()
    if train == []:
        import_dataset()

    train_X = [x[0] for x in train[:int(len(train)/div_dataset)]]
    train_Y = [x[1] for x in train[:int(len(train)/div_dataset)]]

    val_X = [x[0] for x in validation[:int(len(validation)/div_dataset)]]
    val_Y = [x[1] for x in validation[:int(len(validation)/div_dataset)]]

    #LB
    LB = LabelBinarizer()
    train_Y = LB.fit_transform(train_Y)
    val_Y = LB.fit_transform(val_Y)

    train_X = np.array(train_X)/255.0
    train_X = train_X.reshape(-1,32,32,1)
    train_Y = np.array(train_Y)

    val_X = np.array(val_X)/255.0
    val_X = val_X.reshape(-1,32,32,1)
    val_Y = np.array(val_Y)


    history = model.fit(train_X,train_Y, epochs=epochs, batch_size=32, validation_data = (val_X, val_Y),  verbose=1)

def takeSecond(elem):
    return elem[1]

def split_by_letter(img):
    cropped_imgs = []
    ret,thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # for c in contours:
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # plt.imshow(img)

    # print(contours)

    # boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    # key=lambda b:b[1][i], reverse=reverse))

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cropped = img[y-5:y+h+5, x-5:x+w+5]
        if cropped.size <= 5:
            continue
        cropped_imgs.append((cropped, x))

    cropped_imgs.sort(key=takeSecond)

    return [x[0] for x in cropped_imgs]

if __name__ == "__main__":
    train_model(50, 1)
    save_model("models/model_4_50e_all.mdl")