# Imports
import os
import pickle

import cv2
import tensorflow as tf
import numpy as np
import json

from skimage import transform as transf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import random
import csv

from keras.models import Sequential
from keras.layers import Dense, Lambda, Activation, Flatten, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

train_samples_per_epoch = 2560
valid_samples_per_epoch = 16384
trainBatchSize = 128
validationBatchSize = 128

xCropUp = .20  # % of crop
xCropBottom = .875  #

imageWidth = 64
imageHeight = 64


def trans_angle(steer):

    """
    translate image and compensate for the translation on the steering angle
    """
    trans_range = 50
    # horizontal translation with 0.008 steering compensation per pixel
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * .2

    if(steer_ang > 1):
        steer_ang = 1
    elif(steer_ang < -1):
        steer_ang = -1

    return steer_ang

def trans_image(dir, steer):
    img = cv2.imread(dir)  # opencv opens images in BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to standard RGB

    """
    translate image and compensate for the translation on the steering angle
    """
    # print(image.shape)
    rows, cols, chan = img.shape
    trans_range = 50

    # horizontal translation with 0.008 steering compensation per pixel
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * .2

    Trans_M = np.float32([[1, 0, tr_x], [0, 1, 0]])
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = CropImage(img)
    img = cv2.resize(img, (imageWidth, imageHeight))
    img = np.reshape(img, (1, imageHeight, imageWidth, 3))
    return img, steer_ang


def CropImage(image):
    height = len(image)
    return image[int(height * xCropUp):int(height * xCropBottom), :, :]


def ReadAndProcessImage(path):
    img = cv2.imread(path)  # opencv opens images in BGR format
    img = CropImage(img)
    img = cv2.resize(img, (imageWidth, imageHeight))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to standard RGB
    img = np.reshape(img, (1, imageHeight, imageWidth, 3))
    return img


def RandomBrightness(img):
    scale = random.uniform(.3, 1)
    img[:, :, :, 2] = img[:, :, :, 2] * scale
    return img

def ValidDataGenerator(dir, batchSize):

    dir = './session_data/'
    with open(dir + 'driving_log.csv', 'r') as drivingLog:
        reader = csv.reader(drivingLog)
        drivingLog = list(reader)
    drivingLog = shuffle(drivingLog)

    negative = 0
    positive = 0

    zero = 0

    while True:
        batchx, batchy = [], []
        drivingLog = shuffle(drivingLog)
        for row in drivingLog:
            indx = np.random.permutation(3)
            steering = float(row[3])

            for i in range(3):

                if indx[i] is 1:  # left camera image
                    steering = min(1, steering + .25)
                elif indx[i] is 2:  # right camera image
                    steering = max(-1, steering - .25)

                file = row[indx[i]]

                img = ReadAndProcessImage(dir + file)

                batchx.append(img)
                batchy.append(steering)
                negative+= steering < 0
                positive+= steering > 0
                zero+= steering == 0
                if len(batchx) >= batchSize:
                    yield (shuffle(np.vstack(batchx),np.vstack(batchy)))
                    batchx, batchy = [], []
        print('\n Valid: negative: ', negative, ' positive: ', positive, ' zero: ', zero)

def DataGenerator(dir, batchSize):
    dir = './data/'
    with open(dir + 'driving_log.csv', 'r') as drivingLog:
        reader = csv.reader(drivingLog)
        drivingLog = list(reader)

    negative = 0
    positive = 0
    zero = 0

    while True:
        batchx, batchy = [], []
        drivingLog = shuffle(drivingLog)
        for row in drivingLog:
            indx = np.random.permutation(3)
            steering = float(row[3])

            for i in range(3):

                if indx[i] is 1:  # left camera image
                    steering = min(1, steering + .25)
                elif indx[i] is 2:  # right camera image
                    steering = max(-1, steering - .25)

                file = row[indx[i]]

                if (steering < -0.9 or steering > 0.1):
                    img = ReadAndProcessImage(dir + file)
                    # Change Brightness
                    img = RandomBrightness(img)
                    batchx.append(img)
                    batchy.append(steering)

                    # Flip image
                    img2 = img.copy()
                    img2[:, ] = cv2.flip(img2[0], 1)
                    batchx.append(img2)
                    batchy.append(-steering)

                    # Translate image.
                    img, steering = trans_image(dir + file, steering)
                    batchx.append(img)
                    batchy.append(steering)

                    # Flip the translated image.
                    img2 = img.copy()
                    img2[:, ] = cv2.flip(img2[0], 1)
                    batchx.append(img2)
                    batchy.append(-steering)

                    positive += 2
                    negative += 2
                else:
                    if zero % 10 == 0 :
                        if steering == 0 :
                            zero+=1
                            img = ReadAndProcessImage(dir + file)
                            batchx.append(img)
                            batchy.append(steering)
                    steeringTemp = trans_angle(steering)
                    if (negative > positive and steeringTemp > 0) or (negative < positive and steeringTemp < 0):
                        img, steering = trans_image(dir + file, steering)
                        batchx.append(img)
                        batchy.append(steering)

                        img[:, ] = cv2.flip(img[0], 1)
                        batchx.append(img)
                        batchy.append(-steering)

                        positive += 1
                        negative += 1
                if len(batchx) >= batchSize:
                    yield (shuffle(np.vstack(batchx),np.vstack(batchy)))
                    batchx, batchy = [], []
                #break

        #print('\n Train negative: ', negative, ' positive: ', positive, ' zero: ', zero/20)


def CreateModel():
    print("Creating Convnet Model")
    input_shape = (imageHeight, imageWidth, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='uniform'))
    model.add(ELU())
    #model.add(Dropout(0.2))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(48, 5, 5, subsample=(1, 1), init='uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, subsample=(1, 1), init='uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, subsample=(1, 1), init='uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))

    print("Creating FC Model")

    model.add(Flatten())

    model.add(Dense(200, init='uniform', W_regularizer=l2(.001)))
    model.add(ELU())
    model.add(Dropout(0.4))

    model.add(Dense(80, init='uniform', W_regularizer=l2(.001)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(50, init='uniform', W_regularizer=l2(.001)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(10, init='uniform', W_regularizer=l2(.001)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(1, init='uniform'))

    #model.load_weights('./model.h5')

    return model


# with open('model.json', 'r') as file:
#    model = model_from_json(json.loads(file.read()))
#    model.load_weights('model.h5')

model = CreateModel()

adam = Adam(lr=0.0001)
model.compile(optimizer="adam", loss="mse")

#validGenerator = ValidDataGenerator('./session_data/',validationBatchSize)
trainGenerator = DataGenerator('./data/', trainBatchSize)

print("Created generator and starting training")
weight_save_callback = ModelCheckpoint('./weights/weights.{epoch:02d}-{loss:.4f}.h5', monitor='loss', verbose=2, save_best_only=False, mode='auto')
model.summary()
model.fit_generator(
    trainGenerator,
    samples_per_epoch=train_samples_per_epoch, nb_epoch=40,
 #   validation_data=validGenerator,
  #  nb_val_samples=valid_samples_per_epoch,
    callbacks=[weight_save_callback],
    verbose=1
)

model.save_weights('model.h5', True)
with open('model.json', 'w') as file:
    json.dump(model.to_json(), file)

