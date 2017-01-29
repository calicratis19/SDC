#Imports
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
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.models import model_from_json


train_samples_per_epoch = 40064
valid_samples_per_epoch = 16384
trainBatchSize = 256
validationBatchSize = 256
mean = 83.587824273 #Pre Calculated


xCropUp = .35      # % of crop
xCropBottom = .875   #

imageWidth = 200
imageHeight = 66


def CropImage(image):
    height = len(image)
    return image[int(height*xCropUp):int(height*xCropBottom), :, :]



def ReadAndProcessImage(path):
    img = cv2.imread(path)  # opencv opens images in BGR format
    img = CropImage(img)
    img = cv2.resize(img, (imageWidth, imageHeight))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to standard RGB
    img = np.reshape(img, (1, imageHeight, imageWidth, 3))
    return img

def DataGenerator(dir,batchSize):
    #dir = './session_data/'
    with open(dir+'driving_log.csv', 'r') as drivingLog:
        reader = csv.reader(drivingLog)
        drivingLog = list(reader)
        drivingLog = shuffle(drivingLog)
    while True:
        batchx , batchy= [], []
        for row in drivingLog:
            indx = np.random.permutation(3)
            steering = float(row[3])

            for i in range(3):

                if indx[i] is 1: # left camera image
                    steering += .25
                elif indx[i] is 2: # right camera image
                    steering -= .25

                file = row[indx[i]]
                img = ReadAndProcessImage(dir + file)

                if (indx[i] != 0 or steering != 0):
                    batchx.append(img)
                    batchy.append(steering)

                # Decrease Brightness
                img1 = img.copy()
                scale = random.uniform(.1, .8)
                img1[:, :, :,2] = img1[:,:,:, 2] * scale
                batchx.append(img1)
                batchy.append(steering)

                #Flip image
                img2 = img.copy()
                img2[:,] = cv2.flip(img2[0],1)
                batchx.append(img2)
                batchy.append(-steering)
                if len(batchx) >= batchSize:
                    yield (np.vstack(batchx),np.vstack(batchy))
                    batchx, batchy= [],[]


def CreateModel():

    print("Creating Convnet Model")
    input_shape=(imageHeight, imageWidth, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal'))
    model.add(ELU())

    print("Creating FC Model")

    model.add(Flatten())

    model.add(Dense(1164, init='he_normal'))
    model.add(ELU())

    model.add(Dense(100, init='he_normal'))
    model.add(ELU())

    model.add(Dense(50, init='he_normal'))
    model.add(ELU())

    model.add(Dense(10, init='he_normal'))
    model.add(ELU())

    model.add(Dense(1, init='he_normal'))
    return model



#with open('model.json', 'r') as file:
#    model = model_from_json(json.loads(file.read()))
#    model.load_weights('model.h5')

model = CreateModel()

model.compile(optimizer="adam", loss="mse")

#mean =  MeanCalculator() # calculated once
print('mean: ',mean)
totalTrain = 0
totalValid = 0

trainGenerator = DataGenerator('./session_data/',trainBatchSize)
validGenerator = DataGenerator('./data/',validationBatchSize)

print("Created generator and starting training")

model.fit_generator(
	trainGenerator,
	samples_per_epoch=train_samples_per_epoch, nb_epoch=10,
	validation_data=validGenerator,
	nb_val_samples=valid_samples_per_epoch, verbose=1
)


model.save_weights('model.h5', True)
with open('model.json', 'w') as file:
	json.dump(model.to_json(), file)


