import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from skimage.feature import hog
from sklearn.utils import shuffle

from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import ELU
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Lambda
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.models import model_from_json
import simplejson as json
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import random
import math

### Make data frame in Pandas

import pandas as pd

#Reading the first data set from crowd-ai
rootDir = "object-detection-crowdai/"
csvFile = pd.read_csv(rootDir+'labels.csv', header=0)

dataFile = csvFile[(csvFile['Label']!='Pedestrian')].reset_index()
dataFile = dataFile.drop('index', 1)
dataFile = dataFile.drop('Preview URL', 1)
dataFile['Frame'] = './' + rootDir + dataFile['Frame']
dataFile.columns = ['xmin', 'ymin', 'xmax','ymax', 'Frame', 'Label']
dataFile.head(10)
print('first data set len: ',len(dataFile))

#Reading the second data set.
names = ['Frame',  'xmin', 'ymin', 'xmax','ymax', 'occluded', 'Label']
rootDir = "object-dataset/"
csvFile1 = pd.read_csv(rootDir+'labels.csv', delim_whitespace=True, names=names)
dataFile1 = csvFile1[(csvFile1['Label']!=str.lower('Pedestrian'))].reset_index()
dataFile1 = dataFile1.drop('index',1)
dataFile1 = dataFile1.drop('occluded',1)
dataFile1['Frame'] = './' + rootDir + dataFile1['Frame']
dataFile1.tail(10)
print('second dataset len: ',len(dataFile1))


train_samples_per_epoch = 10000
trainBatchSize = 16
imgRow = 630
imgCol = 960
croppedImgRow = math.floor((imgRow*6)/7)-math.floor((imgRow*2)/5)
croppedImgCol = 960
smooth = 1.

def CropImage(image):
    #Crops the iamge so that the hood of the car and top of the image
    #which contains sky trees and other stuffs are removed.
    height = len(image)
    return image[int(height * 2/5.):int(height * 6/7.0), :, :]

def RandomBrightness(img):
    scale = random.uniform(.7, 1.25)
    img[:, :, :, 2] = img[:, :, :, 2] * scale
    return img

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def TrainDataGenerator(dataInfoList,batchSize):
    batch_x, batch_y = [], []
    while True:
        dataInfo = dataInfoList[random.randint(0, 1)]
        row = np.random.randint(len(dataInfo))

        fileName = dataInfo['Frame'][row]
        img = cv2.imread(fileName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origShape = img.shape
        img = cv2.resize(img, (imgCol, imgRow))

        data = dataInfo[dataInfo['Frame'][row] == dataInfo['Frame']].reset_index()
        data['xmin'] = np.round(data['xmin'] / origShape[1] * imgCol)
        data['xmax'] = np.round(data['xmax'] / origShape[1] * imgCol)
        data['ymin'] = np.round(data['ymin'] / origShape[0] * imgRow)
        data['ymax'] = np.round(data['ymax'] / origShape[0] * imgRow)

        targetImg = np.reshape(np.zeros_like(img[:, :, 2]), (imgRow, imgCol, 1))
        for i in range(len(data)):
            targetImg[data.iloc[i]['ymin']:data.iloc[i]['ymax'], data.iloc[i]['xmin']:data.iloc[i]['xmax']] = 1

        img = CropImage(img)
        targetImg = CropImage(targetImg)
        batch_x.append(img)
        batch_y.append(targetImg)

        if len(batch_x) == batchSize:
            x_array = np.asarray(batch_x)
            y_array = np.asarray(batch_y)
            yield (x_array, y_array)
            batch_x, batch_y = [], []



def CreateModel():
    input_layer = Input((croppedImgRow, croppedImgCol, 3))
    conv0 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(input_layer)
    conv0 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool0)
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv4)

    up5 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=3)
    conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up5)
    conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv2], mode='concat', concat_axis=3)
    conv6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv1], mode='concat', concat_axis=3)
    conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv0], mode='concat', concat_axis=3)
    conv8 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv8)

    conv9 = Convolution2D(1, 1, 1, activation='sigmoid')(conv8)

    model = Model(input=input_layer, output=conv9)

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    return model


model = CreateModel()

trainGenerator = TrainDataGenerator((dataFile,dataFile1), trainBatchSize)

weight_save_callback = ModelCheckpoint('./weights/weights.{epoch:02d}-{loss:.4f}.h5', monitor='loss', verbose=2,
                                       save_best_only=False, mode='auto')
model.summary()

#model.load_weights('model.h5')

print("Created generator and call backs. Starting training")

model.fit_generator(
    trainGenerator,
    samples_per_epoch=train_samples_per_epoch, nb_epoch=40,
    # validation_data=validGenerator,
    # nb_val_samples=valid_samples_per_epoch,
    callbacks=[weight_save_callback],
    verbose=1
)

model.save_weights('model.h5', True)
with open('model.json', 'w') as file:
    json.dump(model.to_json(), file)