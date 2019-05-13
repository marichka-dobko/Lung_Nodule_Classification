import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D, Convolution2D, AveragePooling2D, MaxPooling2D, ZeroPadding3D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
import cv2
import operator


def CNNT4():
    model = keras.Sequential()
    model.add(Convolution2D(filters=6, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(32,32,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(filters=16, kernel_size=(5, 5),strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(units=150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=2, activation = 'softmax'))
    return model


def CNNT5():
    model = keras.Sequential()
    model.add(Convolution2D(filters=8, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(32,32,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(filters=16, kernel_size=(5, 5),strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(units=150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=2, activation = 'softmax'))
    return model


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def VGG_11(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(32,32,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model


def LeNet3d():
    model = Sequential()
    model.add(Convolution3D(6, kernel_size=(5, 5, 5), activation='relu', input_shape=(32,32,32,1)))
    model.add(MaxPooling3D(strides=2)) 
    model.add(Convolution3D(16, kernel_size=(5, 5, 5), activation='relu'))
    model.add(MaxPooling3D(strides=2)) 

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model


def CNNT5_3D():
    model = keras.Sequential()
    model.add(Convolution3D(filters=8, kernel_size=(5, 5, 5), strides=1, activation='relu', input_shape=(32,32,32,1)))
    model.add(MaxPooling3D(pool_size=(2, 2,2), strides=2))
    model.add(Convolution3D(filters=16, kernel_size=(5, 5, 5),strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(units=150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=2, activation = 'softmax'))
    
    return model


def VGG11_3D(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding3D((0,1,1),input_shape=(32,32,32, 1)))
    model.add(Convolution3D(64, 3, 3,3, activation='relu',dim_ordering='tf'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))

    model.add(ZeroPadding3D((0,1,1)))
    model.add(Convolution3D(128, 3, 3,3, activation='relu',dim_ordering='tf'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2),dim_ordering='tf'))

    model.add(ZeroPadding3D((0,1,1)))
    model.add(Convolution3D(256, 3, 3,3, activation='relu',dim_ordering='tf'))
    model.add(ZeroPadding3D((0,1,1)))
    model.add(Convolution3D(256, 3,3, 3, activation='relu',dim_ordering='tf'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2),dim_ordering='tf'))

    model.add(ZeroPadding3D((0,1,1)))
    model.add(Convolution3D(512, 3,3, 3, activation='relu',dim_ordering='tf'))
    model.add(ZeroPadding3D((0,1,1)))
    model.add(Convolution3D(512,3, 3, 3, activation='relu',dim_ordering='tf'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2),dim_ordering='tf'))

    model.add(ZeroPadding3D((0,1,1)))
    model.add(Convolution3D(512, 3,3, 3, activation='relu',dim_ordering='tf'))
    model.add(ZeroPadding3D((0,1,1)))
    model.add(Convolution3D(512, 3,3, 3, activation='relu',dim_ordering='tf'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2),dim_ordering='tf'))

    model.add(Flatten())
    xout=model.output_shape
    xin=model.input_shape
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model