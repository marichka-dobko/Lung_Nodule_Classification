from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D, Convolution2D, AveragePooling2D, MaxPooling2D, ZeroPadding3D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import operator
from tqdm import tqdm_notebook as tqdm
import keras
import os
from models_keras import CNNT4, CNNT5, VGG_11



def train_model(model, opt=None):
    if opt is None:
        opt = keras.optimizers.rmsprop(lr=0.0001, rho=0.95)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    callback = [EarlyStopping(monitor='val_loss', patience=7),
                ReduceLROnPlateau(patience=5, verbose=1)]

    history = model.fit(x=X_train_range, y=Y_train, epochs=30, validation_data=(X_test_range, Y_test),
              batch_size=128, callbacks=callback) 
    
    return model, history


def save_history_training(history, path_to_save_csv):
    df_logs = pd.DataFrame(columns=['val_acc', 'val_loss', 'train_loss', 'train_acc'])
    df_logs['val_acc'] =history.history['val_acc']
    df_logs['val_loss'] = history.history['val_loss']
    df_logs['train_acc'] = history.history['acc']
    df_logs['train_loss'] = history.history['loss']
    
    df_logs.to_csv(path_to_save_csv, index=False)
    
    return "Saved training history"


def save_model(model, path_json, path_h5):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5 
    model.save_weights(path_h5)
    
    return "Saved model weights and architecture"
