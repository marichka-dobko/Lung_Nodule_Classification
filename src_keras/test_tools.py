from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
import operator
from tqdm import tqdm
import keras
import os
import pandas as pd
import numpy as np


def load_trained_model(path_model_json, path_model_h5):
    json_file = open(path_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_model_h5)
    
    return loaded_model

def evaluate_full_candidates_list(loaded_model, candidates=None, dim3=False, model_name=None ,save_file=False):
    if candidates is None:
        candidates = pd.read_csv('/datagrid/temporary/dobkomar/output_path_32/candidates.csv')
    predicted, probas = [], []
    
    for row in tqdm(candidates.iterrows()):
        image = row[1]
        y_class = int(image['class'])     
        X_pred = []

        lung_img = np.load(image['name'])

        if dim3:
            X = lung_img.reshape((32, 32, 32))
            X = (X-mean_x)/std_x
            X_pred.append(X)
        else:
            for ind in range(5):
                X = lung_img[14 + ind, :, :].reshape((32, 32))
                X = (X-mean_x)/std_x
                X_pred.append(X)

        predicted_temp, probas_temp = [], []    
        for x in X_pred:
            if d3:
                pred = np.argmax(loaded_model.predict(x.reshape((1, 32, 32, 32, 1))))
                prob = float(loaded_model.predict_proba(x.reshape((1, 32, 32, 32, 1)))[0][1])
                predicted.append(pred), probas.append(prob)

            else:
                pred = np.argmax(loaded_model.predict(x.reshape((1, 32, 32, 1))))
                prob = float(loaded_model.predict_proba(x.reshape((1, 32, 32, 1)))[0][1])
                predicted_temp.append(pred), probas_temp.append(prob)

        if not d3:
            predicted.append(np.median(predicted_temp)), probas.append(np.median(probas_temp))    

    candidates['predicted'] = predicted
    candidates['probability'] = probas  
    
    if save_file:
        data[['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']].to_csv('/home.stud/dobkomar/NoduleDetection/{}.csv'.format(model_name), index=False)
    return candidates