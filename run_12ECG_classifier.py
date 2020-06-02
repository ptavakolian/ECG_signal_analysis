#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
from sklearn import model_selection

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)
    print(np.shape( feats_reshape(1,1) ))
    label = model.predict(feats_reshape)
    score = model.predict_proba(feats_reshape)
    score=np.squeeze(score)

    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] = np.array(score[i][1])

    return np.squeeze(label), current_score



def load_12ECG_model(filename):
    # load the model from disk 
    #filename='finalized_model.sav'
    loaded_model = joblib.load(filename)

    return loaded_model
