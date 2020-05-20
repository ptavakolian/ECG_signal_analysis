#!/usr/bin/env python

import numpy as np
import os
import sys
from scipy.io import loadmat
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from get_12ECG_features import get_12ECG_features

# input_directory='Desktop/ECG/ECG2/Training_WFDB' 
input_directory = sys.argv[1]

def get_classes1(filename):
        classes = []
        g = filename.replace('.mat', '.hea')
        input_file = os.path.join(g)
        with open(input_file, 'r') as f:
            
            header_data = f.readlines()
            for lines in header_data:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.append(c.strip())
        return sorted(classes), header_data

input_files1=[]
for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files1.append(f)
features_ECG=[]
classes_c = []
y=[]
for i in range(len(input_files1)):
    if i<9:
        filename=input_directory+'/A000{}.mat'.format(i+1)
        xx=loadmat(filename)
        data = np.asarray(xx['val'], dtype=np.float64)
        classes, header_data=get_classes1(filename)
        classes_c.append(classes)
        features_ECG.append(get_12ECG_features(data, header_data))
#         y.append(header_data[15])
        
    if i>=9 and i<99:
        filename=input_directory+'/A00{}.mat'.format(i+1)
        xx=loadmat(filename)
        data = np.asarray(xx['val'], dtype=np.float64)
        classes, header_data=get_classes1(filename)
        classes_c.append(classes)
        features_ECG.append(get_12ECG_features(data, header_data))
#         y.append(header_data[15])
    if i>=99 and i<999:
        filename=input_directory+'/A0{}.mat'.format(i+1)
        xx=loadmat(filename)
        data = np.asarray(xx['val'], dtype=np.float64)
        classes, header_data=get_classes1(filename)
        classes_c.append(classes)
        features_ECG.append(get_12ECG_features(data, header_data))
#         y.append(header_data[15])
    if i>=999:
        filename=input_directory+'/A{}.mat'.format(i+1)
        xx=loadmat(filename)
        data = np.asarray(xx['val'], dtype=np.float64)
        classes, header_data=get_classes1(filename)
        classes_c.append(classes)
        features_ECG.append(get_12ECG_features(data, header_data))
#         y.append(header_data[15])


import pandas as pd
classes=pd.DataFrame(classes_c)
classes_dummy= pd.get_dummies(classes[0]) # First, the first column is converted to dummy variable
for i in range(1,len(classes.iloc[0])):  # for all the other columns in classes 
    classes_d= pd.get_dummies(classes[i])# we convert categorical to dummy variable
    for j in range(len(classes_d.columns)): # Then, we assign the  values of the dummy variables to the one 
#         print(classes_d.columns[j])         #extracted from the first column
        classes_dummy[classes_d.columns[j]]=classes_dummy[classes_d.columns[j]]+classes_d[classes_d.columns[j]]
# len(y_test)
# X_train=features_ECG
# y_train=classes_d[classes_d.columns[0]]
# y_train=classes_c
X_train, X_test, y_train, y_test=train_test_split(features_ECG, classes_dummy, test_size=0.01, random_state=42)

model = RandomForestClassifier(n_estimators=450, random_state=0)
model.fit(X_train, y_train)
# y_pred_rf=model.predict(X_test)
# y_rf_proba=model.predict_log_proba(X_test)
# yyy=np.array(y_rf_proba).reshape(-1,1)
# y_pred_rf=(model.predict(X_test))
filename = 'finalized_model.sav'
joblib.dump(model, filename)

