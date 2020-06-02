#!/usr/bin/env python

import numpy as np
import os
import sys
from scipy.io import loadmat
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from get_12ECG_features import get_12ECG_features
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# input_directory='Desktop/ECG/ECG2/Training_WFDB' 
input_directory = sys.argv[1]

def get_classes_and_header(filename):
        classes = []
        g = filename.replace('.mat', '.hea')
        input_file = os.path.join(g)
        with open(input_file, 'r') as f:
            
            header_data = f.readlines()
            for lines in header_data:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
#                     print(tmp)
                    for c in tmp:
                        classes.append(c.strip())
        return sorted(classes), header_data

features_ECG=[]
classes_c = []
for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            filename=input_directory+'/'+f
            mat_file= loadmat(filename)
            ECGdata = np.asarray(mat_file['val'], dtype=np.float64)
            classes, header_data=get_classes_and_header(filename)
            classes_c.append(classes)
            features_ECG.append(get_12ECG_features(ECGdata, header_data))
# np.size(input_files1)
import pandas as pd
classes=pd.DataFrame(classes_c)
classes_dummy= pd.get_dummies(classes[0]) # First, the first column is converted to dummy variable
for i in range(1,len(classes.iloc[0])):  # for all the other columns in classes 
    classes_d= pd.get_dummies(classes[i])# we convert categorical to dummy variable
    for j in range(len(classes_d.columns)): # Then, we assign the  values of the dummy variables to the one 
#         print(classes_d.columns[j])         #extracted from the first column
        classes_dummy[classes_d.columns[j]]=classes_dummy[classes_d.columns[j]]+classes_d[classes_d.columns[j]]

X_train, X_test, y_train, y_test=train_test_split(features_ECG, classes_dummy, test_size=0.01, random_state=42)

model = RandomForestClassifier(n_estimators=650, random_state=0)
model.fit(X_train, y_train)
filename = 'finalized_model2.sav'
joblib.dump(model, filename)

