# ECG_signal_analysis

The goal here is to predict cardiac abnormalities from 12-Lead ECG signal recorded from 6800 patients. First, we extracted features from ECG signals of each subject. Number of cardiac abnormalities (output classes) here is 9. Then, we used random forest and trained the model on the dataset. Recorded the model with joblib command to be used for the challenge testing data.


""" First run the randomforest.py/rf2.py/neigh.py which generates the model as finalized_model.sav.
The input directory for random forest is Training_WFDB which can be downloaded here:
https://physionetchallenges.github.io/2020/

First link

To run it on Terminal write: python3 randomforest.py Training_WFDB 
This program calls get_12ECG_features.py.
Second, run the driver.py which generates the scores and probability for each cardiac disease.
The input directory is again Training_WFDB, and the output name should be specified as well
Second link:

https://physionetchallenges.github.io/2020/

To run it on Terminal write: python3 driver.py Training_WFDB output_directory.
This program calls get_12ECG_features.py and run_12ECG_classifier."""
