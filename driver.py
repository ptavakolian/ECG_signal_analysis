import numpy as np
import os
import sys
import pandas as pd
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier


def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


def save_challenge_predictions(output_directory, filename, scores, labels, classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat', '.csv')
    output_file = os.path.join(output_directory, new_file)

    labels=np.asarray(labels,dtype=np.int)
    scores=np.asarray(scores,dtype=np.float64)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')


#rbd_Debug_begin
def get_prob_distr(output_directory, files):
    distr= set()
    entry = []
    df1 = pd.DataFrame()
    for f in files:
        input_file = os.path.join(output_directory, f)
        patient = f
        with open(input_file, 'r') as f:
             tmp = f.readlines()[3:]
             for c in tmp:
                 distr.add(c.strip())
                 tmp2 = c.split(',')
                 for d in tmp2:
                     entry.append(d.rstrip("\n"))
                 df2 = pd.DataFrame({patient : entry}, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
                 df1.reset_index(drop=True, inplace=True)
                 df2.reset_index(drop=True, inplace=True)
                 df1 = pd.concat([df1, df2], axis=1, sort=False)
                 entry.clear()

    return (df1)

# Find unique number of classes
def get_classes(input_directory, files):

    classes = set()
    for f in files:
        g = f.replace('.mat', '.hea')
        input_file = os.path.join(input_directory, g)
        with open(input_file, 'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())

    return sorted(classes)


if __name__ == '__main__':
    # Parse arguments.

    #rbd_Debug
    #if len(sys.argv) != 3:
    #    raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')
    #rbd_Debug

    #rbd_Debug
    #input_directory = sys.argv[1]
    #output_directory = sys.argv[2]
    #rbd_Debug
    input_directory = "../Training_WFDB"
    output_directory = "../output_data_exp_test4"
    output_directory2 = "../output_data_rf2"
    model2 = "finalized_model2.sav"



    # Find files.
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    classes = get_classes(input_directory, input_files)

    """"
    #rbd_Debug_begin : We are debugging and want to skip the creation of class outputs
    So that we can test our snippet of code that reads in CSV files

    # Load model.
    print('Loading 12ECG model...')
    model = load_12ECG_model()

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)

    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        current_label, current_score = run_12ECG_classifier(data, header_data, classes, model)
        # Save results.
        
        save_challenge_predictions(output_directory, f, current_score, current_label, classes)
        
    #rbd_Debug_end
    """
    print('Done.')
    #rbd_debug
    # Parse Distributions from one classifier
    # Find files.
    parse_files = []
    for f in os.listdir(output_directory):
        if os.path.isfile(os.path.join(output_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('csv'):
            parse_files.append(f)

    print('Debug: Start reading CSV files of classifier')
    df1 = get_prob_distr(output_directory, parse_files)
    print('Debug: Done Reading CSV files of classifier')
    df1
    print('Done with forming DataFrame for Random Forest')

    # Load model.
    print('Loading 12ECG model for RF 2...')
    model = load_12ECG_model(model2)

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)

    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i + 1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        current_label, current_score = run_12ECG_classifier(data, header_data, classes, model2)
        # Save results.

        save_challenge_predictions(output_directory2, f, current_score, current_label, classes)

    print('Done Random Forest 2.')
    # dummy comment..
    dummy_debug = 0