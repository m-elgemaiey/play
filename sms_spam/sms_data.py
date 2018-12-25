import numpy as np
import spacy
import os.path as path
import pandas as pd
from sklearn.model_selection import train_test_split

max_sms_length = 30 # max number of words in sms
word_vec_size = 300 # spacy word vector size
sms_shape = (max_sms_length, word_vec_size)

# get sentence word vectors 
def get_sentence_vectors(nlp, sentence, max_size):
    sentence_vec = []
    doc = nlp(sentence)
    for token in doc:
        sentence_vec.append(token.vector)
        max_size = max_size - 1
        if max_size == 0:
            break
    if max_size > 0:
        zeros = np.zeros((300))
        while max_size > 0:
            sentence_vec.append(zeros)
            max_size = max_size - 1

    return np.array(sentence_vec)

# return a matrix the represents all sentences (in word vectors)
def convert_sentences_to_vectors(sentences):
    print("Loading spacy model...")
    nlp = spacy.load('en_core_web_md')
    print("Loaded spacy model")
    X = [get_sentence_vectors(nlp, sent, max_sms_length) for sent in sentences]
    return X

# get training and test data
def get_sms_data():
    data_file_name = 'preprocessed.npz'
    if path.isfile(data_file_name):
        npzfile = np.load(data_file_name)
        return npzfile['xtrain'], npzfile['xtest'], npzfile['ytrain'], npzfile['ytest']
    else:
        # read data
        print('Read sms data from spam.csv ...')
        csv = pd.read_csv('spam.csv', encoding='iso-8859-1')
        X = csv.loc[:, 'v2']
        X = convert_sentences_to_vectors(X)
        y = csv.loc[:, 'v1'].as_matrix()
        y = np.where(y == 'spam', 1, 0)
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print('Save training data to preprocessed.npz ...')
        np.savez(data_file_name, xtrain=X_train, xtest=X_test, ytrain=y_train, ytest=y_test)
        return X_train, X_test, y_train, y_test
