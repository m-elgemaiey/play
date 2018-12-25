import numpy as np
import spacy
import os.path as path
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


max_sms_length = 30 # max number of words in sms
glove_word_vec_size = 300 # spacy word vector size
sms_glove_shape = (max_sms_length, glove_word_vec_size)
embedding_vector_size = 25 # Embedding vector size

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
    X = np.array([get_sentence_vectors(nlp, sent, max_sms_length) for sent in sentences])
    return X

# read data from the csv file
def get_sms_text_from_csv():
    print('Read sms data from spam.csv ...')
    csv = pd.read_csv('spam.csv', encoding='iso-8859-1')
    messages = csv.loc[:, 'v2']
    y = csv.loc[:, 'v1'].as_matrix()
    y = np.where(y == 'spam', 1, 0)

    return messages, y
    
# get training and test data
def get_sms_data():
    data_file_name = 'preprocessed.npz'
    if path.isfile(data_file_name):
        npzfile = np.load(data_file_name)
        return npzfile['xtrain'], npzfile['xtest'], npzfile['ytrain'], npzfile['ytest']
    else:
        messages, y = get_sms_text_from_csv()
        X = convert_sentences_to_vectors(messages)
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        print('Save training data to preprocessed.npz ...')
        np.savez(data_file_name, xtrain=X_train, xtest=X_test, ytrain=y_train, ytest=y_test)
        return X_train, X_test, y_train, y_test

def get_sms_embedding_data():
    messages, y = get_sms_text_from_csv()
    X_train, X_test, y_train, y_test = train_test_split(messages, y, test_size=0.2, shuffle=False)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    # integer encode the documents
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    # pad documents to a max length of 4 words
    X_train = pad_sequences(X_train, maxlen=max_sms_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_sms_length, padding='post')

    vocab_size = len(tokenizer.word_index) + 1

    return X_train, X_test, y_train, y_test, vocab_size
