from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.layers.embeddings import Embedding
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas_ml import ConfusionMatrix
import numpy as np
import sms_data

messages, y = sms_data.get_sms_text_from_csv()
X_train, X_test, y_train, y_test = train_test_split(messages, y, test_size=0.2, shuffle=False)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
# integer encode the documents
encoded_docs = tokenizer.texts_to_sequences(X_train)
# pad documents to a max length of 4 words
padded_docs = pad_sequences(encoded_docs, maxlen=sms_data.max_sms_length, padding='post')

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 25, input_length=sms_data.max_sms_length))
model.add(LSTM(32, return_sequences=True))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, y_train, epochs=25)
encoded_docs = tokenizer.texts_to_sequences(X_test)
padded_docs = pad_sequences(encoded_docs, maxlen=sms_data.max_sms_length, padding='post')
# evaluate
score = model.evaluate(padded_docs, y_test)
# print score
print('Test score:')
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], score[i])

# confusion matrix
y_pred = model.predict(padded_docs)
y_pred = np.where(y_pred > 0.5, 1, 0)
y_test = y_test.reshape(y_test.shape[0])
y_pred = y_pred.reshape(y_pred.shape[0])
cm = ConfusionMatrix(y_test, y_pred)

print(cm)
cm.print_stats()      
