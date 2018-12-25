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

# get data
X_train, X_test, y_train, y_test, vocab_size = sms_data.get_sms_embedding_data()
# define the model
model = Sequential()
model.add(Embedding(vocab_size, sms_data.embedding_vector_size, input_length=sms_data.max_sms_length))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(X_train, y_train, epochs=20)
# evaluate
score = model.evaluate(X_test, y_test)
# print score
print('Test score:')
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], score[i])

# confusion matrix
y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
y_test = y_test.reshape(y_test.shape[0])
y_pred = y_pred.reshape(y_pred.shape[0])
cm = ConfusionMatrix(y_test, y_pred)

print(cm)
cm.print_stats()      
