from keras.models import Model
from keras.layers import LSTM, Dense, Flatten, Input, Concatenate
from keras.layers.embeddings import Embedding
from pandas_ml import ConfusionMatrix
import numpy as np
import sms_data

# glove vector side
X_glove_train, X_glove_test, y_train, y_test = sms_data.get_sms_data()
glove_input = Input(shape=sms_data.sms_glove_shape)
lstm_glove = LSTM(32)(glove_input)

# Embedding side
X_embed_train, X_embed_test, y_train, y_test, vocab_size = sms_data.get_sms_embedding_data()
embedding_input = Input(shape=(sms_data.max_sms_length,))
X = Embedding(vocab_size, sms_data.embedding_vector_size)(embedding_input)
lstm_embed = LSTM(32)(X)
X = Concatenate()([lstm_glove, lstm_embed])
X = Dense(32, activation='relu')(X)
y = Dense(1, activation='sigmoid')(X)
# model
model = Model(inputs=[glove_input, embedding_input], outputs=y)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
model.fit([X_glove_train, X_embed_train], y_train, epochs=20)
score = model.evaluate([X_glove_test, X_embed_test], y_test)
# print score
print('Test score:')
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], score[i])

# confusion matrix
y_pred = model.predict([X_glove_test, X_embed_test])
y_pred = np.where(y_pred > 0.5, 1, 0)
y_test = y_test.reshape(y_test.shape[0])
y_pred = y_pred.reshape(y_pred.shape[0])
cm = ConfusionMatrix(y_test, y_pred)

print(cm)
cm.print_stats()      
