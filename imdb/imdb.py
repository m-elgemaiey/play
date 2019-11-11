import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data()

print('Training samples: ', len(x_train))
print('Testing samples: ', len(x_test))

word_to_index = imdb.get_word_index()
print('Word index size: ', len(word_to_index))

word_to_index = {k:v+3 for k,v in word_to_index.items()}
index_to_word = {v:k for k,v in word_to_index.items()}

index_to_word[0] = '<PAD>'
index_to_word[1] = '<START>'
index_to_word[2] = '<OOV>'

'''
print(x_train[:2])
print(y_train[:2])
#print(word_to_index)

sample = ''
for i in x_train[0]:
    sample += index_to_word[i] + ' '

print(sample)
'''

max_length = 256
x_train = pad_sequences(x_train, max_length, padding='post', truncating='post', value=0)
x_test = pad_sequences(x_test, max_length, padding='post', truncating='post', value=0)

model = Sequential(layers=[
    Embedding(len(index_to_word), 128, input_length=max_length),
    LSTM(128, return_sequences=True),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=128, epochs=10)

score = model.evaluate(x_test, y_test)

print('Test score:')
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], score[i])