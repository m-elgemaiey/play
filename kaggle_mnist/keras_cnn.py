import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D,Flatten, BatchNormalization
from keras.utils import to_categorical
import numpy as np
from callback import EarlyStoppingByAccuracy, EarlyStoppingByLoss

# input
print('Reading MNIST data from train.csv ...')
csv = pd.read_csv('train.csv', encoding='iso-8859-1')
X = csv.loc[:, 'pixel0':].as_matrix()
y = csv.loc[:, 'label'].as_matrix()
num_classes = 10
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255
y = to_categorical(y, num_classes)

# build the model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=60, batch_size=128, callbacks=[EarlyStoppingByAccuracy(0.99992)])
#model.fit(X, y, epochs=160, batch_size=128, callbacks=[EarlyStoppingByLoss(1.0e-7)])

# load test data
print('Reading MNIST data from test.csv ...')
csv = pd.read_csv('test.csv', encoding='iso-8859-1')
X = csv.loc[:, 'pixel0':].as_matrix()
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255
y = model.predict(X)
y = np.argmax(y, 1)
img_id = np.arange(1, y.shape[0] + 1)

data_frame = pd.DataFrame(data = {'ImageId': img_id, 'Label': y})
print(data_frame.head())
data_frame.to_csv('results.csv', index=False)