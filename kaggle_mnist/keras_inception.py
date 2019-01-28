import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np

# input
print('Reading MNIST data from train.csv ...')
csv = pd.read_csv('train.csv', encoding='iso-8859-1')
X = csv.loc[:, 'pixel0':].as_matrix()
y = csv.loc[:, 'label'].as_matrix()
num_classes = 10
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255
y = to_categorical(y, num_classes)

def conv_bn_activation_layer(input, filters, kernel_size, strides=(1,1), padding='same'):
    layer = Conv2D(filters, kernel_size, strides=strides, padding=padding)(input)
    layer = BatchNormalization(scale=False)(layer)
    layer = Activation('relu')(layer)
    return layer

# build the model
input = Input(shape=(28,28,1))
layer = conv_bn_activation_layer(input, 32, (3, 3), strides=(2, 2), padding='valid')
layer = conv_bn_activation_layer(layer, 32, (3, 3), padding='valid')
layer = conv_bn_activation_layer(layer, 64, (3, 3))
layer = MaxPooling2D((3, 3), strides=(2, 2))(layer)
layer = conv_bn_activation_layer(layer, 80, (1, 1), padding='valid')
layer = conv_bn_activation_layer(layer, 192, (3, 3), padding='valid')
#layer = MaxPooling2D((3, 3), strides=(2, 2))(layer)

layer = Flatten()(layer)
layer = Dense(512)(layer)
layer = Dropout(0.2)(layer)
layer = Dense(512, )(layer)
layer = Dropout(0.2)(layer)
layer = Dense(num_classes, activation='softmax')(layer)

model = Model(inputs=input, outputs=layer)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
es_cb = EarlyStopping(monitor='acc', patience=10, restore_best_weights=True, verbose=1)
model.fit(X, y, epochs=100, batch_size=128, callbacks=[lr_cb, es_cb])

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

