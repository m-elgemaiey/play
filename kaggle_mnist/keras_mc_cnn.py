import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# load training data
print('Reading MNIST data from train.csv ...')
csv = pd.read_csv('train.csv', encoding='iso-8859-1')
X = csv.loc[:, 'pixel0':].values
y = csv.loc[:, 'label'].values
num_classes = 10
X_train = X.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y, num_classes)

# load test data
print('Reading MNIST data from test.csv ...')
csv = pd.read_csv('test.csv', encoding='iso-8859-1')
X = csv.loc[:, 'pixel0':].values
X_test = X.reshape(-1, 28, 28, 1).astype('float32') / 255

# build the model
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=[5, 5], padding="same", input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=[5, 5], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=256, kernel_size=[5, 5], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=512, kernel_size=[3, 3], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

datagen = ImageDataGenerator(
    rotation_range=10,  
    width_shift_range=0.1, 
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1) 

for i in range(5):
    print("==========================")
    print("Training model #", i)
    print("==========================")
    lr_cb = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=5, verbose=1)
    es_cb = EarlyStopping(monitor='acc', patience=10, restore_best_weights=True, verbose=1)
    model = build_model()
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=128), epochs=100, steps_per_epoch=X_train.shape[0] // 128, callbacks=[lr_cb, es_cb])
    y = model.predict(X_test)
    if i == 0:
        y_test = y
    else:
        y_test += y

print("Saving Results...")
y = np.argmax(y_test, 1)
img_id = np.arange(1, y.shape[0] + 1)

data_frame = pd.DataFrame(data = {'ImageId': img_id, 'Label': y})
print(data_frame.head())
data_frame.to_csv('results.csv', index=False)