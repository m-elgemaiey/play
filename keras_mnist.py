from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print (y_train.shape, x_train.shape, x_train[1]) # x is 28 x 28

x_train = x_train.reshape(-1, 784).astype('float32')/255
x_test = x_test.reshape(-1, 784).astype('float32')/255

#print (y_train.shape, x_train.shape, x_train[1])
num_classes = 10

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=60, batch_size=128)

score = model.evaluate(x_test, y_test)

# print score
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], score[i])

model.save('./models/keras_mnist.h5')