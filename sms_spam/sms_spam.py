from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from pandas_ml import ConfusionMatrix
import numpy as np
import sms_data

# read data
X_train, X_test, y_train, y_test = sms_data.get_sms_data()

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=sms_data.sms_shape))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15)
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


