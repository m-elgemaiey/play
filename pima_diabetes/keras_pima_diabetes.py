import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Dropout
from pandas_ml import ConfusionMatrix


csv = pd.read_csv('diabetes.csv')
print(csv.head())
X = csv.loc[:, 'Pregnancies':'Age']
y = csv.loc[:,'Outcome'].as_matrix()
y = y.reshape(y.shape[0], 1)
#y = csv.values(['Outcome'])
#print(y)
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(8,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
score = model.evaluate(X_test, y_test)
# print score
print('Test score:')
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], score[i])

y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
y_test = y_test.reshape(y_test.shape[0])
y_pred = y_pred.reshape(y_pred.shape[0])
cm = ConfusionMatrix(y_test, y_pred)

print(cm)
cm.print_stats()