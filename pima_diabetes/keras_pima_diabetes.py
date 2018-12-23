import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Dropout


csv = pd.read_csv('diabetes.csv')
print(csv.head())
X = csv.loc[:, 'Pregnancies':'Age']
Y = csv.loc[:,'Outcome']
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

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
model.fit(X_train, y_train, epochs=600)
score = model.evaluate(X_test, y_test)
# print score
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], score[i])