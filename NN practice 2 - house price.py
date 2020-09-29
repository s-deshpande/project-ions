# This code was a practise nueral network code written to find whether the price of a house was greater than the median price.
# As this is a begineers code, it involves no data engineering.

import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt

db = pd.read_csv('housepricedata.csv')

predict = 'AboveMedianPrice'
x = np.array(db.drop([predict],1))
y = np.array(db[predict])
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(x)
x_train,x_test_and_val,y_train,y_test_and_val= train_test_split(X,y,test_size=0.3)
x_val,x_test,y_val,y_test = train_test_split(x_test_and_val,y_test_and_val,test_size=0.5)
print(X)

def Nueral_network(x_train,x_test,y_train,y_test):
    model = Sequential()
    model.add(Dense(10, input_shape=(10,),activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
    hist = model.fit(x_train,y_train,epochs=10, validation_data=(x_val,y_val))
    loss, accuracy = model.evaluate(x_test,y_test)
    return accuracy,hist

def Linear_Regression(x_train,x_test,y_train,y_test):
    model = linear_model.LinearRegression()
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)
    return acc

accuracy, hist = Nueral_network(x_train,x_test,y_train,y_test)
print('Nueral Network:' ,accuracy *100)
acc = Linear_Regression(x_train,x_test,y_train,y_test)
print('Linear regression:',acc*100)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()