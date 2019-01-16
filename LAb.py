import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from keras.layers import Dropout

#set seed for reproduction purpose
from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

import random as rn
rn.seed(12345)

import tensorflow as tf
tf.set_random_seed(1234)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_train = StandardScaler().fit_transform(x_train)

x_test = StandardScaler().fit_transform(x_test)

data_train, data_test, target_train, target_test =x_train,x_test,y_train,y_test

data_train = np.array(data_train)
data_test = np.array(data_test)
target_train = np.array(target_train)
target_test = np.array(target_test)

neural_network_d = Sequential()
neural_network_d.add(Dense(10, activation='relu', input_shape=(13,)))
neural_network_d.add(Dropout(0.1))
neural_network_d.add(Dense(5, activation='relu'))
neural_network_d.add(Dropout(0.1))
neural_network_d.add(Dense(1, activation='relu'))

neural_network_d.summary()

neural_network_d2 = Sequential()
neural_network_d2.add(Dense(10, activation='relu', input_shape=(13,)))
neural_network_d2.add(Dense(5, activation='relu'))
neural_network_d2.add(Dense(1, activation='relu'))

neural_network_d2.summary()

neural_network_d.compile(SGD(lr = .003), "mean_squared_error",
                   )

neural_network_d2.compile(SGD(lr = .003), "mean_squared_error",
                   )
np.random.seed(0)
run_hist_1 = neural_network_d.fit(data_train, target_train, epochs=500,\
                              validation_data=(data_test, target_test), \
                              verbose=False, shuffle=False)
run_hist_2 = neural_network_d2.fit(data_train, target_train, epochs=500,\
                              validation_data=(data_test, target_test), \
                              verbose=False, shuffle=False)

print("Training neural network with dropouts..\n")

print("Model evaluation Train data [loss]: ", neural_network_d.evaluate(data_train, target_train))
print("Model evaluation  Test Data [loss]: ", neural_network_d.evaluate(data_test, target_test))

print("Training neural network without dropouts..\n")
print("Model evaluation Train data [loss]: ", neural_network_d2.evaluate(data_train, target_train))
print("Model evaluation  Test Data [loss]: ", neural_network_d2.evaluate(data_test, target_test))
plt.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
plt.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
plt.title("Train loss and validation error with dropouts")
plt.legend()
plt.grid()
from keras.models import load_model
model = load_model('mnist-model.h5')
new_data=np.array(data_test[1])
new_data=new_data.reshape(1,-1)
print(new_data.shape)
new_target=np.array(target_test[1])
new_target=new_target.reshape(1,-1)
print(new_target.shape)
print(new_target)
print(neural_network_d.predict(new_data))
print("Model evaluation [loss]: ", neural_network_d.evaluate(new_data, new_target))
