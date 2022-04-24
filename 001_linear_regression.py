import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.optimizers import gradient_descent_v2

x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [11, 22, 33, 44, 53, 66, 77, 87, 95]

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))
sgd = gradient_descent_v2.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

model.fit(x, y, epochs=300)

plt.plot(x, model.predict(x), 'b', x, y, 'k.')  
plt.show()