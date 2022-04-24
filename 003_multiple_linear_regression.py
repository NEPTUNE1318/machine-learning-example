import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import gradient_descent_v2

X = np.array([[70,85,11], [71,89,18], [50,80,20], [99,20,10], [50,10,10]]) 
y = np.array([73, 82 ,72, 57, 34])

model = Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))

sgd = gradient_descent_v2.SGD(lr=0.0001)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(X, y, epochs=2000)

print(model.predict(X))

X_test = np.array([[20,99,10], [40,50,20]])
print(model.predict(X_test))    