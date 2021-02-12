import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


trX = np.linspace(-1, 1, 101)
trY = 3 * trX + np.random.randn(*trX.shape) * 0.33

model = Sequential()
# model.add(Dense(input_dim=1, output_dim=1, init='uniform', activation='linear'))
model.add(Dense(input_dim=1, units=1, kernel_initializer='uniform', activation='linear'))

weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init))
## Linear regression model is initialized with weight w: -0.03, b: 0.00


model.compile(optimizer='sgd', loss='mse')


model.fit(trX, trY, epochs=200, verbose=1)


weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))
##Linear regression model is trained to have weight w: 2.94, b: 0.08


