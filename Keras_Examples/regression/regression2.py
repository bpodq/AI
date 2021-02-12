# -*- coding: UTF-8 -*-
"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

import os
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt

np.random.seed(1337)  # for reproducibility

# create some data   创建散点图数据
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  # randomize the data
w0 = 0.5
b0 = 2
Y = w0 * X + b0 + np.random.normal(0, 0.05, (200,))
# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]  # first 160 data points
X_test, Y_test = X[160:], Y[160:]  # last 40 data points


file = 'regression2'
if os.path.exists(file+'.h5'):
    model = load_model(file+'.h5')
else:
    # build a neural network from the 1st layer to the last layer
    model = Sequential()

    model.add(Dense(units=1, input_dim=1))

    # choose loss function and optimizing method
    model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(1001):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        (W, b) = model.layers[0].get_weights()
        print('train cost: ', cost)
        print('Weights=', W, '\nbiases=', b)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()

model.save(file+'.h5', overwrite=True)  # 保存模型
model.save(file+'_'+time.strftime("%Y%m%d-%H%M%S", time.localtime())+'.h5')  # 再保存一遍，加上时间
