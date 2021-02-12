import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 不使用GPU

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers.core import Dropout

import time
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
# import icecream.ic as ic

def generate_sample(n):
    X1 = np.random.randn(n, 2)
    Y1 = np.zeros(n)
    X2 = np.random.randn(n, 2) + [10, 5]
    Y2 = np.ones(n)
    X = np.vstack((X1, X2))
    Y = np.hstack((Y1, Y2))

    return X, Y


start = time.time()

X_train = pd.read_csv('X_train.csv').values
Y_train = pd.read_csv('Y_train.csv').values
X_test = pd.read_csv('X_test.csv').values
Y_test = pd.read_csv('Y_test.csv').values

print(X_train.shape)
print(Y_train.shape)

# ic(X_train.max())
print('max: ', X_train.max())
X_train /= X_train.max()
Y_train = to_categorical(Y_train, 2)
X_test /= X_test.max()
Y_test = to_categorical(Y_test, 2)

fig = plt.figure(1)
plt.plot(X_train[:, 0], X_train[:, 1], '.')
# plt.plot(2[:, 0], X2[:, 1], '.')
plt.draw()
plt.pause(1)
plt.close(fig)


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
# K.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)

file = 'playground'
if os.path.exists(file+'.h5'):
    model = load_model(file+'.h5')
    # os.rename(file+)
else:
    model = Sequential()
    model.add(Dense(input_dim=2, units=2, activation='softmax'))
    # model.add(Dense(1, activation='relu'))
    # model.add(Dense(2, activation='softmax'))
    # model.add(Dropout(0.2))

    model.compile(loss='mse',
                  optimizer='sgd',
                  metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=len(Y_train), epochs=100, verbose=1, validation_data=(X_test, Y_test))
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
print('loss:%.4f accuracy:%.4f' % (loss, accuracy))

model.save(file+'.h5', overwrite=True)  # 保存模型
model.save(file+'_'+time.strftime("%Y%m%d-%H%M%S", time.localtime())+'.h5')  # 再保存一遍，加上时间
