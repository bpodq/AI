# 在import tensorflow之前
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 不使用GPU

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import load_model
import time
from icecream import ic

import tensorflow as tf
from keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
# K.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)

start = time.time()

train_X, train_y = mnist.load_data()[0]
ic(train_X.shape)
ic(train_y.shape)

train_X = train_X.reshape(-1, 28, 28, 1)
train_X = train_X.astype('float32')
train_X /= 255
train_y = to_categorical(train_y, 10)


file = 'keras_mnist_model.h5'
if file in os.listdir('.'):
    model = load_model(file)
    # os.rename(file+)
else:
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28, 28, 1]))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

# 6000个样本，如果batch_size=100，则600批
batch_size = 100
epochs = 100
model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1
          )

test_X, test_y = mnist.load_data()[1]
ic(test_X.shape)
ic(test_y.shape)

test_X = test_X.reshape(-1, 28, 28, 1)
test_X = test_X.astype('float32')
test_X /= 255
test_y = to_categorical(test_y, 10)
loss, accuracy = model.evaluate(test_X, test_y, verbose=1)  # 输出中的313是什么？
print('loss:%.4f accuracy:%.4f' % (loss, accuracy))

end = time.time()

print("循环运行时间:%.2f秒" % (end-start))

model.save(file, overwrite=True)  # 保存模型
model.save(file+'_'+time.strftime("%Y%m%d-%H%M%S", time.localtime())+'.h5')  # 再保存一遍，加上时间

