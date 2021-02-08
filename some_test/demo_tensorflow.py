

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

import tensorflow as tf

import numpy as np

hello=tf.constant('hhh')

sess=tf.compat.v1.Session()


print (sess.run(hello))

