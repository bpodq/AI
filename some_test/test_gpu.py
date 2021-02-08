import tensorflow as tf


size = 500
W = tf.random.normal([size, size], name='W')
X = tf.random.normal([size, size], name='X')
mul = tf.matmul(W, X, name='mul')
sum_result = tf.reduce_sum(mul, name='sum')


# tfconfig = tf.ConfigProto(log_device_placement=True)
# with tf.Session(config=tfconfig) as sess:
#     result = sess.run(sum_result)

