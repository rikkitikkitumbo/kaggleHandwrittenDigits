import tensorflow as tf
from getData import *

# set up our weights (or kernals?) and biases for each pixel
def init_Weights(shape):
    initial = tf.truncated_normal(shape, stddev=1, dtype=tf.float64)
    return tf.Variable(initial)

def init_Biases(shape):
    initial = tf.constant(1, shape=shape)
    return tf.Variable(initial)

# convolution function... pay special attention to the shapes that go into x and W
def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1,1,1,1], 'SAME')





sess = tf.Session()
shape = tf.shape(validationImages)
weights = init_Weights(shape)
product = multiplyByWeights(validationImages,weights)

sess.run(tf.initialize_all_variables())


print (sess.run(product))







