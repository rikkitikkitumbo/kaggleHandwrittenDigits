import numpy as np
import pandas as pd
import tensorflow as tf


# read in the image data from the csv file
# the format is:    imagelabel  pixel0  pixel1 ... pixel783  (there are 42,000 rows like this)
data = pd.read_csv('../train.csv')
labels = data.iloc[:,:1].values.ravel()  # shape = (42000, 1)
labels_count = np.unique(labels).shape[0]  # = 10
images = data.iloc[:,1:].values   # shape = (42000, 784)
images = images.astype(np.float64)
image_size = images.shape[1]
image_width = image_height = np.sqrt(image_size).astype(np.int32)  # since these images are sqaure... hieght = width


# turn all the gray-pixel image-values into percentages of 255
# a 1.0 means a pixel is 100% black, and 0.0 would be a pixel that is 0% black (or white)
images = np.multiply(images, 1.0/255)


# create oneHot vectors from the label #s
oneHots = tf.one_hot(labels, labels_count, 1, 0)  #shape = (42000, 10)


#split up the training data even more (into validation and train subsets)
VALIDATION_SIZE = 2000

validationImages = images[:VALIDATION_SIZE]
validationLabels = labels[:VALIDATION_SIZE]

trainImages = images[VALIDATION_SIZE:]
trainLabels = labels[VALIDATION_SIZE:]





# -------------  Building the NN -----------------

# set up our weights (or kernals?) and biases for each pixel
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1,1,1,1], 'SAME')

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# placeholder variables
# images
x = tf.placeholder('float', shape=[None, image_size])
# labels
y_ = tf.placeholder('float', shape=[None, labels_count])



# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# turn shape(40000,784)  into   (40000,28,28,1)
image = tf.reshape(trainImages[0], [-1,image_width , image_height,1])
image = tf.cast(image, tf.float32)
# print (image.get_shape()) # =>(40000,28,28,1)




h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_conv2 = conv2d(image, W_conv1) + b_conv1
# print (h_conv1.get_shape()) # => (40000, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) # => (40000, 14, 14, 32)


# Prepare for visualization
# display 32 fetures in 4 by 8 grid
layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))

# reorder so the channels are in the first dimension, x and y follow.
layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))

layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8))






sess = tf.Session()
sess.run(tf.initialize_all_variables())
# print(sess.run(W_conv1))
print(sess.run(image[0,6,14,0]))
print(sess.run(h_conv1[0,6,14,0]))
print(sess.run(h_conv2[0,6,14,0]))
print(sess.run(W_conv1[2,3,0,14]))





