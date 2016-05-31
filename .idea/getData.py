# you will need these files!
# https://www.kaggle.com/c/digit-recognizer/download/train.csv
# https://www.kaggle.com/c/digit-recognizer/download/test.csv

import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

sess = tf.Session()

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
VALIDATION_SIZE = 41990

validationImages = images[:VALIDATION_SIZE]
validationLabels = labels[:VALIDATION_SIZE]

trainImages = images[VALIDATION_SIZE:]
trainLabels = labels[VALIDATION_SIZE:]

def next_batch_x(index1, index2):
    return images[index1:index2]

def next_batch_y_(index1, index2):
    val = oneHots[index1:index2,:]
    val = val.eval(session=sess)
    val = val.astype(float)
    return val


print next_batch_y_(0,10).shape

# display image
def display(img):
    # (784) => (28,28)
    one_image = img.reshape(image_width, image_height)

    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()

# output image
# display(images[2])



# -------------  Building the NN -----------------

# set up our weights (or kernals?) and biases for each pixel
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
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
image = tf.reshape(x, [-1,image_width , image_height,1])
image = tf.cast(image, tf.float32)
# print (image.get_shape()) # =>(40000,28,28,1)




h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
# print (h_conv1.get_shape()) # => (40000, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)
# print (h_pool1.get_shape()) # => (40000, 14, 14, 32)



# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#print (h_conv2.get_shape()) # => (40000, 14,14, 64)
h_pool2 = max_pool_2x2(h_conv2)
#print (h_pool2.get_shape()) # => (40000, 7, 7, 64)


# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# (40000, 7, 7, 64) => (40000, 3136)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#print (h_fc1.get_shape()) # => (40000, 1024)





# dropout
# keep_prob = tf.placeholder('float')
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# print h_fc1_drop.get_shape()


#readout layer for deep neural net
W_fc2 = weight_variable([1024,labels_count])
b_fc2 = bias_variable([labels_count])


y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

sess.run(tf.initialize_all_variables())
# print sess.run(y[0,:]) =>  [.000453, .002345, .99234, .0000234, etc]
# log(.00000000001) = -11, log(1) = 0

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#print (y.get_shape()) # => (40000, 10)
#print (y_.get_shape()) #=> (40000, 10)



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.initialize_all_variables())






with sess.as_default():
    for i in range(420):
        xx = next_batch_x(i*100, (i+1)*100)
        yy = next_batch_y_(i*100, (i+1)*100)
        train_step.run(feed_dict={x: xx, y_: yy}, session=sess)
        train_accuracy = accuracy.eval(feed_dict={x: xx, y_: yy}, session=sess)
        print train_accuracy




#
# print(sess.run(accuracy))


# print(sess.run(W_conv1))
# print(sess.run(image[0,6,14,0]))
# print(sess.run(h_conv1[0,6,14,0]))
# print(sess.run(h_conv2[0,6,14,0]))
#print(sess.run(W_conv1[2,3,0,14]))




