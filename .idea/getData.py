import numpy as np
import pandas as pd
import tensorflow as tf


# read in the image data from the csv file
# the format is:    imagelabel  pixel0  pixel1 ... pixel783  (there are 42,000 rows like this)
data = pd.read_csv('../train.csv')
labels = data.iloc[:,:1].values.ravel()  # shape = (42000, 1)
labelsCount = np.unique(labels).shape[0]  # = 10
images = data.iloc[:,1:].values   # shape = (42000, 784)
images = images.astype(np.float)


# turn all the gray-pixel image-values into percentages of 255
# a 1.0 means a pixel is 100% black, and 0.0 would be a pixel that is 0% black (or white)
images = np.multiply(images, 1.0/255)


# create oneHot vectors from the label #s
oneHots = tf.one_hot(labels, labelsCount, 1, 0)  #shape = (42000, 10)


#split up the training data even more (into validation and train subsets)
VALIDATION_SIZE = 2000

validationImages = images[:VALIDATION_SIZE]
validationLabels = labels[:VALIDATION_SIZE]

trainImages = images[VALIDATION_SIZE:]
trainLabels = labels[VALIDATION_SIZE:]



# set up our weights (or kernals?) and biases for each pixel
def init_Weights(shape):
    initial = tf.truncated_normal(shape, stddev=1, dtype=tf.float64)
    return tf.Variable(initial)

def init_Biases(shape):
    initial = tf.constant(1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)


# multiply our images tensor by the weights tensor
def multiplyByWeights(x, W):
    return tf.nn.conv2d(x, W, [1,1,1,1], 'SAME')




#


# print validationLabels.shape
# print trainImages.shape
#
# sess = tf.Session()
#
# print (sess.run(tf.shape(oneHots)))
# print labels.shape
# print labelsCount




