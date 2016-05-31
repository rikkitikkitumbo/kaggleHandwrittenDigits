
import numpy as np
import tensorflow as tf
sess = tf.Session()


# ---------------tf.transpose()
# tensr = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
# tensr = tf.ones([2,2,3])
#
# print tensr.get_shape() # => (2,2,3)
# # print sess.run(tensr)
#
# tensr2 = tf.transpose(tensr,perm=[2,1,0])
# print tensr2.get_shape() # => (3,2,2)
# # print sess.run(tensr2)
#
# tensr3 = tf.transpose(tensr,perm=[1,2,0])
# print tensr3.get_shape() # => (2,3,2)
# # print sess.run(tensr3)


# --------------tf.conv2d
# tensr = tf.constant([[1,2,3,4,5,6,7,8.0,9,10,11,12,13,14,15,16]])
# tensr = tf.ones([1,16])
# print tensr.get_shape() # => (1,16)
# print sess.run(tensr)
#
# tensr = tf.reshape(tensr,[-1,4,4,1])
# print tensr.get_shape() # => (1,4,4,1)
#
# # tensr2 = tf.constant([[1,2],[5,2]])
# tensr2 = tf.ones([2,2,1,4],dtype=tf.float32)
#
# print tensr2.get_shape() # => (2,2,1,5)
#
# conv = tf.nn.conv2d(tensr,tensr2,[1,1,1,1],'SAME')
# print conv.get_shape # => (1,4,4,5)
# # print sess.run(conv[0,3,0,4])
# conv = tf.reshape(conv,[-1])
# print sess.run(conv)
# print conv.get_shape()
#
# r = tf.fill([55000],0.34)
# # r = tf.ones([55000])

g = tf.fill([3,4],2)

j = tf.fill([4,2],3)

r = tf.matmul(j,g)


print r.get_shape()
sess.run(r)










