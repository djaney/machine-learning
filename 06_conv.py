
'''
Use MNIST database

convert it to convolutional layer
'''

# Imports
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()
mnist = input_data.read_data_sets(FLAGS.data_dir)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.95, staircase=True)



# input - None is for batch, 3 is for number of input per batch
x = tf.placeholder(tf.float32, [None,784])
x_reshaped = tf.reshape(x, [-1,28,28 ,1])
c1 = tf.layers.conv2d(x_reshaped,4, 5, activation=tf.nn.relu) # 4 channel output, 5x5 filter
c1 = tf.layers.max_pooling2d(c1, 1, 1) # 1x1 with 1 stride

c2 = tf.layers.conv2d(c1,8, 4, activation=tf.nn.relu)
c2 = tf.layers.max_pooling2d(c2, 2, 2)

c3 = tf.layers.conv2d(c2,12, 4, activation=tf.nn.relu)
c3 = tf.layers.max_pooling2d(c3, 2, 2)

f = tf.layers.flatten(c3)
h1 = tf.layers.dense(f, 200, tf.nn.relu)
m_ = tf.layers.dense(h1, 10, tf.nn.softmax) # for testing
m = tf.layers.dropout(m_, rate=0.25) # for training


# initialize the variables defined above
init = tf.global_variables_initializer()



# labes or correct answers
y = tf.placeholder(tf.int64, [None])

# calculate the loss distance using cross entropy
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=m)
cross_entropy_ = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=m_)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(10000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_data = {x: batch_xs, y: batch_ys}
	#training
	sess.run(train_step, feed_dict=train_data)
	if 0 == i % 100:
		correct_prediction = tf.equal(tf.argmax(m_, 1), y)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# train
		a,c = sess.run([accuracy, cross_entropy_], feed_dict={x: batch_xs, y: batch_ys})
		# test
		a_,c_ = sess.run([accuracy, cross_entropy_], feed_dict={x: mnist.test.images,y: mnist.test.labels})
		print(a,c,a_,c_)