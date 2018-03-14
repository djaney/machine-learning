
'''
Very simple neural network that identifies if the three numbers are high or low.
labels has 2 values, for low (0) and high (1)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf


# just 2 test datas
data = np.array([
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0]
])

# labels or the correct answers of the test datas
labels = np.array([[0.0,1.0],[1.0,0.0]])


# input - None is for batch, 3 is for number of input per batch
x = tf.placeholder(tf.float32, [None,3])
w = tf.Variable(tf.zeros([3,2])) # create 3 weights (1 for each input) for 2 neurons
b = tf.Variable(tf.zeros([2])) # create a bias for each neuron. This is just another weight that we do not matmul with the input

# initialize the variables defined above
init = tf.initialize_all_variables()

# model using softmax activation function. Formula is always input * weights + bias then feed to the activation
m = tf.nn.softmax(tf.matmul(x, w) + b)

# labes or correct answers
y = tf.placeholder(tf.float32, [None, 2])

# calculate the loss distance using cross entropy
cross_entropy = -tf.reduce_sum(y * tf.log(m))

# check if highest probability is the same as correct answers
is_correct = tf.equal(tf.argmax(m,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(1000):
	train_data = {x: data, y: labels}
	a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
	sess.run(train_step, feed_dict=train_data)

	if (0 == i % 100):
		a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
		print(a,c)

r = sess.run(m, feed_dict=train_data)
print(r)