
# source: https://aimatters.wordpress.com/2016/01/16/solving-xor-with-a-neural-network-in-tensorflow/
import tensorflow as tf
import time

x_ = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input')
y_ = tf.placeholder(tf.float32, shape=[4,1], name = 'y-input')


def create_layer(name, shape):
	tf.Variable(tf.random_uniform(shape, -1, 1), name = name)

Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name = "Theta1")
Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "Theta2")

Bias1 = tf.Variable(tf.zeros([2]), name = "Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name = "Bias2")

with tf.name_scope("layer2") as scope:
	A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)

with tf.name_scope("layer3") as scope:
	Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

with tf.name_scope("cost") as scope:
	cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 
		((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)

with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

layer1 = create_layer('layer1',[2,2])
layer1 = create_layer('layer1',[2,1])


init = tf.initialize_all_variables()
sess = tf.Session()

writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph_def)

sess.run(init)

t_start = time.clock()
for i in range(100000):
	sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})


print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))

