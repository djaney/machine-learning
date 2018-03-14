import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

num_features = 2
clusters = 5

data = tf.random_normal([3, num_features])

x = tf.placeholder(tf.float32, shape=[None, num_features])


kmeans = KMeans(inputs=x, num_clusters=clusters, distance_metric='cosine',
use_mini_batch=True)

# Build KMeans graph
training_graph = kmeans.training_graph()

(all_scores, cluster_idx, scores, cluster_centers_initialized, cluster_centers_var, init_op, train_op) = training_graph

cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple

avg_distance = tf.reduce_mean(scores)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

array_data = sess.run(data)

sess.run(init_vars, feed_dict={x: array_data})
sess.run(init_op, feed_dict={x: array_data})

# Iterate kmeans algorithm
for i in range(100):
    _, d, data_clusters_idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={x: array_data})

groupings = [[] for x in range(clusters)]
for i in range(len(data_clusters_idx)):
	groupings[data_clusters_idx[i]].append(array_data[i])

print(groupings)