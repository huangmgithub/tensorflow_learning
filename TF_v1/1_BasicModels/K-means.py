from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import numpy as np

# Ignore all GPUs, tf k-means does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

mnist = input_data.read_data_sets("./data/", one_hot=True)
full_data_x = mnist.train.images

# Params
n_steps = 50
batch_size = 1024
k = 25
n_classes = 10
n_features = 784

# Images
X = tf.placeholder(tf.float32, shape=(None, n_features))
y = tf.placeholder(tf.float32, shape=(None, n_classes))

# K-means
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

# Build Kmeans graph
training_graph = kmeans.training_graph()

if len(training_graph) > 6:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start Session
sess = tf.Session()

# Run initializer
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, n_steps+1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X:full_data_x})
    if i % 10 == 0 or i == 1:
        print("step:%s, avg:%s" % (i, d))

# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')

# print(idx) idx 每个样本所属的聚类中心
# print(len(idx))
# print(idx.max())
counts = np.zeros(shape=(k, n_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i] # 每个聚类中心对应的标签数量

# Assign the most frequent label to the centroid  -> 并没与取中心值坐标
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
# tf.nn.embedding_lookup:选取一个张量里面索引对应的元素
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx) # 每个元素为对应样本的label(标量)

# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, y: test_y}))
