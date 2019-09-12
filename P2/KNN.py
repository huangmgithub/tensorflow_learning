from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
import numpy as np

mnist = input_data.read_data_sets("./data/", one_hot=True)

# Data
n1 = mnist.train.images.shape[0]
random_samples = random.sample(range(n1), 2000)
X_train = mnist.train.images[random_samples]
y_train = mnist.train.labels[random_samples]

n2 = mnist.test.images.shape[0]
random_samples = random.sample(range(n2), 500)
X_test = mnist.test.images[random_samples]
y_test = mnist.test.labels[random_samples]

# Dimension
n_feature = X_train.shape[1]

graph = tf.Graph()
with graph.as_default():

    # data
    x_tr = tf.placeholder(tf.float32, shape=([None, n_feature]), name="images")
    x_te = tf.placeholder(tf.float32, shape=([n_feature]), name="labels")

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(x_tr, tf.negative(x_te))), reduction_indices=1) # 1 by cols，default:all sum
    #  Prediction: Get min distance index (Nearest neighbor)
    target = tf.argmin(distance, 0) # by rows

accuracy = 0.

with tf.Session(graph=graph) as sess:
    # initial all variables
    sess.run(tf.global_variables_initializer())

    for i in range(len(X_test)):
        # Get nearest neighbor
        target_index = sess.run(target, feed_dict={x_tr:X_train,x_te:X_test[i,:]})
        # Get nearest neighbor class label and compare it to its true label
        print("prediction label:{},true label:{}".format(
            np.argmax(y_train[target_index]),np.argmax(y_test[i])))
        # Calculate accuracy
        if np.argmax(y_train[target_index]) == np.argmax(y_test[i]):
            accuracy += 1. / len(X_test)

    print("Accuracy:",accuracy)