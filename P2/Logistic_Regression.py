from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
import numpy as np

mnist = input_data.read_data_sets("./data/", one_hot=True)

# Params
learning_rate = 0.01
n_epochs = 100
batch_size = 10000

# Data
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# Dimension
n_samples = X_train.shape[0]
n_feature = X_train.shape[1]
n_classes = y_train.shape[1]


def accuracy(y_pred, y_true):
    """accuracy"""
    return 100 * np.sum(np.argmax(y_pred, 1) == np.argmax(y_true, 1)) / y_pred.shape[0]

graph = tf.Graph()
with graph.as_default():

    # data
    x = tf.placeholder(tf.float32, shape=([batch_size, n_feature]), name="images")
    y = tf.placeholder(tf.float32, shape=([batch_size, n_classes]), name="labels")
    x_test = tf.constant(X_test)

    # weight and bias
    W = tf.Variable(tf.truncated_normal([n_feature, n_classes], stddev= 1 / math.sqrt(n_feature)), name="weights")
    b = tf.Variable(tf.zeros([n_classes]))

    # Model
    logits = tf.matmul(x, W) + b

    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Prediction
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(tf.matmul(x_test, W) + b)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        total_batch = int(n_samples / batch_size)
        for step in range(total_batch):
            start = step * batch_size
            end = (step + 1) * batch_size
            batch_data, batch_labels = X_train[start:end], y_train[start:end]

            # run session
            _, batch_loss, prediction = sess.run([optimizer, loss, train_prediction],
                                                 feed_dict={x:batch_data, y:batch_labels})
        if epoch % 10 == 0:
            print("epoch:{}, loss:{}".format(epoch, batch_loss))
            print("train_prediction:{}".format(accuracy(prediction, batch_labels)))
            print("test_prediction:{}".format(accuracy(test_prediction.eval(), y_test)))