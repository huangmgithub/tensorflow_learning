from __future__ import print_function
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Params
lam = 0.01
n_epochs = 150
batch_size = 5000

# learning rate（指数衰减法）
epochs_per_decay = 10 # 每epochs衰减learning_rate
initial_learning_rate = 0.5 # 初始learning_rate
learning_rate_decay_factor = 0.95 # 衰减系数


# Network Params
hidden_1 = 500
hidden_2 = 400
n_features = 784
n_classes = 10

# Data
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# Dimension
n_samples = X_train.shape[0]

def accuracy(y_pred, y_true):
    """accuracy"""
    return 100 * np.sum(np.argmax(y_pred, 1) == np.argmax(y_true, 1)) / y_pred.shape[0]

graph = tf.Graph()
with graph.as_default():

    # data
    train_data = tf.placeholder(tf.float32, shape=([batch_size, n_features]), name="images")
    train_labels = tf.placeholder(tf.float32, shape=([batch_size, n_classes]), name="labels")
    test_data = tf.constant(X_test)

    # Store layers weight & bias
    weights = {
        'w1': tf.Variable(tf.truncated_normal([n_features, hidden_1], stddev=1 / math.sqrt(n_features)),name="w1"),
        'w2': tf.Variable(tf.truncated_normal([hidden_1, hidden_2], stddev=1 / math.sqrt(hidden_1)),name="w2"),
        'w3': tf.Variable(tf.truncated_normal([hidden_2, n_classes], stddev=1 / math.sqrt(hidden_2)),name="w3")
    }

    biases = {
        'b1': tf.Variable(tf.zeros([hidden_1]), name="b1"),
        'b2': tf.Variable(tf.zeros([hidden_2]), name="b2"),
        'b3': tf.Variable(tf.zeros([n_classes]), name="b3")
    }

    # dropout
    keep_pro = tf.placeholder(tf.float32)

    # Model
    def neural_net(x, keep_pro):
        # hidden_1 layer
        layer_1 = tf.nn.relu(tf.matmul(x, weights['w1']) + biases['b1'])
        # hidden_2 layer
        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['w2']) + biases['b2'])
        # Output layer

        # dropout：改善过拟合
        layer2_drop = tf.nn.dropout(layer_2, keep_pro)

        # Output layer
        output = tf.matmul(layer2_drop, weights['w3']) + biases['b3']
        return output

    # output
    logits = neural_net(train_data, keep_pro)

    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits) +
                          lam * tf.nn.l2_loss(weights['w1']) +
                          lam * tf.nn.l2_loss(weights['w2']) +
                          lam * tf.nn.l2_loss(weights['w3'])   # 正则项：改善过拟合
                          )

    # global step
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # learning rate policy
    decay_steps = int(n_samples / batch_size * epochs_per_decay)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               learning_rate_decay_factor,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Prediction
    train_prediction = tf.nn.softmax(logits)

    test_output = neural_net(test_data, keep_pro)
    test_prediction = tf.nn.softmax(test_output)

# save loss
loss_array = []

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
                                                 feed_dict={train_data:batch_data,
                                                            train_labels:batch_labels,
                                                            keep_pro: 0.5})

        if epoch % 10 == 0:
            print("Epoch:{}, loss:{}".format(epoch, batch_loss))
            print("Train prediction:{}".format(accuracy(prediction, batch_labels)))
            print("Test prediction:{}".format(accuracy(test_prediction.eval(
                feed_dict={train_data:batch_data, train_labels:batch_labels, keep_pro:1.}
            ), y_test)))

        loss_array.append(batch_loss)

# 可视化loss
plt.plot(np.array(range(n_epochs)), np.array(loss_array))
plt.legend()
plt.show()
