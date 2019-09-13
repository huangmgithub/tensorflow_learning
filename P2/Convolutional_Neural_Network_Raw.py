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
n_epochs = 5
batch_size = 512

# learning rate（指数衰减法）
epochs_per_decay = 10 # 每epochs衰减learning_rate
initial_learning_rate = 0.005 # 初始learning_rate
learning_rate_decay_factor = 0.95 # 衰减系数

# Network Params
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
    test_data = tf.constant(X_test[:1280])

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="w1"),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="w2"),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name="w3"),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]), name="w_out")
    }

    biases = {
        'bc1': tf.Variable(tf.zeros([32]), name="b1"),
        'bc2': tf.Variable(tf.zeros([64]), name="b2"),
        'bd1': tf.Variable(tf.zeros([1024]), name="b3"),
        'out': tf.Variable(tf.zeros([n_classes]), name="b_out")
    }

    # dropout
    keep_pro = tf.placeholder(tf.float32)

    # Function
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        X = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

    def conv_net(x, keep_pro):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully Connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.matmul(fc1, weights['wd1']) + biases['bd1']
        fc1 = tf.nn.relu(fc1)

        # dropout：改善过拟合
        fc1 = tf.nn.dropout(fc1, keep_pro)

        # Output
        out = tf.matmul(fc1, weights['out']) + biases['out']
        return out

    # output
    logits = conv_net(train_data, keep_pro)

    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits)
                          # lam * tf.nn.l2_loss(weights['wc1']) +
                          # lam * tf.nn.l2_loss(weights['wc2']) +
                          # lam * tf.nn.l2_loss(weights['wd1']) +
                          # lam * tf.nn.l2_loss(weights['out'])  # 正则项：改善过拟合
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
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Prediction
    train_prediction = tf.nn.softmax(logits)

    test_output = conv_net(test_data, keep_pro)
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
                                                 feed_dict={train_data: batch_data,
                                                            train_labels: batch_labels,
                                                            keep_pro: 0.75})

            if step % 20 == 0:
                print("Step:{}, loss:{}".format(step, batch_loss))
                print("Train prediction:{}".format(accuracy(prediction, batch_labels)))
                print("Test prediction:{}".format(accuracy(test_prediction.eval(
                    feed_dict={train_data: batch_data, train_labels: batch_labels, keep_pro: 1.0}
                ), y_test[:1280])))

        loss_array.append(batch_loss)

# 可视化loss
plt.plot(np.array(range(n_epochs)), np.array(loss_array))
plt.legend()
plt.show()

