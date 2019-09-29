import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
step_size = 28
input_size = 28
output_size = 10

learning_rate = 0.001

batch_size = 128
n_epochs = 10

hidden_size = 128

# Data
mnist = input_data.read_data_sets("../data/")


X_train = mnist.train.images
X_train = X_train.reshape([-1, step_size, input_size])
y_train = mnist.train.labels
X_test = mnist.test.images
X_test = X_test.reshape([-1, step_size, input_size])
y_test = mnist.test.labels

n__examples = X_train.shape[0]

X = tf.placeholder(tf.float32, shape=(batch_size, step_size, input_size))
y = tf.placeholder(tf.int32, shape=(batch_size,))

# RNN
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
print(state)

# Loss
logits = tf.layers.dense(state, output_size)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Prediction
prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Session Init")
    for epoch in range(n_epochs):
        total_batch = n__examples // batch_size
        for batch in range(total_batch):
            batch_train = X_train[batch*batch_size:(batch+1)*batch_size,:,:]
            batch_labels = y_train[batch*batch_size:(batch+1)*batch_size]
            _, batch_loss, train_accuracy = sess.run([optimizer, loss, accuracy],
                     feed_dict={X:batch_train, y:batch_labels})
        if epoch % 2 == 0:
            print("Epoch:{}, loss:{}, accuracy:{}".format(epoch, batch_loss, train_accuracy))


