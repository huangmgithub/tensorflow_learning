import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# random number
# rng = np.random.RandomState(0)

# Parameter
learning_rate = 0.01
num_epochs = 1000
display_step = 50

# Training data
data = load_boston()
X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Dimension
n_samples = X_train.shape[0]
n_features  = X_train.shape[1]

graph = tf.Graph()
with graph.as_default():

    # Data
    X = tf.placeholder(tf.float32, shape=(n_samples, n_features))
    Y = tf.placeholder(tf.float32, shape=(n_samples,))
    x_test = tf.constant(X_test)

    # Params
    W = tf.Variable(tf.truncated_normal([n_features, 1], stddev= 1 / math.sqrt(n_features)), name="weight")
    b = tf.Variable(tf.zeros([1]), name='bias')
    print(b,W)

    # Model
    train_prediction = tf.matmul(X, W) + b

    # Loss
    loss = tf.reduce_mean(tf.pow(train_prediction - Y, 2))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Prediction
    test_prediction = tf.matmul(x_test, W) + b

with tf.Session(graph=graph) as sess:

    # Run initializer
    sess.run(tf.global_variables_initializer())

    # Fitting training data
    for epoch in range(num_epochs):
        loss = sess.run(loss, feed_dict={X: X_train, Y: y_train})
        if epoch % display_step == 0:
            print("epoch:{}, loss:{}, w:{}, b:{}".format(epoch, loss, sess.run(W), sess.run(b)))

    print("Optimization Finished")

    # Graph display
    plt.plot(X_train, y_train, 'ro', label='Original data')
    plt.plot(X_train, sess.run(train_prediction), label='Fitted line')
    plt.legend()
    plt.show()

    plt.plot(X_test, y_test, 'bo', label='Testing data')
    plt.plot(X_test, sess.run(test_prediction), label='Fitted line')
    plt.legend()
    plt.show()




