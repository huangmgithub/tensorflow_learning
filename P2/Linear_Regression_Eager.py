import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Set Eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager

# random number
rng = np.random.RandomState(0)

# Parameter
learning_rate = 0.01
n_steps = 1000
display_step = 100

# Training Data
train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
n_samples = len(train_X)

# Params
W = tfe.Variable(rng.randn())
b = tfe.Variable(rng.randn())

# LR
def linear_regression(inputs):
    return inputs * W + b

# MSE
def MSE(model_fn, inputs ,labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2)) / (2*n_samples)

# SGD
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Compute gradient
grad = tfe.implicit_gradients(MSE)

# Training
for step in range(n_steps):
    optimizer.apply_gradients(grad(linear_regression, train_X, train_Y))
    if step % display_step == 0:
        print("step:{}, loss:{}".format(step,MSE(linear_regression, train_X, train_Y)))


# Graph display
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, np.array(W * train_X + b), label='Fitted line')
plt.legend()
plt.show()
