import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.utils import check_random_state


# Generating data by hand
n = 50
X = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-20, 20, size=(n,)) + 2.0 * X
data = np.stack([X, y], axis=1)

#######################
## Defining flags #####
#######################
tf.app.flags.DEFINE_integer('num_epochs', 50, 'The number of epochs for training the model. Default=50')
# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# Creating the weight and bias
# The defined variables will be initialized to zero
W = tf.Variable(0., name="weights")
b = tf.Variable(0., name="bias")


#  Creating placeholders for input X and label Y.
def inputs():
    """
    Defining the place_holders.
    :return:
            Returning the data and label lace holders.
    """
    X = tf.placeholder(tf.float32, shape=(data.shape[0],), name="X")
    y = tf.placeholder(tf.float32, shape=(data.shape[0],), name="y")
    return X, y


# Creating the prediction
def inference(X):
    """
    Forward passing the X.
    :param X: Input.
    :return: X*W + b.
    """
    return X * W + b


# Loss
def loss(X, y):
    """
    compute the loss by comparing the predicted value to the actual label.
    :param X: The input.
    :param Y: The label.
    :return: The loss over the samples.
    """
    y_pred = inference(X)
    return tf.reduce_sum(tf.squared_difference(y, y_pred)) / (2 * data.shape[0])


# The training function
def train(loss):
    learning_rate = 0.001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    # Initialize the variable [w and b]
    sess.run(tf.global_variables_initializer())

    # Get the input tensors
    X, y = inputs()

    # Return the train loss and create the train_op
    train_loss = loss(X, y)
    train_op = train(train_loss)

    # train the model
    for epoch_num in range(100):
        loss_value, _ = sess.run([train_loss, train_op], feed_dict={X: data[:, 0], y: data[:, 1]})

        # Displaying the loss per epoch
        print("epoch %s, loss=%s" % (epoch_num + 1, loss_value))

        # save the values of weight and bias
        weight, bias = sess.run([W, b])


##############################
#### Evaluate and plot ########
###############################
Input_values = data[:,0]
Labels = data[:,1]
Prediction_values = data[:,0] * weight + bias
plt.plot(Input_values, Labels, 'ro', label='main')
plt.plot(Input_values, Prediction_values, label='Predicted')

# Saving the result.
plt.legend()
# plt.savefig('plot.png')
# plt.close()
plt.show()