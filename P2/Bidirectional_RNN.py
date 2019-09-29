from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

# Import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/', one_hot=True)

'''
To classify images using a bidirectional recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training parameters
learning_rate = 0.001
n_epochs = 100
batch_size = 128

# Network parameters
input_size = 28
step_size = 28
n_hidden = 128
n_classes = 10
n_examples = mnist.train.images.shape[0]

# Graph input
X = tf.placeholder(tf.float32, shape=(batch_size, step_size, input_size), name="Input")
y = tf.placeholder(tf.float32, shape=(batch_size, n_classes), name="Label")

# Weights
weights = {
    'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(x, weights, biases):
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, step_size, axis=1) # Many tensor ->(batch_size, n_inputs)

    # Define a lstm cell with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                     dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                               dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
print(logits.shape)

# Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
loss_count = []

with tf.Session() as sess:
    # Run Init
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        total_batches = n_examples // batch_size
        for batch in range(total_batches):
            x = mnist.train.images[batch*batch_size:(batch+1)*batch_size, :]
            batch_x = x.reshape((-1, step_size, input_size))
            batch_y = mnist.train.labels[batch*batch_size:(batch+1)*batch_size]
            batch_loss, _, batch_accuracy = sess.run([loss, optimizer, accuracy],
                                                       feed_dict={X:batch_x,y:batch_y})

            if batch % 100 == 0:
                print('batch:{}, loss:{}, accuracy:{}'.format(batch, batch_loss, batch_accuracy))
                loss_count.append(batch_loss)

X = len(loss_count)
plt.plot(range(X), loss_count)
plt.xlabel("epochs")
plt.ylabel("loss")
