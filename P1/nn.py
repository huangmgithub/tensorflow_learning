from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

##### FLAGS #####
tf.app.flags.DEFINE_integer('num_classes', 10,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('batch_size', 10000,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('num_epochs', 10,
                            'Number of epochs for training.')

tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 10, 'Number of epoch pass to decay learning rate.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

##### Data #####
mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=True)
train_data = mnist.train.images
train_label = mnist.train.labels
test_data = mnist.test.images
test_label = mnist.test.labels

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

# Dimension of train sets
d = train_data.shape
n_train_samples = d[0] # the num of train sets
n_features = d[1]  # features num

##### Graph #####
graph = tf.Graph()
with graph.as_default():
    # global step
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # learning rate policy
    decay_steps = int(n_train_samples / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               FLAGS.learning_rate_decay_factor,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')

    image_place = tf.placeholder(tf.float32, shape=([None, n_features]), name="image")
    label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name="label")

    # LAYER-1
    net = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=1024, scope='fc-1')

    # LAYER-2
    net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=512, scope='fc-2')

    # Softmax
    logits = tf.contrib.layers.fully_connected(inputs=net, num_outputs=FLAGS.num_classes, scope='fc-3')

    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_place))

    # Accuracy
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(label_place, 1)), "float")
    )

    # Train
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Gradient update
    with tf.name_scope('train_scope'):
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)

##### Graph #####
with tf.Session(graph=graph) as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())

    for epoch in range(FLAGS.num_epochs):
        total_batch_training = int(train_data.shape[0] / FLAGS.batch_size)

        for batch_num in range(total_batch_training):
            start = batch_num * FLAGS.batch_size
            end = (batch_num + 1) * FLAGS.batch_size

            # batch data
            train_batch_data,train_batch_label = train_data[start:end], train_label[start:end]

            # run
            batch_loss, _, training_step = sess.run(
                [loss, train_op, global_step],
                feed_dict={image_place:train_batch_data,
                           label_place:train_batch_label}
            )

        print("Epoch:{},Train Loss:{}".format(epoch, batch_loss))

        #Evaluation of the model
        total_test_accuracy = sess.run(accuracy, feed_dict={
            image_place: test_data,
            label_place: test_label})

        print(total_test_accuracy)