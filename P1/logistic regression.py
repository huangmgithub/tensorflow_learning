from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)

########################
### Data Processing ####
########################
# Organize the data and feed it to associated dictionaries.
data = {}
data['train/image'] = mnist.train.images
data['train/label'] = mnist.train.labels
data['test/image'] = mnist.test.images
data['test/label'] = mnist.test.labels

# Get only the samples with zero and one label for training.
index_list_train = []
for sample_index in range(data['train/label'].shape[0]):
    label = data['train/label'][sample_index]
    if label == 1 or label == 0:
        index_list_train.append(sample_index)

# Reform the train data structure.
data['train/image'] = mnist.train.images[index_list_train]
data['train/label'] = mnist.train.labels[index_list_train]

# Get only the samples with zero and one label for test set.
index_list_test = []
for sample_index in range(data['test/label'].shape[0]):
    label = data['test/label'][sample_index]
    if label == 1 or label == 0:
        index_list_test.append(sample_index)

# Reform the test data structure.
data['test/image'] = mnist.test.images[index_list_test]
data['test/label'] = mnist.test.labels[index_list_test]


######################################
######### Flags ############
######################################

tf.app.flags.DEFINE_integer('num_classes', 2,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('batch_size', 1000,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('num_epochs', 10,
                            'Number of epochs for training.')

tf.app.flags.DEFINE_float('initial_learning_rate', 0.5, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 1, 'Number of epoch pass to decay learning rate.')

# Store all elements in FLAG structure
FLAGS = tf.app.flags.FLAGS

# Dimension of train sets
d = data['train/image'].shape
n_train_samples = d[0] # the num of train sets
n_features = d[1]  # features num

#######################################
########## Defining Graph ############
#######################################
graph = tf.Graph()
with graph.as_default():
    # global step
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # learning rate policy
    decay_steps = int(n_train_samples / FLAGS.batch_size * FLAGS.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               FLAGS.learning_rate_decay_factor,
                                               staircase=True,
                                               name="exponential_decay_learning_rate")

    image_place = tf.placeholder(tf.float32, shape=([FLAGS.batch_size, n_features]), name="image")
    label_place = tf.placeholder(tf.int32, shape=([FLAGS.batch_size, ]), name="label")
    label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1) # 转换成one_hot
    print(label_one_hot.shape)
    ########### Model + Loss + Accuracy ##############
    # A simple fully connected with two class and a softmax is equivalent to Logistic Regression.
    logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=FLAGS.num_classes)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))

    def accuracy(y_pred, y_true):
        return np.sum(np.argmax(y_pred, 1) == np.argmax(y_true, 1)) / y_pred.shape[0]

    # Define optimizer by its default values
    # 梯度下降
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # gradient update.
    with tf.name_scope("train_op"):
        gradients_and_variables = optimizer.compute_gradients(loss)
        train_op  = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)

    # train prediction
    # train_prediction = tf.nn.softmax(logits)

with tf.Session(graph=graph) as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())

    for epoch in range(FLAGS.num_epochs):
        total_batch_training = int(data['train/image'].shape[0] / FLAGS.batch_size) # train 次数 per epoch

        for batch_num in range(total_batch_training):
            start = batch_num * FLAGS.batch_size
            end = (batch_num + 1) * FLAGS.batch_size

            # Fit training using batch data
            train_batch_data, train_batch_label = data['train/image'][start:end], data["train/label"][start:end]
            print(train_batch_data.shape, train_batch_label.shape)
            # run session
            batch_loss, _, training_step = sess.run(
                [loss, train_op, global_step],
            feed_dict={image_place:train_batch_data,
                       label_place:train_batch_label,
                       })

        print("Epoch" + str(epoch + 1) +", training loss=" + \
              "{:.5f}".format(batch_loss))

    # Evaluation of the model
    # test_accuracy = 100 * sess.run(accuracy, feed_dict={
    #     image_place: data['test/image'],
    #     label_place: data['test/label']})
    #
    # print("Final Test Accuracy is %% %.2f" % test_accuracy)