from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/', one_hot=False) # y非one_hot，为标量

print(mnist.train.images.shape)
print(mnist.train.labels.shape)

# Params
n_steps = 50
batch_size = 1024
n_classes = 10
n_features = 784
n_trees = 10
max_nodes = 1000

# Data
X = tf.placeholder(tf.float32, shape=([None, n_features]))
# For random forest, labels must be integers (the class id)
y = tf.placeholder(tf.int32, shape=([None]))

# Random forest
hparams = tensor_forest.ForestHParams(
    num_classes = n_classes,
    num_features = n_features,
    num_trees = n_trees,
    max_nodes = max_nodes
).fill()

# Build random forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Getting training graph and loss
train_op = forest_graph.training_graph(X, y)
loss_op = forest_graph.training_loss(X, y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)

correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))

# Start session
sess = tf.Session()

# Run initializer
sess.run(init_vars)

# Training
for i in range(1 ,n_steps+1):
    start = (i-1) * batch_size
    end = i * batch_size
    batch_x, batch_y = mnist.train.images[start:end], mnist.train.labels[start:end]
    _, l, acc= sess.run([train_op, loss_op, accuracy_op], feed_dict={X: batch_x, y: batch_y})
    print(acc)
    if i % 50 == 0 or i == 1:
        print('step:{}, loss:{}, acc:{}'.format(i, l, acc))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test accuracy:{}".format(sess.run(accuracy_op, feed_dict={X:test_x,y:test_y})))