from __future__ import print_function
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/', one_hot=True)

# Params
learning_rate = 0.01
n_epochs = 25
batch_size = 100
logs_path = '../data/logs/'

# Data
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# Data Params
n_features = X_train.shape[1]
n_samples = X_train.shape[0]
n_classes = y_train.shape[1]

# Graph input
x = tf.placeholder(tf.float32, shape=(batch_size, n_features), name='data')
y = tf.placeholder(tf.float32, shape=(batch_size, n_classes), name='label')

# Weight and bias
W = tf.Variable(tf.truncated_normal([n_features, n_classes], stddev=0.1), name='weight')
b = tf.Variable(tf.zeros([n_classes]), name='bias')

# Construct model and encapsulating all ops into scopes, making
# Tensorboard 's Graph visualization more convenient
with tf.name_scope('Loss'):
    logits = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for epoch in range(n_epochs):
        avg_loss = 0.
        total_batch = int(n_samples / batch_size)
        for step in range(total_batch):
            start = step * batch_size
            end = (step + 1) * batch_size
            batch_data, batch_labels = X_train[start:end], y_train[start:end]

            # run session
            _, batch_loss, summary = sess.run([optimizer, loss, merged_summary_op],
                                                 feed_dict={x: batch_data,
                                                            y: batch_labels})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + step)
            # Compute average loss
            avg_loss += batch_loss / total_batch


        print("epoch:{}, avg_loss:{}".format(epoch, avg_loss))

# print("Run the command line:\n" \
#           "--> tensorboard --logdir=../data/logs " \
#           "\nThen open http://0.0.0.0:6006/ into your web browser")