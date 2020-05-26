from __future__ import print_function
import tensorflow as tf
import os

"""
Since we are aimed to use Tensorboard, we need a directory to store the information (the operations and their 
corresponding outputs if desired by the user). This information is exported to event files by TensorFlow. The even 
files can be transformed to visual data such that the user is able to evaluate the architecture and the operations. 
The path to store these even files is defined as below:
"""

# The default path for saving event files is the same folder of this python file.
tf.app.flags.DEFINE_string(
'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
'Directory where event logs are written to .')

# store all elements in FLAG structure!
FLAGS = tf.app.flags.FLAGS

"""
The os.path.dirname(os.path.abspath(__file__)) gets the directory name of the current python file. 
The tf.app.flags.FLAGS points to all defined flags using the FLAGS indicator. From now on the flags can be
called using FLAGS.flag_name.
"""

# Defining some constant values
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name='b')
# Some basic operations
x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")


# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print("output:", sess.run([a, b, x, y]))

# Closing the writer
writer.close()
sess.close()

"""
The tf.summary.FileWriter is defined to write the summaries into event files.The command of sess.run() must be used 
for evaluation of any Tensor otherwise the operation won't be executed. In the end by using the writer.close(), the 
summary writer will be closed.
"""