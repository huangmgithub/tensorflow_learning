"""
For a variable generation, the class of tf.Variable() will be used. When we define a variable, we basically pass a
tensor and its value to the graph. Basically, the following will happen:

+ A variable tensor that holds a value will be pass to the graph.
+ By using tf.assign, an initializer set initial variable value.

Some arbitrary variables can be defined as follows:
"""
import tensorflow as tf
from tensorflow.python.framework import ops

#######################################
######## Defining Variables ###########
#######################################

# Create three variables with some default values.
weights = tf.Variable(tf.random_normal([2,3], stddev=0.1), name="weights")
biases = tf.Variable(tf.zeros([3]), name="biases")
custom_variable = tf.Variable(tf.zeros([3]), name="custom")

# Get all the variables' tensor and store them in a list
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

# Initialization

# 1.Initializing Specific Variables

# "variable_list_custom" is the list of variables that we want to initialize.
variable_list_custom = [weights, custom_variable]

# The initializer
init_custom_op = tf.variables_initializer(var_list=variable_list_custom)

# 2.Global variable initialization
# Method-1
# Add an op to initialize the vairables
init_all_op = tf.global_variables_initializer()

# Method-2
init_all_op = tf.variables_initializer(var_list=all_variables_list)

# 3.Initialization of a variables using other existing variables
# Create another variable with the same value as 'weights'.
WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

# Now, the variable must be initialized.
init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])

# All we did so far was to define the initializers' ops and put them on the graph. In order to truly initialize
# variables, the defined initializers' ops must be run in the session. The script is as follows:
with tf.Session() as sess:
    # Run the initializer operation
    sess.run(init_all_op)
    sess.run(init_custom_op)
    sess.run(init_WeightsNew_op)
