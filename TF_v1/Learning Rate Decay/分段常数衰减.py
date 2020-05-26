import matplotlib.pyplot as plt
import tensorflow as tf

global_step = tf.Variable(0, name='global_step', trainable=False) # 迭代次数
boundaries = [10, 20, 30]
learning_rates = [0.1, 0.07, 0.025, 0.0125]

y = []
N = 40

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(N):
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learning_rates)
        lr = sess.run([learning_rate])
        y.append(lr)

x = range(N)
plt.plot(x, y, 'r-', linewidth=2)
plt.title('piecewise_constant')
plt.show()