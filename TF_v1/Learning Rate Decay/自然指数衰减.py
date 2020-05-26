import matplotlib.pyplot as plt
import tensorflow as tf

global_step = tf.Variable(0, name='global_step', trainable=False)

y = []
z = []
w = []
m = []
EPOCH = 200

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(EPOCH):

        # 阶梯型衰减
        learning_rate1 = tf.train.natural_exp_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=True)

        # 标准指数型衰减
        learning_rate2 = tf.train.natural_exp_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=False)

        # 阶梯型指数衰减
        learning_rate3 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=True)

        # 标准指数衰减
        learning_rate4 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=False)

        lr1 = sess.run([learning_rate1])
        lr2 = sess.run([learning_rate2])
        lr3 = sess.run([learning_rate3])
        lr4 = sess.run([learning_rate4])

        y.append(lr1)
        z.append(lr2)
        w.append(lr3)
        m.append(lr4)

x = range(EPOCH)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0, 0.55])

plt.plot(x, y, 'r-', linewidth=2)
plt.plot(x, z, 'g-', linewidth=2)
plt.plot(x, w, 'r--', linewidth=2)
plt.plot(x, m, 'g--', linewidth=2)

plt.title('natural_exp_decay')
ax.set_xlabel('step')
ax.set_ylabel('learning rate')
plt.legend(labels = ['natural_staircase', 'natural_continuous', 'staircase', 'continuous'], loc = 'upper right')
plt.show()