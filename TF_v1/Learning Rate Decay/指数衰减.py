import matplotlib.pyplot as plt
import tensorflow as tf

global_step = tf.Variable(0, name='global_step', trainable=False) # 迭代次数

y = []
z = []
epochs = 200

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(epochs):
        # 阶梯型衰减
        learning_rate_1 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=True
        )
        # 标准指数衰减
        learning_rate_2 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=False
        )
        lr1 = sess.run([learning_rate_1])
        lr2 = sess.run([learning_rate_2])
        y.append(lr1)
        z.append(lr2)

x = range(epochs)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0, 0.55])

plt.plot(x, y, 'r-', linewidth=2)
plt.plot(x, z, 'g-', linewidth=2)
plt.title('exponential_decay')
ax.set_xlabel('step')
ax.set_ylabel('learning_rate')
plt.legend(labels=['staircase', 'continuous'], loc='upper right')
plt.show()