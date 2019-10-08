import matplotlib.pyplot as plt
import tensorflow as tf

y = []
z = []
EPOCH = 200
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(EPOCH):
        # 阶梯型衰减
        learning_rate1 = tf.train.inverse_time_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=20,
            decay_rate=0.2, staircase=True)

        # 连续型衰减
        learning_rate2 = tf.train.inverse_time_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=20,
            decay_rate=0.2, staircase=False)

        lr1 = sess.run([learning_rate1])
        lr2 = sess.run([learning_rate2])

        y.append(lr1)
        z.append(lr2)

x = range(EPOCH)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, z, 'r-', linewidth=2)
plt.plot(x, y, 'g-', linewidth=2)
plt.title('inverse_time_decay')
ax.set_xlabel('step')
ax.set_ylabel('learning rate')
plt.legend(labels=['continuous', 'staircase'])
plt.show()