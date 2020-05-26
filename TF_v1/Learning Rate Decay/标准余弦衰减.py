import matplotlib.pyplot as plt
import tensorflow as tf

y = []
z = []
EPOCH = 200
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(EPOCH):
        # 余弦衰减
        learning_rate1 = tf.train.cosine_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=50)
        learning_rate2 = tf.train.cosine_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=100)

        lr1 = sess.run([learning_rate1])
        lr2 = sess.run([learning_rate2])
        y.append(lr1)
        z.append(lr2)

x = range(EPOCH)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, y, 'r-', linewidth=2)
plt.plot(x, z, 'b-', linewidth=2)
plt.title('cosine_decay')
ax.set_xlabel('step')
ax.set_ylabel('learning rate')
plt.legend(labels=['decay_steps=50', 'decay_steps=100'], loc='upper right')
plt.show()