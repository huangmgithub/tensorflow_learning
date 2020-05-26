from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from six.moves.urllib.request import urlretrieve
import zipfile
from collections import Counter, deque
import random

# Training Params
batch_size = 128
n_steps = 500
display_step = 100
learning_rate = 0.1

# Evaluation Params
eval_words =['nine', 'of', 'going', 'hardware', 'american', 'britain']

# Word2vec Params
embedding_size = 200 # 词嵌入向量维度
max_vocabulary_size = 50000 # 词汇表中不同单词的总数
min_occurrence = 10  # 移除未出现至少多少次的单词
skip_window = 3 # 考虑的上下文单词数
n_skips = 2 # 重复使用 一个输入生成标签的次数
n_ne_sample = 64 # 可用的负样本数量

# Download
url = 'http://mattmahoney.net/dc/text8.zip'
data_path = './data/text8.zip'
if not os.path.exists(data_path):
    print("Download the data sets")
    filename, _ = urlretrieve(url, data_path)
    print("Done")

# Unzip
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()

text_words = [str(word, encoding='utf-8') for word in text_words]
# Build dictionary and replace rare words with UNK token
count = [('UNK', -1)]
# Retrieve the most common
count.extend(Counter(text_words).most_common(max_vocabulary_size - 1))
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        break

# Compute the vocabulary size
vocabulary_size = len(count)
# Assign an id to each word
word_2_id = dict()
for i, (word, _) in enumerate(count):
    word_2_id[word] = i

data = []
unk_count = 0
for word in text_words:
    # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
    index = word_2_id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
id_2_word = dict(zip(word_2_id.values(), word_2_id.keys()))

print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print('Most common words:', count[:10])

data_index = 0

# Generate training batch for the skpi-gram model
def next_batch(batch_size, n_skips, skip_window):
    global data_index
    assert batch_size % n_skips == 0
    assert n_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # Get window size (words left and right + current one)
    span = 2 * skip_window + 1
    buffer = deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index+span])
    data_index += span
    for i in range(batch_size // n_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, n_skips) # 随机选两个词作为样本
        for j, context_word in enumerate(words_to_use):
            batch[i * n_skips + j] = buffer[skip_window]
            labels[i * n_skips +j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skip words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)

    return batch, labels

# Input data
X = tf.placeholder(tf.int32, shape=(None,), name='context_word')
y = tf.placeholder(tf.int32, shape=(None, 1), name='key_word')

with tf.device('/cpu:0'):
    # Create the embedding variable (each row represent a word embedding vector)
    embedding = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    # Lookup the corresponding embedding vectors for each sample in X
    X_embed = tf.nn.embedding_lookup(embedding, X)

    # Construct the variable for the NCE loss
    nce_weights = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch
loss = tf.reduce_mean(tf.nn.nce_loss(
    weights=nce_weights,
    biases=nce_biases,
    labels=y,
    inputs=X_embed,
    num_sampled=n_ne_sample,
    num_classes=vocabulary_size
))

# Define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluation
# Compute the cosine similarity between input data embedding and every embedding vectors
X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))
embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
cosine_sim = tf.matmul(X_embed_norm, embedding_norm, transpose_b=True) # b transpose


# Run Session
with tf.Session() as sess:
    # initial
    sess.run(tf.global_variables_initializer())

    # Test data
    x_test = np.array([word_2_id[w] for w in eval_words])

    average_loss = 0
    for step in range(1, n_steps+1):
        # batch data
        batch_x, batch_y = next_batch(batch_size, n_skips, skip_window)
        # training
        _, train_loss, sim = sess.run([optimizer, loss, cosine_sim],
                                 feed_dict={X:batch_x, y:batch_y})
        average_loss += loss

        if step % display_step == 0 or step == 1:
            if step > 1:
                average_loss /= display_step
            print('step:{}, average loss:{}, sim:{}'.format(step, average_loss, sim))
            average_loss = 0 #