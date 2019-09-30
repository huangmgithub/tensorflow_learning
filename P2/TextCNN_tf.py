import numpy as np
import re
import itertools
from collections import Counter

def cut(string):
    """切词"""
    return list(jieba.cut(string))

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.findall(r'[\d\w]+', string)
    string = ''.join(string)
    return string.strip()

def process_labels(labels):
    """Reformat label style"""
    labels = np.array(labels)
    labels = (np.arange(1, 6) == labels[:,None]).astype(np.float32)
    return labels

def process_data(data):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Process data
    data_sets = [clean_str(sent) for sent in data]
    data_sets = [cut(s) for s in data_sets]
    return data_sets


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(data,labels):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences = process_data(data)
    labels = process_labels(labels)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]



from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime

epochs = 100
batch_size = 128

class TextCNN:
    def __init__(self, sequence_length, n_classes, vocab_size,
                 embedding_size, filter_sizes, n_filters, l2_reg_lambda):
        """
        A CNN for text classification.
        Uses an embedding layer, followed by a convolution, max-pooling and softmax layer.

        :param sequence_length: 序列长度
        :param n_classes: 类别数量
        :param vocab_size: 词库大小
        :param embedding_size: 嵌入大小
        :param filter_sizes: 过滤器大小（width不变，height变化）的集合
        :param n_filters: 相同大小的过滤器的数量
        :param l2_reg_lambda: L2 正则化参数
        :param input_x: batch data
        :param input_y: batch labels
        :param dropout_keep_pro: dropout参数
        """
        
        # Placeholder for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=(None, sequence_length), name="input_x")
        self.input_y = tf.placeholder(tf.float32, shape=(None, n_classes), name="input_y")
        self.dropout_keep_pro = tf.placeholder(tf.float32, name="dropout_keep_pro")

        # Keepping track of l2 regularization los
        l2_loss = tf.constant(0.)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # W包含词库大小的词嵌入，随机初始化
            self.W = tf.Variable(tf.random_normal([vocab_size, embedding_size], -1.0, 1.0), name="W")
            # 输入序列词对应的嵌入，(None, sequence_length, embedding_size)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 增加一个维度, 扩展至四维 (None, sequence_length, embedding_size, 1)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + max_pool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # W: [filter_height, filter_width, in_channels, out_channels]
                # Input: [batch, in_height, in_width, in_channels]
                # b: [out_channels]
                filter_shape = [filter_size, embedding_size, 1, n_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[n_filters]), name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply non linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max pooling over the outputs
                # height after conv：sequence_length - filter_size + 1
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="pool")
                # pooled_outputs: [batch, height, width, channels]
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        n_filters_total = n_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, n_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_pro)

        # Final (unnormalized) scores and prediction
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[n_filters_total, n_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(loss + l2_reg_lambda * l2_loss) # Add L2 Loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")



def train(x_train, y_train, vocab_size):
    graph = tf.Graph()
    with graph.as_default():

        sequence_length = x_train.shape[1]
        n_classes = y_train.shape[1]

        cnn = TextCNN(
        sequence_length = sequence_length,
        n_classes = n_classes,
        vocab_size = vocab_size,
        embedding_size = 128,
        filter_sizes = [2,3,4],
        n_filters = 64,
        l2_reg_lambda = 0.001)

        # Define Training procedure
        loss = cnn.loss
        accuracy = cnn.accuracy
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    with tf.Session(graph=graph) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            total_batches = x_train.shape[0] // batch_size
            for step in range(total_batches):
                x_batch = x_train[step * batch_size:(step+1) * batch_size,:]
                y_batch = y_train[step * batch_size:(step+1) * batch_size,:]
                feed_dict = {
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_pro: 0.5
                            }
                batch_loss, batch_accuracy = sess.run(
                            [loss, accuracy], feed_dict)
                time_str = datetime.datetime.now()
                if step % 100 == 0:
                    print("{}: step {}, loss {}, acc {}".format(time_str, step, batch_loss, batch_accuracy))








