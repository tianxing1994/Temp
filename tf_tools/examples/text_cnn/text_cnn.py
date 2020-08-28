#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, x, y_true, y_pred):
        self.x = x
        self.y_true = y_true
        self.y_pred = y_pred

        self.loss = None
        self.metrics = None
        self.optimizer = None

        self.sess = None

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def fit(self, x, y, batch_size, epochs):
        loss = self.loss(self.y_true, self.y_pred)
        metrics = list()
        for metric_fn in self.metrics:
            metric = metric_fn(self.y_true, self.y_pred)
            metrics.append(metric)

        train_op = self.optimizer.minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = list()
            epoch_metrics = list()
            for batch_x, batch_y in self.get_next_batch(x, y, batch_size=batch_size):
                _, loss_, metrics_ = self.sess.run(
                    fetches=[train_op, loss, metrics],
                    feed_dict={self.x: batch_x, self.y_true: batch_y}
                )

                epoch_loss.append(loss_)
                epoch_metrics.append(metrics_)

            loss_ = np.mean(np.array(epoch_loss), axis=0)
            metrics_ = np.mean(np.array(epoch_metrics), axis=0)
            print(f'epoch: {epoch}, loss: {loss_}, metrics: {metrics_}')
        return

    def restore(self, save_path):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, save_path)
        return

    def predict(self, inputs):
        y_pred = tf.nn.softmax(self.y_pred, axis=-1)
        output = self.sess.run(y_pred, feed_dict={self.x: inputs})
        return output

    def save(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
        return

    @staticmethod
    def get_next_batch(x, y, batch_size):
        l = len(x)
        idx = np.arange(l)
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]
        steps = l // batch_size

        for step in range(steps):
            b_idx = step * batch_size
            e_idx = b_idx + batch_size
            batch_x = x[b_idx: e_idx]
            batch_y = y[b_idx: e_idx]
            yield batch_x, batch_y

    @staticmethod
    def loss(y_true, y_pred, l2_loss=None, l2_reg_lambda=0.0):
        if l2_loss is None:
            l2_loss = 0.0
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
            losses = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        return losses

    @staticmethod
    def accuracy(y_true, y_pred):
        with tf.name_scope('accuracy'):
            y_pred = tf.nn.softmax(logits=y_pred, axis=-1, name='softmax')
            y_pred = tf.argmax(y_pred, axis=-1, name='argmax')
            y_true = tf.argmax(y_true, axis=-1)
            correct = tf.equal(y_pred, y_true)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        return accuracy


class Embeddings(object):
    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embeddings = self.init_embeddings()

    def init_embeddings(self):
        embeddings = tf.get_variable(
            name='embeddings',
            shape=(self.vocab_size, self.embedding_size),
            dtype=tf.float32
        )
        return embeddings

    def embedding_lookup(self, ids: list):
        sentence = tf.nn.embedding_lookup(params=self.embeddings, ids=ids)
        return sentence


class TextConv(object):
    """
    VALID: 卷积, 只在卷积核 kernel 始终在 input 内, 不能 padding 到外部. 所以卷积后的 output 大小比 input 要小.
    SAME: 卷积, 通过 padding, 卷积核可以扩展到 input 外部, 以使得卷积后的 output 大小与 input 相同.
    """
    def __init__(self, height, width, in_channels, out_channels, variable_scope=None):
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable_scope = 'text_conv' if variable_scope is None else variable_scope

    def __call__(self, input):
        if isinstance(self.variable_scope, str):
            scope = tf.variable_scope(self.variable_scope)
        else:
            scope = self.variable_scope
        with scope:
            return self.text_conv(input)

    def text_conv(self, input):
        w = tf.get_variable(
            name='w',
            shape=(self.height, self.width, self.in_channels, self.out_channels),
            initializer=tf.truncated_normal_initializer(stddev=0.1),
        )

        b = tf.get_variable(
            name='b',
            shape=(self.out_channels,),
            initializer=tf.constant_initializer(value=0.1),
        )

        # shape = (batch, seq_len, 1, out_channels)
        features = tf.nn.conv2d(input=input, filter=w, strides=(1, 1, 1, 1), padding='VALID', name='conv2d')
        features = tf.nn.bias_add(features, b, name='bias_add')
        features = tf.nn.relu(features=features, name='relu')

        seq_len = input.shape[1]

        # 只在 seq_len 方向进行宽度为 1 的最大池化.
        # shape = (batch, 1, 1, out_channels)
        features = tf.nn.max_pool(
            value=features,
            ksize=(1, seq_len - self.height + 1, 1, 1),
            strides=(1, 1, 1, 1),
            padding='VALID',
            name='pool'
        )
        return features


class TextCNN(object):
    def __init__(self, seq_len, embedding_size, n_classes, vocab_size, filter_sizes: list, n_filters: int, dropout_keep_prob, l2_reg_lambda=0.0):
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda

        self.text_conv_list = self.init_text_conv()
        self.embeddings = self.init_embeddings()

    def init_embeddings(self):
        with tf.variable_scope('embeddings'):
            embeddings = Embeddings(vocab_size=self.vocab_size, embedding_size=self.embedding_size)
        return embeddings

    def init_text_conv(self):
        kernels = list()
        for filter_size in self.filter_sizes:
            kernel = TextConv(
                height=filter_size,
                width=self.embedding_size,
                in_channels=1,
                out_channels=self.n_filters,
            )
            kernels.append(kernel)
        return kernels

    def build(self):
        x = tf.placeholder(tf.int32, shape=[None, self.seq_len], name="input_x")
        y_true = tf.placeholder(tf.float32, shape=[None, self.n_classes], name="input_y")

        inputs = self.embeddings.embedding_lookup(x)
        inputs = tf.expand_dims(inputs, axis=-1)

        output = self.conv_on_inputs(inputs)
        y_pred, _ = self.ffnn(output)

        model = Model(x, y_true, y_pred)
        return model

    def build2(self):
        x = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.embedding_size, 1], name="input_x")
        y_true = tf.placeholder(tf.float32, shape=[None, self.n_classes], name="input_y")

        o1 = self.conv_on_inputs(x)
        y_pred, _ = self.ffnn(o1)

        model = Model(x, y_true, y_pred)
        return model

    def conv_on_inputs(self, inputs):
        with tf.name_scope('conv_on_input'):
            output_list = list()
            for i, text_conv in enumerate(self.text_conv_list):
                with tf.variable_scope(f'text_conv_{i}'):
                    output = text_conv(inputs)
                    output_list.append(output)
            s = self.n_filters * len(self.filter_sizes)
            feature = tf.concat(output_list, -1, name='concat')
            feature = tf.reshape(feature, shape=(-1, s))
            feature = tf.nn.dropout(feature, keep_prob=self.dropout_keep_prob, name='dropout')
        return feature

    def ffnn(self, inputs):
        in_dims = self.n_filters * len(self.filter_sizes)

        with tf.name_scope('ffnn'):
            w = tf.get_variable(
                name='w',
                shape=(in_dims, self.n_classes),
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )

            b = tf.get_variable(
                name='b',
                shape=(self.n_classes,),
                initializer=tf.constant_initializer(value=0.1)
            )

            l2_loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)

            output = tf.nn.xw_plus_b(inputs, w, b, name='scores')
        return output, l2_loss


def demo1():
    def load_data(n1=10000, n2=10000):
        inputs1 = np.random.randn(n1, 10, 3, 1) + 50
        labels1 = np.array([[1, 0]], dtype=np.float32)
        labels1 = np.tile(labels1, (n1, 1))

        inputs2 = np.random.randn(n2, 10, 3, 1) - 50
        labels2 = np.array([[0, 1]], dtype=np.float32)
        labels2 = np.tile(labels2, (n2, 1))

        inputs = np.concatenate([inputs1, inputs2])
        labels = np.concatenate([labels1, labels2])
        return inputs, labels

    inputs, labels = load_data()

    model = TextCNN(
        seq_len=10,
        embedding_size=3,
        n_classes=2,
        vocab_size=1000,
        filter_sizes=[2, 3],
        n_filters=2,
        dropout_keep_prob=0.9,
        l2_reg_lambda=0.0
    ).build()

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    loss = Model.loss
    accuracy = Model.accuracy

    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

    model.fit(inputs, labels, batch_size=50, epochs=10)
    return


def demo2():
    seq_len = 10

    def load_data(n1=10000, n2=10000):
        inputs1 = np.random.randint(0, 1000, size=(n1, seq_len))
        labels1 = np.array([[1, 0]], dtype=np.float32)
        labels1 = np.tile(labels1, (n1, 1))

        inputs2 = np.random.randint(0, 1000, size=(n2, seq_len))
        labels2 = np.array([[0, 1]], dtype=np.float32)
        labels2 = np.tile(labels2, (n2, 1))

        inputs = np.concatenate([inputs1, inputs2])
        labels = np.concatenate([labels1, labels2])
        return inputs, labels

    inputs, labels = load_data()

    model = TextCNN(
        seq_len=seq_len,
        embedding_size=3,
        n_classes=2,
        vocab_size=1000,
        filter_sizes=[2, 3],
        n_filters=2,
        dropout_keep_prob=0.9,
        l2_reg_lambda=0.0
    ).build()

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    loss = Model.loss
    accuracy = Model.accuracy

    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

    model.fit(inputs, labels, batch_size=50, epochs=10)
    return


def demo3():
    with tf.Session() as sess:
        with tf.variable_scope('test'):
            embeddings = Embeddings(vocab_size=10, embedding_size=3)
        ids = tf.constant(value=[0, 2, 3, 5], dtype=tf.int32)
        output = embeddings.embedding_lookup(ids=ids)
        print(embeddings.embeddings.name)
        sess.run(tf.global_variables_initializer())
        ret = sess.run(output)
        print(ret)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
