#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.keras import Sequential
from tf_tools.keras.layers.embedding import Embedding
from tf_tools.keras.layers.conv1d import Conv1D
from tf_tools.keras.layers.max_pool import MaxPool1D
from tf_tools.keras.layers.dense import Dense
from tf_tools.keras.layers.flatten import Flatten
from tf_tools.keras.layers.softmax import Softmax

from tf_tools.debug_tools.common import session_run


class CNNTextModel(object):
    def __init__(self, vocab_size, batch_size, seq_len, embedding_size,
                 conv1d_size_list, filters_list, pool_size_list, feed_forward_units_list,
                 n_classes, name='cnn_text_model'):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._conv1d_size_list = conv1d_size_list
        self._filters_list = filters_list
        self._pool_size_list = pool_size_list
        self._feed_forward_units_list = feed_forward_units_list
        self._n_classes = n_classes
        self._name = name

    def build(self):
        layers = [
            Embedding(
                input_dim=self._vocab_size,
                output_dim=self._embedding_size
            ),
        ]

        conv_pool_activation_layers = list()
        for i, (filters, kernel_size, pool_size) in enumerate(zip(self._filters_list, self._conv1d_size_list, self._pool_size_list)):
            sub_layers = [
                Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=1,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    padding='full',
                    name='conv1d_{}_1'.format(i+1)
                ),
                # Conv1D(
                #     filters=filters,
                #     kernel_size=kernel_size,
                #     strides=1,
                #     activation=tf.nn.relu,
                #     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                #     padding='full',
                #     name='conv1d_{}_2'.format(i+1)
                # ),
                # MaxPool1D(
                #     pool_size=pool_size,
                #     strides=pool_size,    # 几倍池化就缩小几倍.
                #     name='max_pool_1d_{}'.format(i+1)
                # ),
            ]
            conv_pool_activation_layers.extend(sub_layers)
        layers.extend(conv_pool_activation_layers)

        feed_forward_layers = list()
        for i, units in enumerate(self._feed_forward_units_list):
            sub_layers = [
                Dense(
                    units=units,
                    use_bias=True,
                    name='dense_{}'.format(i+1)
                )
            ]
            feed_forward_layers.extend(sub_layers)
        layers.extend(feed_forward_layers)

        classify_layers = [
            Flatten(),
            Dense(
                units=self._n_classes,
                use_bias=True,
                name='dense_cls'
            ),
            Softmax(),
        ]
        layers.extend(classify_layers)

        model = Sequential(layers=layers, name=self._name)
        return model


def parser():
    tf.flags.DEFINE_integer("vocab_size", 10, 'vocabulary size. ')

    tf.flags.DEFINE_integer("batch_size", 3, 'batch size. ')
    tf.flags.DEFINE_integer("seq_len", 48, 'sentence max length. ')
    tf.flags.DEFINE_integer("embedding_size", 128, 'word embedding size, will be use for word embedding, position embedding. ')

    tf.flags.DEFINE_integer("n_classes", 10, 'layers')

    # 48, (1 - 1/4)*48=36, (1 - 2/4)*48=24, (1 - 3/4)*48=12,
    tf.flags.DEFINE_string("conv1d_size_list", '3,3,3,3', 'conv1d kernel size of each layers')
    tf.flags.DEFINE_string("filters_list", '32,32,32,32', 'conv1d kernel filters number of each layers, it should be multiple of corresponding `fold size`. ')
    tf.flags.DEFINE_string("pool_size_list", '2,2,2,2', 'fold size of each layers')
    tf.flags.DEFINE_string("feed_forward_units_list", '512,1024,512,256', 'k (k max pool) of each layers. it should less than `seq_len` and decreasing.')

    FLAGS = tf.flags.FLAGS
    FLAGS.conv1d_size_list = list(map(lambda x: int(x), FLAGS.conv1d_size_list.split(',')))
    FLAGS.filters_list = list(map(lambda x: int(x), FLAGS.filters_list.split(',')))
    FLAGS.pool_size_list = list(map(lambda x: int(x), FLAGS.pool_size_list.split(',')))
    FLAGS.feed_forward_units_list = list(map(lambda x: int(x), FLAGS.feed_forward_units_list.split(',')))
    return FLAGS


def demo1():
    FLAGS = parser()

    x = np.random.randint(0, FLAGS.vocab_size, size=(FLAGS.batch_size, FLAGS.seq_len), dtype=np.int)
    inputs = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.seq_len), name='inputs')

    model = CNNTextModel(
        vocab_size=FLAGS.vocab_size,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.seq_len,
        embedding_size=FLAGS.embedding_size,
        conv1d_size_list=FLAGS.conv1d_size_list,
        filters_list=FLAGS.filters_list,
        pool_size_list=FLAGS.pool_size_list,
        feed_forward_units_list=FLAGS.feed_forward_units_list,
        n_classes=FLAGS.n_classes,
        name='cnn_text_model'
    ).build()

    outputs = model(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    if isinstance(ret, list):
        for a in ret:
            print(a.shape)
    else:
        print(ret.shape)
    return


if __name__ == '__main__':
    demo1()
