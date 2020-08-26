#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.keras import Sequential
from tf_tools.keras.layers.conv1d import Conv1D, WideConv1D
from tf_tools.keras.layers.k_maxpool import FoldKMaxPooling


from tf_tools.debug_tools.common import session_run


def parser():
    tf.flags.DEFINE_integer("batch_size", 1, 'batch size. ')

    tf.flags.DEFINE_integer("seq_len", 12, 'sentence max length. ')
    tf.flags.DEFINE_integer("embedding_size", 4, 'word embedding size, will be use for word embedding, position embedding. ')

    tf.flags.DEFINE_integer("filters", 4, 'let it equal to `embedding_size`')
    tf.flags.DEFINE_integer("kernel_size", 3, 'conv1d kernel size. ')
    tf.flags.DEFINE_integer("strides", 1, 'conv1d strides. ')

    FLAGS = tf.flags.FLAGS
    return FLAGS


def demo1():
    """
    paper: https://www.aclweb.org/anthology/P14-1062.pdf
    """
    FLAGS = parser()
    x = np.array(
        [[[2, 1, 0, 0],
          [0, 0, 1, 2],
          [2, 1, 0, 0],
          [0, 0, 1, 2],
          [2, 1, 0, 0],
          [0, 0, 1, 2],
          [2, 1, 0, 0],
          [0, 0, 1, 2],
          [2, 1, 0, 0],
          [0, 0, 1, 2],
          [2, 1, 0, 0],
          [0, 0, 1, 2]]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 12, 4), name='inputs')

    model = Sequential([
        WideConv1D(
            kernel_size=FLAGS.kernel_size,
            strides=FLAGS.strides,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            padding='full',
            name='wide_conv1d_1'
        ),
        FoldKMaxPooling(
            batch_size=FLAGS.batch_size,
            k=9,
            fold_size=1,
            name='fold_k_max_pool_1'
        ),
        Conv1D(
            filters=4,
            kernel_size=FLAGS.kernel_size,
            strides=FLAGS.strides,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            padding='full',
            name='conv1d_1'
        ),
        WideConv1D(
            kernel_size=FLAGS.kernel_size,
            strides=FLAGS.strides,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            padding='full',
            name='wide_conv1d_2'
        ),
        FoldKMaxPooling(
            batch_size=FLAGS.batch_size,
            k=5,
            fold_size=2,
            name='fold_k_max_pool_2'
        ),
        Conv1D(
            filters=2,
            kernel_size=FLAGS.kernel_size,
            strides=FLAGS.strides,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            padding='full',
            name='conv1d_2'
        ),
        WideConv1D(
            kernel_size=FLAGS.kernel_size,
            strides=FLAGS.strides,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            padding='full',
            name='wide_conv1d_3'
        ),
        FoldKMaxPooling(
            batch_size=FLAGS.batch_size,
            k=3,
            fold_size=2,
            name='fold_k_max_pool_3'
        ),
        Conv1D(
            filters=1,
            kernel_size=FLAGS.kernel_size,
            strides=FLAGS.strides,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            padding='full',
            name='conv1d_3'
        ),
    ])

    outputs = model.call(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    print(ret.shape)
    return


if __name__ == '__main__':
    demo1()
