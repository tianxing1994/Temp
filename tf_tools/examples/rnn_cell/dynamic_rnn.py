#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


def demo1():
    x = np.array(
        [[[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]],

         [[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=3)

    outputs, state = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=inputs,
        dtype=tf.float32
    )
    y, h = session_run([outputs, state], feed_dict={inputs: x})
    print(y)
    print(h)
    return


def demo2():
    x = np.array(
        [[[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]],

         [[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=3)

    outputs, state = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=inputs,
        dtype=tf.float32
    )
    y, h = session_run([outputs, state], feed_dict={inputs: x})
    print(y)
    print(h)
    return


def demo3():
    x = np.array(
        [[[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]],

         [[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=3)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=3)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_fw_cell,
        cell_bw=lstm_bw_cell,
        inputs=inputs,
        dtype=tf.float32
    )
    fw_bw_ret = session_run([outputs], feed_dict={inputs: x})
    fw, bw = fw_bw_ret[0]
    print(fw)
    print(bw)

    ret = tf.concat(fw_bw_ret[0], axis=-1)
    print(ret)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
