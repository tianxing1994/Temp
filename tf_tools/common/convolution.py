#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tf_tools.debug_tools.common import session_run


class Conv1D(object):
    same = 'same'
    valid = 'valid'

    def __init__(self, filters, kernel_size, padding='valid', kernel_initializer=None, name='conv1d', reuse=tf.AUTO_REUSE):
        # stride = 1. (完全实现, 有点难度, 不搞了).
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding.lower()
        self.kernel_initializer = kernel_initializer

        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, inputs):
        with self.variable_scope:
            with self.name_scope:
                dim = inputs.shape[-1]
                kernel = tf.get_variable(
                    name='kernel',
                    shape=(dim, self.kernel_size * self.filters),
                    dtype=tf.float32,
                    initializer=self.kernel_initializer
                )
                x = tf.reshape(inputs, shape=(-1, dim))
                o = tf.matmul(x, kernel)
                new_shape = tf.concat(
                    values=[
                        tf.shape(inputs)[:-1],
                        tf.constant([self.filters, self.kernel_size], shape=(2,))
                    ],
                    axis=0
                )
                o = tf.reshape(o, shape=new_shape)
                o_split_list = tf.split(o, num_or_size_splits=self.kernel_size, axis=-1)

                o_split_padded_list = list()
                for i, o_split in enumerate(o_split_list):
                    paddings = [[0, 0]] * (len(inputs.shape) - 2) + [[i, self.kernel_size - 1 - i]] + [[0, 0]] * 2
                    o_split = tf.pad(
                        tensor=o_split,
                        paddings=tf.constant(paddings)
                    )
                    o_split_padded_list.append(o_split)
                o = tf.concat(o_split_padded_list, axis=-1)
                o = tf.reduce_sum(o, axis=-1)

                if self.padding == self.valid:
                    idx = self.kernel_size - 1
                    o = o[:, idx: - idx]
                elif self.padding == self.same:
                    idx = (self.kernel_size - 1) // 2
                    parity = (self.kernel_size - 1) % 2
                    if parity == 0:
                        o = o[:, idx: - idx]
                    else:
                        if idx == 0:
                            o = o[:, idx + 1:]
                        else:
                            o = o[:, idx + 1: - idx]
                else:
                    raise NotImplemented()
        return o


def demo1():
    x = np.array(
        [[[0, 1, 2, 3],
          [0, 0, 1, 2],
          [0, 1, 0, 1],
          [0, 2, 1, 0]],

         [[0, 1, 0, 3],
          [1, 0, 0, 2],
          [2, 1, 0, 1],
          [3, 2, 0, 0]],

         [[0, 1, 2, 3],
          [1, 0, 1, 2],
          [0, 0, 0, 0],
          [3, 2, 1, 0]]],
        np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4, 4), name='inputs')

    # conv1d = Conv1D(filters=2, kernel_size=2, padding='valid', kernel_initializer=tf.constant_initializer(1))
    # conv1d = Conv1D(filters=2, kernel_size=2, padding='same', kernel_initializer=tf.constant_initializer(1))
    # conv1d = Conv1D(filters=2, kernel_size=3, padding='valid', kernel_initializer=tf.constant_initializer(1))
    conv1d = Conv1D(filters=3, kernel_size=3, padding='same', kernel_initializer=tf.constant_initializer(1))

    outputs = conv1d(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    print(ret.shape)
    return


if __name__ == '__main__':
    demo1()
