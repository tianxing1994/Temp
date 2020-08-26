#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class MaxPool1D(object):
    # warp
    def __init__(self, pool_size=2, strides=None, padding='valid', **kwargs):
        self._pool_size = pool_size
        self._strides = strides or 1
        self._padding = padding.upper()
        self._name = kwargs.get('name', 'max_pool_1d')
        self._name_scope = tf.name_scope(self._name)

    def __call__(self, inputs):
        with self._name_scope:
            assert len(inputs.shape) == 3, 'rank of inputs is expected 3 but {}'.format(len(inputs.shape))
            x = inputs
            x = tf.expand_dims(x, axis=-1)

            x = tf.nn.max_pool(
                value=x,
                ksize=(1, self._pool_size, 1, 1),
                strides=(1, self._strides, 1, 1),
                padding=self._padding,
                name=self._name
            )
            x = tf.squeeze(x, axis=-1)
        return x


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
    print(x.shape)

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4, 4), name='inputs')
    max_pool_1d = MaxPool1D(
        pool_size=2,
        strides=2
    )
    outputs = max_pool_1d(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    print(ret.shape)
    return


if __name__ == '__main__':
    demo1()
