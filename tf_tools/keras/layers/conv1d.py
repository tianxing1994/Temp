#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class Conv1D(object):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None,
                 use_bias=True, kernel_initializer=None, bias_initializer=None, dtype=tf.float32, name='conv1d'):
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding.upper()
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer or tf.truncated_normal_initializer(stddev=0.1)
        self._bias_initializer = bias_initializer or tf.zeros_initializer()

        self._dtype = dtype
        self._name_scope = tf.name_scope(name)

        self._kernel = None
        self._bias = None

    def __call__(self, inputs):
        with self._name_scope:
            assert len(inputs.shape) == 3, 'rank of inputs is expected 3 but {}'.format(
                len(inputs.shape))
            x = inputs
            x = tf.expand_dims(x, axis=-1)

            filter_width = x.shape[-2]._value
            in_channels = x.shape[-1]._value

            if self._kernel is None:
                self._kernel = tf.Variable(
                    self._kernel_initializer(shape=(self._kernel_size, filter_width, in_channels, self._filters)),
                    dtype=self._dtype, name='kernel'
                )

            x = tf.nn.conv2d(
                input=x,
                filters=self._kernel,
                strides=(1, self._strides, filter_width, 1),
                padding=self._padding,
                name='conv1d'
            )
            if self._activation is not None:
                x = self._activation(x)
            x = tf.squeeze(x, axis=-2)
        return x


class Conv1DDeprecated(object):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None,
                 use_bias=True, kernel_initializer=None, bias_initializer=None, dtype=tf.float32, name='conv1d'):
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding.lower()
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer or tf.truncated_normal_initializer(stddev=0.1)
        self._bias_initializer = bias_initializer or tf.zeros_initializer()

        self._dtype = dtype
        self._name_scope = tf.name_scope(name)

        self._kernel = None
        self._bias = None

    def __call__(self, inputs):
        with self._name_scope:
            x = inputs
            dim = x.shape[-1]._value
            if self._kernel is None:
                self._kernel = tf.Variable(
                    self._kernel_initializer(shape=(dim, self._kernel_size * self._filters)),
                    dtype=self._dtype, name='kernel'
                )
            if self._use_bias:
                if self._bias is None:
                    self._bias = tf.Variable(
                        self._bias_initializer(shape=(self._kernel_size * self._filters,)),
                        dtype=self._dtype, name='bias'
                    )

            with tf.name_scope('conv1d'):
                o = tf.matmul(x, self._kernel)
                if self._use_bias:
                    o = tf.add(o, self._bias)
                if self._activation is not None:
                    o = self._activation(o)
                o_list = tf.split(o, num_or_size_splits=self._kernel_size, axis=-1)
                o_pad_l = list()
                for i, o_i in enumerate(o_list):
                    padding = [[0, 0]] * (len(inputs.shape) - 2) + [[i, self._kernel_size - 1 - i]] + [[0, 0]]
                    o_i = tf.pad(
                        tensor=o_i,
                        paddings=tf.constant(padding)
                    )
                    o_pad_l.append(o_i)
                o = tf.stack(o_pad_l, axis=-1)
                o = tf.reduce_sum(o, axis=-1)

            with tf.name_scope('padding'):
                l = o.shape[-2]._value
                range = list(np.arange(l))

                # l = tf.shape(o)[-2]
                # range = tf.range(l)

                if self._padding == 'same':
                    i = self._kernel_size // 2
                    j = (self._kernel_size - 1) % 2
                    indices = range[i: l-i+j: self._strides]
                elif self._padding == 'valid':
                    i = self._kernel_size - 1
                    indices = range[i: l-i: self._strides]
                elif self._padding == 'full':
                    indices = range[::self._strides]
                else:
                    raise NotImplementedError()
                o = tf.gather(params=o, indices=indices, axis=-2, name='padding')
            if self._activation is not None:
                o = self._activation(o)
        return o


class WideConv1D(object):
    """
    paper: https://www.aclweb.org/anthology/P14-1062.pdf
    对 paper 中的宽卷积的实现, (该实现应该是错误的. 论文中应该是普通的 Conv1D, 输出 d 个卷积结果).
    """
    def __init__(self, kernel_size, strides=1, padding='valid', activation=None,
                 use_bias=True, kernel_initializer=None, bias_initializer=None, dtype=tf.float32, name='wide_conv1d'):
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding.lower()
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer or tf.truncated_normal_initializer(stddev=0.1)
        self._bias_initializer = bias_initializer or tf.zeros_initializer()

        self._dtype = dtype
        self._name_scope = tf.name_scope(name)

        self._kernel = None
        self._bias = None

    def __call__(self, inputs):
        with self._name_scope:
            x = inputs
            dim = x.shape[-1]._value
            if self._kernel is None:
                self._kernel = tf.Variable(
                    self._kernel_initializer(shape=(dim, self._kernel_size)),
                    dtype=self._dtype, name='kernel'
                )
            x = tf.expand_dims(x, axis=-1)
            x = tf.multiply(x, self._kernel)

            x_list = tf.split(x, num_or_size_splits=self._kernel_size, axis=-1)
            x_pad_l = list()
            for i, o_i in enumerate(x_list):
                o_i = tf.squeeze(o_i, axis=-1)
                padding = [[0, 0]] * (len(inputs.shape) - 2) + [[i, self._kernel_size - 1 - i]] + [[0, 0]]
                o_i = tf.pad(
                    tensor=o_i,
                    paddings=tf.constant(padding)
                )
                x_pad_l.append(o_i)
            o = tf.stack(x_pad_l, axis=-1)
            o = tf.reduce_sum(o, axis=-1)

            with tf.name_scope('padding'):
                l = tf.shape(o)[-2]
                range = tf.range(l)
                if self._padding == 'same':
                    i = self._kernel_size // 2
                    j = (self._kernel_size - 1) % 2
                    indices = range[i: l-i+j: self._strides]
                elif self._padding == 'valid':
                    i = self._kernel_size - 1
                    indices = range[i: l-i: self._strides]
                elif self._padding == 'full':
                    indices = range[::self._strides]
                else:
                    raise NotImplementedError()
                o = tf.gather(params=o, indices=indices, axis=-2, name='padding')
            if self._activation is not None:
                o = self._activation(0)
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

    conv1d = Conv1D(
        filters=2,
        kernel_size=3,
        strides=1,
        kernel_initializer=tf.constant_initializer(1),
        padding='full'
    )

    outputs = conv1d(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


def demo2():
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

    conv1d = WideConv1D(kernel_size=3, strides=1,
                        kernel_initializer=tf.constant_initializer(1), padding='same')

    outputs = conv1d(inputs)

    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    print(ret.shape)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
