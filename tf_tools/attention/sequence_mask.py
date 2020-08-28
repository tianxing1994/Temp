#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class SequenceMask(object):
    v_standard = 0
    v_neg_inf = 1

    def __init__(self, v_mode=0, axis=None, name='sequence_mask'):
        self.v_mode = v_mode
        self.axis = axis if axis else -2
        # 目前只支持 1 头. 因为我还搞不懂到底按哪种实现方法算是标准的.
        self.heads = 1
        self.name_scope = tf.name_scope(name=name)

    def __call__(self, x, mask):
        with self.name_scope:
            axis = self.axis
            if self.axis < 0:
                axis = len(x.shape) + self.axis
            n = axis - len(mask.shape) + 1
            for _ in range(n):
                mask = tf.expand_dims(mask, axis=-2)
            n = len(x.shape) - len(mask.shape)
            for _ in range(n):
                mask = tf.expand_dims(mask, axis=len(mask.shape))
            if self.v_mode == self.v_standard:
                result = x * mask
            else:
                result = x - (1 - mask) * 1e12
        return result


def demo1():
    x = tf.constant(
        value=[[[1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6]],

               [[6, 5, 4],
                [5, 4, 3],
                [4, 3, 2],
                [3, 2, 1]]],
        dtype=tf.float32)

    mask = tf.constant(
        value=[[1, 1, 1, 0],
               [1, 1, 0, 0]],
        dtype=tf.float32)

    sm = SequenceMask()
    outputs = sm(x, mask)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(outputs)
        print(ret)
    return


def demo2():
    x = tf.constant(
        value=[[[1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7]]],
        dtype=tf.float32)

    mask = tf.constant(
        value=[[1, 1, 1, 0]],
        dtype=tf.float32)

    sm = SequenceMask()
    outputs = sm(x, mask)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(outputs)
        print(ret)
    return


def demo3():
    x = tf.constant(
        value=[[[1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7]]],
        dtype=tf.float32)

    mask = tf.constant(
        value=[[1, 1, 1, 0]],
        dtype=tf.float32)

    sm = SequenceMask(axis=-2)
    outputs = sm(x, mask)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(outputs)
        print(ret)
    return


def demo4():
    x = tf.constant(
        value=[[[[1, 2, 3, 4],
                 [2, 3, 4, 5],
                 [3, 4, 5, 6],
                 [4, 5, 6, 7]]]],
        dtype=tf.float32)

    mask = tf.constant(
        value=[[1, 1, 1, 0]],
        dtype=tf.float32)

    sm = SequenceMask(axis=-1)
    outputs = sm(x, mask)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(outputs)
        print(ret)
    return


def demo5():
    x = np.array(
        [[[[1, 2, 3, 4],
           [2, 3, 4, 5],
           [3, 4, 5, 6],
           [4, 5, 6, 7]]]],
        dtype=np.float)

    y = np.array(
        [[1, 1, 1, 0]],
        dtype=np.float)

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 1, 4, 4), name='inputs')
    mask = tf.placeholder(dtype=tf.float32, shape=(None, 4), name='mask')

    sm = SequenceMask(axis=-1)
    outputs = sm(inputs, mask)
    ret = session_run(outputs, feed_dict={inputs: x, mask: y})
    print(ret)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    demo5()
