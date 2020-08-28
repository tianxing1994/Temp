#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class AttentionMask(object):
    # lower triangle mask
    tri_lower = 0
    # upper triangle mask
    tri_upper = 1

    v_standard = 0
    v_neg_inf = 1

    infinitesimal = 1e12

    def __init__(self, tri_mode=0, v_mode=0, name='attention_mask'):
        self.tri_mode = tri_mode
        self.v_mode = v_mode
        self.name_scope = tf.name_scope(name=name)

    def _get_mask(self, inputs):
        ndim = len(inputs.shape)
        diag_vals = tf.ones(shape=tf.shape(inputs)[-2:], dtype=tf.float32)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        if self.tri_mode == self.tri_lower:
            tril = tril
        else:
            tril = tf.transpose(a=tril)
        n = ndim - 2
        for _ in range(n):
            tril = tf.expand_dims(tril, axis=0)
        multiples = tf.concat(
            [tf.shape(inputs)[:n], tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        mask = tf.tile(
            input=tril,
            multiples=multiples
        )
        return mask

    def __call__(self, inputs):
        with self.name_scope:
            mask = self._get_mask(inputs)

            if self.v_mode == self.v_standard:
                result = inputs * mask
            else:
                result = inputs - (1 - mask) * self.infinitesimal
        return result

def demo1():
    x = tf.constant(
        value=[[[1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7]]],
        dtype=tf.float32)

    am = AttentionMask()
    outputs = am(x)

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

    # am = AttentionMask(tri_mode=AttentionMask.tri_lower)
    # am = AttentionMask(tri_mode=AttentionMask.tri_upper)
    am = AttentionMask(tri_mode=AttentionMask.tri_upper, v_mode=AttentionMask.v_neg_inf)

    outputs = am(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(outputs)
        print(ret)
    return


def demo3():
    x = np.array(
        [[[1, 2, 3, 4],
          [2, 3, 4, 5],
          [3, 4, 5, 6],
          [4, 5, 6, 7]]],
        dtype=np.float)

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4, 4), name='inputs')

    # am = AttentionMask(tri_mode=AttentionMask.tri_lower)
    # am = AttentionMask(tri_mode=AttentionMask.tri_upper)
    am = AttentionMask(tri_mode=AttentionMask.tri_upper, v_mode=AttentionMask.v_neg_inf)

    outputs = am(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    # demo2()
    demo3()
