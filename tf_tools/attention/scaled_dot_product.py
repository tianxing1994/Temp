#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.attention.sequence_mask import SequenceMask
from tf_tools.attention.attention_mask import AttentionMask
from tf_tools.debug_tools.common import session_run


class ScaledDotProduct(object):
    v_standard = 0
    v_neg_inf = 1

    def __init__(self, v_mode=0, a_mask=False,
                 name='scaled_dot_product'):
        self.v_mode = v_mode
        self.a_mask = a_mask
        self.name_scope = tf.name_scope(name=name)

        self.apply_q_mask = SequenceMask(
            v_mode=self.v_mode, axis=-2,
            name='query_sequence_mask'
        )
        self.apply_v_mask = SequenceMask(
            v_mode=self.v_mode, axis=-1,
            name='value_sequence_mask'
        )
        if self.a_mask:
            self.attention_mask = AttentionMask(
                tri_mode=AttentionMask.tri_lower,
                v_mode=self.v_mode
            )

    def __call__(self, q, k, v, q_mask=None, v_mask=None):
        with self.name_scope:
            o1 = tf.matmul(q, k, transpose_b=True, name='dot_product')
            dk = tf.cast(tf.shape(q, name='get_dk')[-1], dtype=tf.float32)
            o2 = tf.div(o1, tf.sqrt(dk), name='scale')
            if q_mask is not None:
                o2 = self.apply_v_mask(o2, v_mask)
            if v_mask is not None:
                o2 = self.apply_q_mask(o2, q_mask)

            if self.a_mask:
                o2 = self.attention_mask(o2)
        return o2


def demo2():
    x = np.array(
        [[[1, 2, 3, 1],
          [3, 2, 1, 3],
          [3, 2, 1, 3],
          [3, 2, 1, 3],
          [4, 3, 2, 4],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]],
        dtype=np.float
    )

    y = np.array([[1, 1, 1, 1, 1, 0, 0]], dtype=np.float)

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 7, 4), name='inputs')
    mask = tf.placeholder(dtype=tf.float32, shape=(None, 7), name='mask')

    scaled_dot_product = ScaledDotProduct(v_mode=ScaledDotProduct.v_standard)

    outputs = scaled_dot_product(inputs, inputs, inputs, mask, mask)
    ret = session_run(outputs, feed_dict={inputs: x, mask: y})
    print(ret)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
