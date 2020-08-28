#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.attention.attention_mask import AttentionMask
from tf_tools.attention.sequence_mask import SequenceMask
from tf_tools.common.dense import Dense
from tf_tools.debug_tools.common import session_run


class MultiHeadAttention(object):
    def __init__(self, heads, dropout=0.1, a_mask=True, kernel_initializer=None,
                 name='multi_head_attention', reuse=tf.AUTO_REUSE):
        self.heads = heads
        self.dropout = dropout
        self.a_mask = a_mask
        self.kernel_initializer = kernel_initializer
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

        self.apply_q_mask = SequenceMask(v_mode=SequenceMask.v_standard, axis=-2, name='apply_query_mask')
        self.apply_v_mask = SequenceMask(v_mode=SequenceMask.v_neg_inf, axis=-1, name='apply_value_mask')
        if self.a_mask:
            self.attention_mask = AttentionMask(
                tri_mode=AttentionMask.tri_lower,
                v_mode=SequenceMask.v_neg_inf,
                name='attention_mask'
            )

    def __call__(self, q, k, v, q_mask=None, v_mask=None):
        with self.variable_scope:
            with self.name_scope:
                f = lambda x: int(x) if x._value else -1
                sq = tuple(map(f, q.shape))
                sk = tuple(map(f, k.shape))
                sv = tuple(map(f, v.shape))

                dk = int(sk[-1] / self.heads)
                dv = int(sv[-1] / self.heads)

                q = Dense(units=dk * self.heads, use_bias=False, kernel_initializer=self.kernel_initializer, name='query_dense')(q)
                k = Dense(units=dk * self.heads, use_bias=False, kernel_initializer=self.kernel_initializer, name='key_dense')(k)
                v = Dense(units=dv * self.heads, use_bias=False, kernel_initializer=self.kernel_initializer, name='value_dense')(v)
                new_shape = tf.concat(
                    [tf.shape(q)[:-1],
                     tf.constant(value=[self.heads, dk])],
                    axis=0
                )
                qw = tf.reshape(q, shape=new_shape, name='reshape_qw')
                new_shape = tf.concat(
                    [tf.shape(k)[:-1],
                     tf.constant(value=[self.heads, dk])],
                    axis=0
                )
                kw = tf.reshape(k, shape=new_shape, name='reshape_kw')
                new_shape = tf.concat(
                    [tf.shape(v)[:-1],
                     tf.constant(value=[self.heads, dk])],
                    axis=0
                )
                vw = tf.reshape(v, shape=new_shape, name='reshape_vw')
                n = np.arange(len(sq) + 1)
                qw = tf.transpose(qw, perm=(*n[:-3], n[-2], n[-3], n[-1]), name='transpose_qw')
                kw = tf.transpose(kw, perm=(*n[:-3], n[-2], n[-3], n[-1]), name='transpose_kw')
                vw = tf.transpose(vw, perm=(*n[:-3], n[-2], n[-3], n[-1]), name='transpose_vw')
                a = tf.matmul(qw, kw, transpose_b=True, name='attention')
                if v_mask is not None:
                    a = self.apply_v_mask(a, v_mask)
                if self.a_mask:
                    a = self.attention_mask(a)
                a = tf.nn.softmax(a, axis=-1, name='attention_softmax')
                o = tf.matmul(a, vw, name='weighted_value')
                if q_mask is not None:
                    o = self.apply_q_mask(o, q_mask)
                n = np.arange(len(sq) + 1)
                o = tf.transpose(o, perm=(*n[:-3], n[-2], n[-3], n[-1]))
                o = tf.reshape(o, shape=tf.shape(v))
                o = Dense(units=dv * self.heads, use_bias=False, kernel_initializer=self.kernel_initializer, name='output_dense')(o)
                o = tf.nn.dropout(o, keep_prob=1 - self.dropout, name='dropout')
        return o


def demo1():
    x = np.array(
        [[[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]],

         [[2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2]]],
        dtype=np.float
    )

    y = np.array(
        [[1, 1, 1, 1, 0],
         [1, 1, 1, 0, 0]],
        dtype=np.float)

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')
    mask = tf.placeholder(dtype=tf.float32, shape=(None, 5), name='mask')

    mh_attention = MultiHeadAttention(
        heads=2,
        dropout=0.1,
        a_mask=True,
        kernel_initializer=tf.truncated_normal_initializer
    )

    outputs = mh_attention(inputs, inputs, inputs, q_mask=mask, v_mask=mask)
    ret = session_run(outputs, feed_dict={inputs: x, mask: y})
    print(ret)
    return


def demo2():
    x = np.array(
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],
        dtype=np.float
    )
    y = np.array(
        [1, 1, 1, 1, 0],
        dtype=np.float)

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4), name='inputs')
    mask = tf.placeholder(dtype=tf.float32, shape=(None,), name='mask')
    mh_attention = MultiHeadAttention(
        heads=2,
        dropout=0.1,
        a_mask=True,
        kernel_initializer=tf.truncated_normal_initializer
    )
    outputs = mh_attention(inputs, inputs, inputs, q_mask=mask, v_mask=mask)
    ret = session_run(outputs, feed_dict={inputs: x, mask: y})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
