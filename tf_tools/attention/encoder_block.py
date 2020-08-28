#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.attention.multihead_attention import MultiHeadAttention
from tf_tools.attention.layer_normalization import LayerNormalization
from tf_tools.attention.feed_forward import FeedForward
from tf_tools.debug_tools.common import session_run


class EncoderBlock(object):
    def __init__(self, heads=8, dropout=0.1, hidden_size=2048, model_dim=512,
                 name='encoder_block', reuse=tf.AUTO_REUSE):
        """
        :param heads:
        :param dropout:
        :param hidden_size: FeedForward 的隐藏层的维度.
        :param model_dim: 在 Attention 中, 由于 Encoder Block 相串连, 因此建议 model_dim 等于 inputs 的 embedding_size.
        :param name:
        :param reuse:
        """
        self.heads = heads
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.model_dim = model_dim
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

        self.self_attention = MultiHeadAttention(
            heads=self.heads,
            dropout=self.dropout,
            a_mask=False,
            kernel_initializer=tf.truncated_normal_initializer,
            name='self_attention',
            reuse=self.reuse
        )
        self.layer_normalization_1 = LayerNormalization(
            name='layer_normalization_1',
            reuse=self.reuse
        )
        self.feed_forward = FeedForward(
            hidden_size=self.hidden_size,
            model_dim=self.model_dim,
            dropout=self.dropout,
            name='feed_forward',
            reuse=self.reuse
        )
        self.layer_normalization_2 = LayerNormalization(
            name='layer_normalization_2',
            reuse=self.reuse
        )

    def __call__(self, inputs, mask):
        with self.variable_scope:
            with self.name_scope:
                o1 = inputs
                o2 = self.self_attention(o1, o1, o1, q_mask=mask, v_mask=mask)
                o3 = tf.add(o1, o2, name='residual_connect_1')
                o4 = self.layer_normalization_1(o3)
                o5 = self.feed_forward(o4)
                o6 = tf.add(o4, o5, name='residual_connect_1')
                o7 = self.layer_normalization_2(o6)
        return o7


def demo1():
    x = tf.constant(
        value=[[[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]],

               [[2, 2, 2, 2],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
                [2, 2, 2, 2]]],
        dtype=tf.float32
    )

    mask = tf.constant(
        value=[[1, 1, 1, 1, 0],
               [1, 1, 1, 0, 0]],
        dtype=tf.float32)

    encoder_block = EncoderBlock(
        heads=2,
        dropout=0.1,
        ffn_dim=8,
        model_dim=4,
    )

    outputs = encoder_block(inputs=x, mask=mask)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(outputs)
        print(ret)
    return


def demo2():
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

    encoder_block = EncoderBlock(
        heads=2,
        dropout=0.1,
        ffn_dim=8,
        model_dim=4,
    )
    outputs = encoder_block(inputs=inputs, mask=mask)

    ret = session_run(outputs, feed_dict={inputs: x, mask: y})
    print(ret)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
