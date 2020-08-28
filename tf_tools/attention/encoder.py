#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.attention.encoder_block import EncoderBlock
from tf_tools.debug_tools.common import session_run


class Encoder(object):
    def __init__(self, heads=8, dropout=0.1, hidden_size=2048, model_dim=512,
                 n_layers=6, share_block=False,
                 name='encoder', reuse=tf.AUTO_REUSE):
        self.heads = heads
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.share_block = share_block
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

    def __call__(self, inputs, mask):
        with self.variable_scope:
            with self.name_scope:
                x = inputs
                encoder_block = None
                for i in range(self.n_layers):
                    if self.share_block is False or encoder_block is None:
                        name = 'encoder_block' \
                            if self.share_block \
                            else f'encoder_block_{i+1}'

                        encoder_block = EncoderBlock(
                            heads=self.heads,
                            dropout=self.dropout,
                            hidden_size=self.hidden_size,
                            model_dim=self.model_dim,
                            name=name,
                        )
                    x = encoder_block(inputs=x, mask=mask)
        return x


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
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')
    mask = tf.placeholder(dtype=tf.float32, shape=(None, 5), name='mask')

    encoder = Encoder(
        heads=2,
        dropout=0.1,
        hidden_size=8,
        model_dim=4,
        n_layers=6,
        share_block=True
    )

    outputs = encoder(inputs=inputs, mask=mask)
    ret = session_run(outputs, feed_dict={inputs: x, mask: y})
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

    encoder = Encoder(
        heads=2,
        dropout=0.1,
        hidden_size=8,
        model_dim=4,
        n_layers=6,
        share_block=False
    )

    outputs = encoder(inputs=inputs, mask=mask)
    ret = session_run(outputs, feed_dict={inputs: x, mask: y})
    print(ret)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
