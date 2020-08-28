#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.attention.encoder import Encoder
from tf_tools.bert.embedding import Embedding
from tf_tools.bert.matmul import Matmul
from tf_tools.attention.layer_normalization import LayerNormalization
from tf_tools.common.dense import Dense
from tf_tools.debug_tools.common import session_run


class Bert4MLM(object):
    def __init__(self, vocab_size, embedding_size, dropout, max_len, hidden_size,
                 n_layers, share_block,
                 heads=1, name='bert_for_mlm', reuse=tf.AUTO_REUSE):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.share_block = share_block
        self.heads = heads

        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            max_len=self.max_len,
            dropout=self.dropout,
            hidden_size=self.hidden_size,
            name='bert_embedding'
        )

        self.encoder = Encoder(
            heads=self.heads,
            dropout=self.dropout,
            hidden_size=self.hidden_size * 2,
            model_dim=self.hidden_size,
            n_layers=self.n_layers,
            share_block=self.share_block,
            name='encoder'
        )

        self.feed_forward_1 = Dense(
            units=self.embedding_size,
            activation=tf.nn.relu,
            name='mlm_dense'
        )

        self.layer_normalization = LayerNormalization(name='layer_normalization')

        self.matmul = Matmul(activation=tf.nn.softmax, name='bert_matmul')

    def __call__(self, tokens, segment, mask=None):
        with self.variable_scope:
            with self.name_scope:
                x = tokens
                s = segment
                if mask is None:
                    mask = tf.cast(tf.greater(x, 0), dtype=tf.float32, name='sequence_mask')
                x = self.embedding(x, s)
                x = self.encoder(x, mask)

                x = self.feed_forward_1(x)
                x = self.layer_normalization(x)
                x = self.matmul(
                    a=x,
                    b=self.embedding.token_embedding_lookup.embeddings,
                    transpose_b=True
                )
        return x


def demo1():
    """
    [PAD]: 0
    [CLS]: 1
    [SEP]: 2
    :return:
    """
    x = np.array(
        [[1, 8, 26, 15, 28, 21, 4, 10, 4, 9, 4, 0, 0, 0, 0, 0],
         [1, 87, 6, 53, 81, 45, 4, 10, 4, 6, 0, 0, 0, 0, 0, 0],
         [1, 2, 61, 22, 78, 45, 4, 10, 4, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int
    )
    y = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int
    )

    # shape=(batch_size, seq_len). seq_len 最大值由 max_len 限定,
    # 但此处可能是为了兼容多种 Bert 模型的下级应用, 没有将 seq_len 写死.
    tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name='input_tokens')
    segment = tf.placeholder(dtype=tf.int32, shape=(None, None), name='input_segment')

    bert4mlm = Bert4MLM(
        vocab_size=100,
        embedding_size=64,
        max_len=64,
        dropout=0.1,
        hidden_size=128,
        n_layers=6,
        share_block=True,
        heads=2,
        name='bert_for_mlm'
    )

    outputs = bert4mlm(tokens, segment)
    ret = session_run(outputs, feed_dict={tokens: x, segment: y})

    print(ret)
    print(ret.shape)
    print(type(ret))
    return


def demo2():
    """
    [PAD]: 0
    [CLS]: 1
    [SEP]: 2
    :return:
    """
    x = np.array(
        [1, 8, 26, 15, 28, 21, 4, 10, 4, 9, 4, 0, 0, 0, 0, 0],
        dtype=np.int
    )
    y = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=np.int
    )

    tokens = tf.placeholder(dtype=tf.int32, shape=(None,), name='input_tokens')
    segment = tf.placeholder(dtype=tf.int32, shape=(None,), name='input_segment')

    bert4mlm = Bert4MLM(
        vocab_size=100,
        embedding_size=64,
        max_len=64,
        dropout=0.1,
        hidden_size=128,
        n_layers=6,
        share_block=True,
        name='bert_for_mlm'
    )

    outputs = bert4mlm(tokens, segment)
    ret = session_run(outputs, feed_dict={tokens: x, segment: y})
    # print(ret)
    # print(ret.shape)

    x = ret[:, 0]
    print(x)
    print(len(x))
    print(type(x))
    return


if __name__ == '__main__':
    demo1()
    # demo2()
