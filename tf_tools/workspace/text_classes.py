#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.attention.attention_encoder import AttentionEncoder
from tf_tools.induction.attention import Attention
from tf_tools.debug_tools.common import session_run


class SentenceEncoder(object):
    def __init__(self, feature_size, attention_size, seq_len=5, embedding_size=128, heads=8, dropout=0.1, hidden_size=2048, model_dim=512,
                 n_layers=6, share_block=False, name='sentence_encoder', reuse=tf.AUTO_REUSE):
        self.feature_size = feature_size
        self.attention_size = attention_size
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.heads = heads
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.share_block = share_block
        self.name_scope = tf.name_scope(name=f'{name}_op_nodes')
        self.variable_scope = tf.variable_scope(name_or_scope=name, reuse=reuse)
        self.reuse = reuse

        self.attention_encoder = AttentionEncoder(
            seq_len=self.seq_len,
            embedding_size=self.embedding_size,
            heads=self.heads,
            dropout=self.dropout,
            hidden_size=self.hidden_size,
            model_dim=self.model_dim,
            n_layers=self.n_layers,
            share_block=self.share_block
        )

        self.attention = Attention(
            feature_size=self.feature_size,
            attention_size=self.attention_size
        )

    def __call__(self, inputs, mask):
        with self.variable_scope:
            with self.name_scope:
                x = inputs
                x = self.attention_encoder(x, mask)
                x = self.attention(x)
        return x


def parser():
    tf.flags.DEFINE_integer("seq_len", 5, 'sentence max length. ')
    tf.flags.DEFINE_integer("embedding_size", 128, 'word embedding size, will be use for word embedding, position embedding. ')
    tf.flags.DEFINE_integer("heads", 8, 'for multihead attention, NOTE: `embedding_size % heads == 0`')
    tf.flags.DEFINE_float("dropout", 0.1, 'dropout, use for multihead attention and feed forward layers. ')
    tf.flags.DEFINE_integer("hidden_size", 512, 'feed forward layer hidden size (units). ')
    tf.flags.DEFINE_integer("model_dim", 256, 'model dim should equal the `word_embedding_size + position_embedding_size`, in this usage, it should be double `embedding_size`. ')
    tf.flags.DEFINE_integer("n_layers", 2, 'the number of encoder_block.')
    tf.flags.DEFINE_bool("share_block", False, 'flag to identify whether to share the `encoder_block`.')

    tf.flags.DEFINE_integer("feature_size", 256, 'the number of encoder_block.')
    tf.flags.DEFINE_integer("attention_size", 256*4, 'the number of encoder_block.')

    FLAGS = tf.flags.FLAGS
    return FLAGS


def demo1():
    FLAGS = parser()
    x = np.array(
        [[1, 1, 1, 1, 2],
         [1, 1, 1, 1, 2],
         [1, 1, 1, 1, 2],
         [2, 3, 0, 1, 2],
         [1, 0, 2, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0]],
        dtype=np.int
    )

    y = np.array(
        [[1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 0, 0]],
        dtype=np.float
    )

    inputs = tf.placeholder(dtype=tf.int32, shape=(None, 5), name='inputs')
    mask = tf.placeholder(dtype=tf.float32, shape=(None, 5), name='mask')

    sentence_encoder = SentenceEncoder(
        feature_size=FLAGS.feature_size,
        attention_size=FLAGS.attention_size,
        seq_len=FLAGS.seq_len,
        embedding_size=FLAGS.embedding_size,
        heads=FLAGS.heads,
        dropout=FLAGS.dropout,
        hidden_size=FLAGS.hidden_size,
        model_dim=FLAGS.model_dim,
        n_layers=FLAGS.n_layers,
        share_block=FLAGS.share_block
    )
    outputs = sentence_encoder(inputs, mask)
    ret = session_run(outputs, feed_dict={inputs: x, mask: y})
    print(ret.shape)
    return


if __name__ == '__main__':
    demo1()


if __name__ == '__main__':
    pass
