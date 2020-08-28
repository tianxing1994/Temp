#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://arxiv.org/pdf/1607.01759.pdf
https://blog.csdn.net/linchuhai/article/details/86648074
https://codeload.github.com/SophonPlus/ChineseNlpCorpus/zip/master
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


class FastText(object):
    def __init__(self, num_classes, seq_length, vocab_size, embeding_dim, learning_rate,
                 learning_decay_rate, learning_decay_steps, epoch, dropout_keep_prob):
        self._num_classes = num_classes
        self._seq_length = seq_length
        self._vocab_size = vocab_size
        self._embedding_dim = embeding_dim
        self._learning_rate = learning_rate
        self._learning_decay_rate = learning_decay_rate
        self._learning_decay_steps = learning_decay_steps
        self._epoch = epoch
        self._dropout_keep_prob = dropout_keep_prob
        self._input_x = tf.placeholder(tf.int32, (None, self._seq_length), name='input_x')
        self._input_y = tf.placeholder(tf.float32, (None, self._num_classes), name='input_y')

        self._embedding = None
        self._logits = None
        self._loss = None
        self._global_step = None

    def net(self):
        with tf.name_scope('embedding'):
            self._embedding = tf.get_variable('embedding', (self._vocab_size, self._embedding_dim))
            embedding_inputs = tf.nn.embedding_lookup(self._embedding, self._input_x)

        with tf.name_scope('dropout'):
            dropout_output = tf.nn.dropout(embedding_inputs, self._dropout_keep_prob)

        with tf.name_scope('average'):
            mean_sentence = tf.reduce_mean(dropout_output, axis=1)

        with tf.name_scope('score'):
            self._logits = tf.layers.dense(mean_sentence, self._num_classes, name='dense_layer')

        self._loss = tf.losses.sparse_softmax_cross_entropy(logits=self._logits,
                                                            labels=self._input_y)

        self._global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate=self._learning_rate,
                                                   global_step=self._global_step,
                                                   decay_steps=self._learning_decay_steps,
                                                   decay_rate=self._learning_decay_rate,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self._optimizer = slim.learning.create_train_op(total_loss=self._loss,
                                                        optimizer=optimizer,
                                                        update_ops=update_ops)
        return

    def train(self, train_x, train_y, val_x, val_y, batch_size):

        return















if __name__ == '__main__':
    pass
