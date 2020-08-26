#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tf_tools.keras.layers.embedding import Embedding


class TestModel(object):
    def __init__(self, vocab_size, batch_size, seq_len, embedding_size,
                 conv1d_size_list, filters_list, pool_size_list, feed_forward_units_list,
                 n_classes, name='cnn_text_model'):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._conv1d_size_list = conv1d_size_list
        self._filters_list = filters_list
        self._pool_size_list = pool_size_list
        self._feed_forward_units_list = feed_forward_units_list
        self._n_classes = n_classes
        self._name = name

    def build(self):
        Embedding(
            input_dim=self._vocab_size,
            output_dim=self._embedding_size
        ),
        return


if __name__ == '__main__':
    pass
