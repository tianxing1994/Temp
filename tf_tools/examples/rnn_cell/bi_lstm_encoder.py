#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class BiLSTMEncoder(object):
    def __init__(self, hidden_sizes, dropout=0.1, name='bi_lstm_encoder', reuse=tf.AUTO_REUSE):
        self._hidden_sizes = hidden_sizes
        self._dropout = dropout
        self.name_scope = tf.name_scope(name=f'{name}_op_node')
        self.variable_scope = tf.variable_scope(name_or_scope=f'{name}', reuse=reuse)
        self.reuse = reuse

        self.lstm_component = None
        self.init_lstm_component()

    def init_lstm_component(self):
        hidden_cell_list = list()
        for size in self._hidden_sizes:
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(
                    num_units=size,
                    state_is_tuple=True
                ),
                output_keep_prob=1. - self._dropout
            )
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(
                    num_units=size,
                    state_is_tuple=True
                ),
                output_keep_prob=1. - self._dropout
            )
            hidden_cell_list.append((lstm_fw_cell, lstm_bw_cell))
        self.lstm_component = hidden_cell_list

    def __call__(self, inputs):
        with self.variable_scope:
            with self.name_scope:
                x = inputs
                for i, (lstm_fw_cell, lstm_bw_cell) in enumerate(self.lstm_component):
                    o, _ = tf.nn.bidirectional_dynamic_rnn(
                        lstm_fw_cell,
                        lstm_bw_cell,
                        x,
                        dtype=tf.float32,
                        scope=f'bi_lstm_block_{i+1}'
                    )
                    x = tf.concat(o, axis=-1)
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

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 5, 4), name='inputs')

    bi_lstm_encoder = BiLSTMEncoder(
        hidden_sizes=[6, 4, 2],
        dropout=0.2
    )
    outputs = bi_lstm_encoder(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
