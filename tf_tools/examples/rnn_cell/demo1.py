#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


def demo1():
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
        cell=tf.nn.rnn_cell.LSTMCell(
            num_units=4,
            state_is_tuple=True
        ),
        output_keep_prob=0.9)
    return


def demo2():
    lstm_cell = tf.nn.rnn_cell.LSTMCell(
        num_units=4,
        state_is_tuple=True
    )
    h0 = lstm_cell.zero_state(batch_size=32, dtype=tf.float32)
    print(h0)
    return


