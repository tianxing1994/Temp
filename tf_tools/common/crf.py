#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.debug_tools.common import session_run


class CRF(object):
    def __init__(self, units, dtype=tf.float32):
        self._units = units
        self._dtype = dtype

        self._kernel = None
        self._bias = None
        self._start = None
        self._end = None
        self._transfer_matrix = None

    def build(self, inputs):
        emb_size = inputs.shape[-1]._value

        self._kernel = tf.get_variable(
            name='kernel',
            shape=(emb_size, self._units),
            dtype=self._dtype
        )
        self._bias = tf.get_variable(
            name='bias',
            shape=(self._units,),
            dtype=self._dtype
        )

        self._start = tf.get_variable(
            name='start',
            shape=(self._units,),
            dtype=self._dtype
        )
        self._end = tf.get_variable(
            name='end',
            shape=(self._units,),
            dtype=self._dtype
        )

        self._transfer_matrix = tf.get_variable(
            name='chain_kernel',
            shape=(self._units, self._units),
            dtype=self._dtype
        )
        return self

    def __call__(self, inputs):
        """
        :param inputs: tensor, shape=(batch_size, seq_len, emb_size).
        :return:
        """
        transmit_score = self.transmit_score(inputs)
        transmit_score = self.add_boundary(transmit_score=transmit_score)

        outputs = self.recursion(transmit_score)
        return outputs

    def recursion(self, transmit_score):
        transfer_matrix = tf.expand_dims(self._transfer_matrix, axis=0)

        initial_inputs = tf.zeros_like(transmit_score[:, :1, :])
        initial_state = [initial_inputs, transfer_matrix]

        outputs, state = self.dynamic_rnn(
            cell=self.crf_step,
            inputs=transmit_score,
            initial_state=initial_state
        )
        return outputs

    def transmit_score(self, inputs):
        _, seq_len, emb_size = inputs.shape
        seq_len = seq_len._value
        emb_size = emb_size._value

        inputs = tf.reshape(inputs, shape=(-1, emb_size))
        transmit_score = tf.nn.xw_plus_b(inputs, self._kernel, self._bias)
        transmit_score = tf.reshape(transmit_score, shape=(-1, seq_len, self._units))
        return transmit_score

    def dynamic_rnn(self, cell, inputs, initial_state: list):
        _, seq_len, _ = inputs.shape
        score_list = tf.split(inputs, num_or_size_splits=seq_len, axis=1)

        state = initial_state
        y_list = list()
        for i, score in enumerate(score_list):
            state.append(i)
            score = tf.squeeze(score, axis=1)
            y_, state = cell(score, state)
            y_ = tf.expand_dims(y_, axis=-2)
            y_list.append(y_)

        outputs = tf.concat(y_list, axis=-2)
        return outputs, state

    def crf_step(self, inputs, state):
        prev_transmit_score, transfer_matrix, _ = state
        inputs = tf.expand_dims(inputs, axis=1)
        prev_inputs = tf.transpose(prev_transmit_score, perm=[0, 2, 1])
        outputs = tf.reduce_logsumexp(
            input_tensor=inputs + prev_inputs + transfer_matrix,
            axis=1,
            keepdims=True
        )
        score = tf.squeeze(outputs, axis=1)
        return score, [inputs, transfer_matrix]

    def add_boundary(self, transmit_score):
        start = tf.expand_dims(tf.expand_dims(self._start, axis=0), axis=0)
        end = tf.expand_dims(tf.expand_dims(self._end, axis=0), axis=0)
        transmit_score = tf.concat(
            values=[transmit_score[:, :1, :] + start, transmit_score[:, 1:, :]],
            axis=1
        )
        transmit_score = tf.concat(
            values=[transmit_score[:, :-1, :], transmit_score[:, -1:, :] + end],
            axis=1
        )
        return transmit_score

    def loss(self, y_true, inputs):
        """
        :param y_true: tensor, shape=(batch_size, seq_len, emb_size)
        :param inputs:
        :return:
        """
        inputs_score = self.transmit_score(inputs)
        inputs_score = self.recursion(inputs_score)
        inputs_score = inputs_score[:, :, -1]
        inputs_score = tf.reduce_logsumexp(input_tensor=inputs_score)

        transmit_score = tf.reduce_sum(y_true * inputs_score)

        y1 = tf.expand_dims(y_true[:, :-1, :], axis=3)
        y2 = tf.expand_dims(y_true[:, 1:, :], axis=2)
        y = y1 * y2
        m = tf.expand_dims(self._transfer_matrix, axis=0)
        m = tf.expand_dims(m, axis=0)
        transfer_score = tf.reduce_sum(y * m)

        loss = inputs_score - (transmit_score + transfer_score)
        return loss

    def viterbi_step(self, inputs, state):
        i = state[-1]
        if i == 0:
            next_idx = tf.argmax(inputs, axis=-1)
            return next_idx, [next_idx],
        else:
            next_idx, i = state
            matrix = tf.gather(params=self._transfer_matrix, indices=next_idx)
            inputs = inputs + matrix
            next_idx = tf.argmax(inputs, axis=-1)
            return next_idx, [next_idx]

    def find_path(self, inputs_score):
        inputs_score = tf.reverse(
            tensor=inputs_score,
            axis=[1]
        )
        paths, _ = self.dynamic_rnn(
            cell=self.viterbi_step,
            inputs=inputs_score,
            initial_state=[]
        )
        paths = tf.reverse(
            tensor=paths,
            axis=[1]
        )
        paths = tf.transpose(paths)
        paths = tf.one_hot(indices=paths, depth=self._units)
        return paths

    def _get_accuracy(self, y_true, y_pred):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        judge = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
        accuracy = tf.reduce_mean(judge)
        return accuracy

    def accuracy(self, y_true, inputs):
        transmit_score = self.transmit_score(inputs)
        inputs_score = self.recursion(transmit_score)
        y_pred = self.find_path(inputs_score)
        accuracy = self._get_accuracy(y_true, y_pred)
        return accuracy


def demo1():
    x = np.array(
        [[[1, 2, 3, 4],
          [4, 3, 2, 1],
          [1, 2, 1, 2],
          [1, 2, 2, 1],
          [2, 3, 3, 2],
          [3, 4, 4, 3]],

         [[1, 2, 3, 4],
          [4, 3, 2, 1],
          [1, 2, 1, 2],
          [1, 2, 2, 1],
          [2, 3, 3, 2],
          [3, 4, 4, 3]]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 6, 4), name='inputs')

    crf = CRF(units=3).build(inputs)
    outputs = crf(inputs)
    ret = session_run(outputs, feed_dict={inputs: x})
    print(ret)
    return


def demo2():
    x = np.array(
        [[[1, 2, 3, 4],
          [4, 3, 2, 1],
          [1, 2, 1, 2],
          [1, 2, 2, 1],
          [2, 3, 3, 2],
          [3, 4, 4, 3]],

         [[1, 2, 3, 4],
          [4, 3, 2, 1],
          [1, 2, 1, 2],
          [1, 2, 2, 1],
          [2, 3, 3, 2],
          [3, 4, 4, 3]]],
        dtype=np.float
    )

    y = np.array(
        [[[0, 1, 0],
          [0, 0, 1],
          [1, 0, 0],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0]],

         [[1, 0, 0],
          [0, 1, 0],
          [1, 0, 0],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0]]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 6, 4), name='inputs')
    targets = tf.placeholder(dtype=tf.float32, shape=(None, 6, 3), name='targets')

    crf = CRF(units=3).build(inputs)
    outputs = crf.loss(y_true=targets, inputs=inputs)
    ret = session_run(outputs, feed_dict={inputs: x, targets: y})
    print(ret)
    return


def demo3():
    x = np.array(
        [[[1, 2, 3, 4],
          [4, 3, 2, 1],
          [1, 2, 1, 2],
          [1, 2, 2, 1],
          [2, 3, 3, 2],
          [3, 4, 4, 3]],

         [[1, 2, 3, 4],
          [4, 3, 2, 1],
          [1, 2, 1, 2],
          [1, 2, 2, 1],
          [2, 3, 3, 2],
          [3, 4, 4, 3]]],
        dtype=np.float
    )

    y = np.array(
        [[[0, 1, 0],
          [0, 0, 1],
          [1, 0, 0],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0]],

         [[1, 0, 0],
          [0, 1, 0],
          [1, 0, 0],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0]]],
        dtype=np.float
    )
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 6, 4), name='inputs')
    targets = tf.placeholder(dtype=tf.float32, shape=(None, 6, 3), name='targets')

    crf = CRF(units=3).build(inputs)
    outputs = crf.accuracy(y_true=targets, inputs=inputs)
    ret = session_run(outputs, feed_dict={inputs: x, targets: y})
    print(ret)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
    # demo3()
