#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.keras import Sequential

from tf_tools.keras.layers import Softmax
from tf_tools.keras.layers import Dense
from tf_tools.keras.losses.categorical_crossentropy import CategoricalCrossentropy
from tf_tools.keras.metrics.categorical_accuracy import CategoricalAccuracy


def demo1():
    x = np.array(
        [[2.4, 1, 0.5, 0.1],
         [0.2, 0.4, 1, 2.2],
         [2.1, 1, 0.6, 0.2],
         [0.3, 0.5, 1, 2.2],
         [2, 1.2, 0.3, 0.1],
         [0.1, 0.5, 1.2, 2],
         [2, 1.1, 0.7, 0.6],
         [0.3, 0.4, 1, 2.1],
         [2, 1.2, 0.4, 0.2],
         [0.1, 0.3, 1.4, 2],
         [2, 1.5, 0.3, 0.2],
         [0.1, 0.5, 2, 4.1],
         [2, 1.2, 0.9, 0.6],
         [0.3, 0.4, 1, 2.1],
         [2, 1.7, 0.8, 0.1],
         [0.1, 0.3, 1.4, 3],
         [2, 1.5, 0.5, 0.1],
         [0.2, 0.7, 1, 4.1]],
        dtype=np.float
    )

    y = np.array(
        [[1, 0],
         [0, 1],
         [1, 0],
         [0, 1],
         [1, 0],
         [0, 1],
         [1, 0],
         [0, 1],
         [1, 0],
         [0, 1],
         [1, 0],
         [0, 1],
         [1, 0],
         [0, 1],
         [1, 0],
         [0, 1],
         [1, 0],
         [0, 1]],
        dtype=np.float
    )

    model = Sequential([
        Dense(units=4, activation=tf.nn.relu, name='dense1'),
        Dense(units=2, name='dense2'),
        Softmax()
    ])

    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )

    model.fit(x, y, batch_size=3, epochs=100)
    return


if __name__ == '__main__':
    demo1()
