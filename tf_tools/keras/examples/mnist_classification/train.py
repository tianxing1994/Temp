#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tf_tools.data.dataset.mnist import load_data
from tf_tools.keras import Sequential
from tf_tools.keras.layers import Dense, Flatten
from tf_tools.keras.losses import SparseCategoricalCrossentropy
from tf_tools.keras.metrics import SparseCategoricalAccuracy


(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = Sequential(layers=[
    Flatten(input_shape=(28, 28)),
    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
    loss=SparseCategoricalCrossentropy(),
    metrics=[SparseCategoricalAccuracy(), ]
)

model.fit(x_train, y_train, batch_size=32, epochs=5)



