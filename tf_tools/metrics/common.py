#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


def accuracy(y_true, y_pred):
    with tf.name_scope('accuracy'):
        y_pred = tf.nn.softmax(logits=y_pred, axis=-1, name='softmax')
        y_pred = tf.argmax(y_pred, axis=-1, name='argmax')
        y_true = tf.argmax(y_true, axis=-1)
        correct = tf.equal(y_pred, y_true)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
    return accuracy


if __name__ == '__main__':
    pass
