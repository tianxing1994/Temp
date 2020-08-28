#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


def loss(y_true, y_pred, l2_loss=None, l2_reg_lambda=0.0):
    if l2_loss is None:
        l2_loss = 0.0
    with tf.name_scope('loss'):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        losses = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
    return losses


if __name__ == '__main__':
    pass
