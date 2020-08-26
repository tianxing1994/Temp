#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, x=None, y_pred=None):
        self.x = x
        self.y_pred = y_pred

        self.loss = None
        self.metrics = None
        self.optimizer = None

        self.sess = None

    def _init_data_placeholder(self, x):
        if isinstance(x, (list, tuple)):
            raise NotImplemented()
        else:
            x_shape = (None, *x.shape[1:])

        self.x = tf.placeholder(dtype=tf.float32, shape=x_shape, name='x')
        self.y_pred = self.call(self.x)

    def _init_target_placeholder(self, y):
        y_shape = (None, *y.shape[1:])
        self.y_true = tf.placeholder(dtype=tf.float32, shape=y_shape, name='y_true')

    def call(self, *args, **kwargs):
        raise NotImplemented()

    def compile(self, optimizer, loss, metrics=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or list()

    def fit(self, x, y, batch_size=32, epochs=10):
        self._init_target_placeholder(y)
        if self.x is None or self.y_pred is None:
            self._init_data_placeholder(x)

        loss = self.loss(self.y_true, self.y_pred)
        metrics = list()
        for metric_fn in self.metrics:
            try:
                metric = metric_fn.update_state(self.y_true, self.y_pred)
            except AttributeError as e:
                metric = metric_fn(self.y_true, self.y_pred)

            metrics.append(metric)

        train_op = self.optimizer.minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.print_summary()
        for epoch in range(epochs):
            epoch_loss = list()
            epoch_metrics = list()
            for batch_x, batch_y in self.get_next_batch(x, y, batch_size=batch_size):
                _, loss_, metrics_ = self.sess.run(
                    fetches=[train_op, loss, metrics],
                    feed_dict={self.x: batch_x, self.y_true: batch_y}
                )

                epoch_loss.append(loss_)
                epoch_metrics.append(metrics_)

            loss_ = np.mean(np.array(epoch_loss), axis=0)
            metrics_ = np.mean(np.array(epoch_metrics), axis=0)
            print(f'epoch: {epoch}, loss: {loss_}, metrics: {metrics_}')
        return

    def restore(self, save_path):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, save_path)
        return

    def predict(self, inputs):
        y_pred = tf.nn.softmax(self.y_pred, axis=-1)
        output = self.sess.run(y_pred, feed_dict={self.x: inputs})
        return output

    def save(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
        return

    @staticmethod
    def get_next_batch(x, y, batch_size):
        l = len(x)
        idx = np.arange(l)
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]
        steps = l // batch_size

        for step in range(steps):
            b_idx = step * batch_size
            e_idx = b_idx + batch_size
            batch_x = x[b_idx: e_idx]
            batch_y = y[b_idx: e_idx]
            yield batch_x, batch_y

    @staticmethod
    def print_summary():
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for v in variables:
            print(v)


if __name__ == '__main__':
    pass
