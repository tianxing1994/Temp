#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import struct

import numpy as np
import matplotlib.pyplot as plt

from config import project_path

pwd = os.path.abspath(os.path.dirname(__file__))


class DataSet(object):
    def __init__(self, path=None):
        self._path = path or os.path.join(pwd, 'data')

        self._train_images = None
        self._train_labels = None
        self._train_num_examples = None

        self._test_images = None
        self._test_labels = None
        self._test_num_examples = None

    @property
    def train_images(self):
        if self._train_images is None:
            self._init_train_data()
        return self._train_images

    @property
    def train_labels(self):
        if self._train_labels is None:
            self._init_train_data()
        return self._train_labels

    @property
    def train_num_examples(self):
        if self._train_num_examples is None:
            self._init_train_data()
        return self._train_num_examples

    @property
    def test_images(self):
        if self._test_images is None:
            self._init_test_data()
        return self._test_images

    @property
    def test_labels(self):
        if self._test_labels is None:
            self._init_test_data()
        return self._test_labels

    @property
    def test_num_examples(self):
        if self._test_num_examples is None:
            self._init_test_data()
        return self._test_num_examples

    def _init_train_data(self):
        images, images_num, image_shape = self._init_images(
            path=os.path.join(self._path, 'train-images.idx3-ubyte')
        )
        labels, labels_num = self._init_labels(
            path=os.path.join(self._path, 'train-labels.idx1-ubyte')
        )
        self._train_images = images
        self._train_labels = labels
        self._train_num_examples = labels_num

    def _init_test_data(self):
        images, images_num, image_shape = self._init_images(
            path=os.path.join(self._path, 'train-images.idx3-ubyte')
        )
        labels, labels_num = self._init_labels(
            path=os.path.join(self._path, 'train-labels.idx1-ubyte')
        )
        self._test_images = images
        self._test_labels = labels
        self._test_num_examples = labels_num

    @staticmethod
    def _init_images(path):
        with open(path, 'rb') as imgpath:
            images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols)
        image_shape = (rows, cols)
        return images, images_num, image_shape

    @staticmethod
    def _init_labels(path):
        with open(path, 'rb') as lbpath:
            labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        return labels, labels_num


def load_data(path=None):
    mnist = DataSet(path=path)
    x_train = mnist.train_images
    y_train = mnist.train_labels

    x_test = mnist.test_images
    y_test = mnist.test_labels
    return (x_train, y_train), (x_test, y_test)


def demo1():
    path = os.path.join(project_path, 'tf_tools/data/dataset/mnist/data')
    mnist = DataSet(path=path)
    # mnist = DataSet()
    print(mnist.train_images.shape)
    print(mnist.train_labels.shape)
    print(mnist.train_num_examples)

    print(mnist.test_images.shape)
    print(mnist.test_labels.shape)
    print(mnist.test_num_examples)
    return


if __name__ == '__main__':
    demo1()
