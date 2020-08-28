#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class TextClassifyBase(object):
    def __init__(self, sentences, labels):
        self._sentences = sentences
        self._labels = labels
        self._id_sentences = None
        self._id_labels = None

    def init_from_data(self, sentences: list, labels: list):
        raise NotImplementedError('init_from_data')

    def init_from_pkl(self, path: str):
        raise NotImplementedError('init_from_pkl')

    @property
    def id_sentences(self):
        if self._id_sentences is None:
            self._id_sentences = self.sentences_to_ids(self._sentences)
        return self._id_sentences

    def sentences_to_ids(self, sentences: list) -> np.ndarray:
        raise NotImplementedError('sentences_to_ids')

    @property
    def id_labels(self):
        if self._id_labels is None:
            self._id_labels = self.labels_to_ids(self._labels)
        return self._id_labels

    def labels_to_ids(self, labels: list) -> np.ndarray:
        raise NotImplementedError('labels_to_ids')



if __name__ == '__main__':
    pass
