#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class LabelsEncoderBase(object):
    def __init__(self):
        self._labels2ids = None
        self._ids2labels = None
        self._labels_size = None

    def labels_to_ids(self, labels: list) -> np.ndarray:
        raise NotImplementedError('labels_to_ids')

    def _build_labels2ids(self) -> dict:
        raise NotImplementedError('_build_labels2ids')

    @property
    def labels2ids(self) -> dict:
        if self._labels2ids is None:
            self._labels2ids = self._build_labels2ids()
        return self._labels2ids

    @property
    def ids2labels(self) -> dict:
        if self._ids2labels is None:
            self._ids2labels = {v: k for k, v in self.labels2ids.items()}
        return self._ids2labels

    @property
    def labels_size(self):
        if self._labels_size is None:
            self._labels_size = len(self.labels2ids)
        return self._labels_size


class TextClassifyBase(object):
    def __init__(self):
        self._words2ids = None
        self._ids2words = None
        self._vocab_size = None

        self._classes2ids = None
        self._ids2classes = None
        self._class_size = None

    def sentences_to_ids(self, sentences: list) -> np.ndarray:
        raise NotImplementedError('sentences_to_ids')

    def labels_to_ids(self, labels: list) -> np.ndarray:
        raise NotImplementedError('labels_to_ids')

    def _build_words2ids(self) -> dict:
        raise NotImplementedError('_build_words2ids')

    def _build_classes2ids(self) -> dict:
        raise NotImplementedError('_build_classes2ids')

    @property
    def words2ids(self) -> dict:
        if self._words2ids is None:
            self._words2ids = self._build_words2ids()
        return self._words2ids

    @property
    def ids2words(self) -> dict:
        if self._ids2words is None:
            self._ids2words = {v: k for k, v in self._words2ids.items()}
        return self._ids2words

    @property
    def vocab_size(self):
        if self._vocab_size is None:
            self._vocab_size = len(self.words2ids)
        return self._vocab_size

    @property
    def classes2ids(self) -> dict:
        if self._classes2ids is None:
            self._classes2ids = self._build_classes2ids()
        return self._classes2ids

    @property
    def ids2classes(self) -> dict:
        if self._ids2classes is None:
            self._ids2classes = {v: k for k, v in self.classes2ids.items()}
        return self._ids2classes

    @property
    def class_size(self):
        if self._class_size is None:
            self._class_size = len(self.classes2ids)
        return self._class_size


if __name__ == '__main__':
    pass
