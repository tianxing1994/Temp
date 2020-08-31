#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import Counter
import pickle

import numpy as np

from .base import LabelsEncoderBase, TextClassifyBase


class LabelsEncoder(LabelsEncoderBase):
    def __init__(self, label_data_or_pkl):
        super(LabelsEncoder, self).__init__()
        self._label_data_or_pkl = label_data_or_pkl

    @staticmethod
    def _build_labels2id_from_pkl(path: str):
        with open(path, 'rb') as f:
            classes2ids = pickle.load(f)
        return classes2ids

    @staticmethod
    def _build_labels2id_from_data(data: list):
        unique_classes = list(set(data))
        ids = list(np.arange(len(unique_classes)))
        classes2ids = dict(zip(unique_classes, ids))
        return classes2ids

    def _build_labels2ids(self):
        if isinstance(self._label_data_or_pkl, str):
            labels2ids = self._build_labels2id_from_pkl(self._label_data_or_pkl)
        elif isinstance(self._label_data_or_pkl, list):
            labels2ids = self._build_labels2id_from_data(self._label_data_or_pkl)
        else:
            raise NotImplementedError()
        return labels2ids

    def labels_to_ids(self, labels: list):
        result = list()
        for label in labels:
            idx = self.labels2ids.get(label)
            if idx is None:
                raise KeyError(label)
            result.append(idx)
        result = np.array(result)
        return result


class TextClassifyEncoder(TextClassifyBase):
    def __init__(self, vocab_data_or_pkl, label_data_or_pkl,
                 max_len, min_word_freq=1, max_vocab=None,
                 padding='<pad>', out_of_vocab='<oov>'):
        super(TextClassifyEncoder, self).__init__()
        self._vocab_data_or_pkl = vocab_data_or_pkl
        self._label_data_or_pkl = label_data_or_pkl
        self._max_len = max_len
        self._min_word_freq = min_word_freq
        self._max_vocab = max_vocab

        self.padding = padding
        self.out_of_vocab = out_of_vocab

    @staticmethod
    def _build_labels2id_from_pkl(path: str):
        with open(path, 'rb') as f:
            classes2ids = pickle.load(f)
        return classes2ids

    def _build_labels2id_from_data(self, data: list):
        unique_classes = list(set(data))
        unique_classes.append(self.out_of_vocab)
        ids = list(np.arange(len(unique_classes)))
        classes2ids = dict(zip(unique_classes, ids))
        return classes2ids

    def _build_classes2ids(self):
        print(self._label_data_or_pkl)
        if isinstance(self._label_data_or_pkl, str):
            classes2ids = self._build_labels2id_from_pkl(self._label_data_or_pkl)
        elif isinstance(self._label_data_or_pkl, list):
            classes2ids = self._build_labels2id_from_data(self._label_data_or_pkl)
        else:
            raise NotImplementedError()
        return classes2ids

    @staticmethod
    def _build_words2ids_from_pkl(path: str):
        with open(path, 'rb') as f:
            words2ids = pickle.load(f)
        return words2ids

    def _build_words2ids_from_data(self, data: list):
        counter = Counter()
        for sentence in data:
            counter.update(sentence)

        counter = Counter(dict(filter(lambda x: x[1] >= self._min_word_freq, counter.items())))
        if self._max_vocab is not None:
            counter = Counter(dict(counter.most_common(n=self._max_vocab)))
        words = list(counter.keys())
        words.append(self.padding)
        words.append(self.out_of_vocab)

        ids = list(np.arange(len(words)))
        words2ids = dict(zip(words, ids))
        return words2ids

    def _build_words2ids(self):

        if isinstance(self._vocab_data_or_pkl, str):
            words2ids = self._build_words2ids_from_pkl(self._vocab_data_or_pkl)
        elif isinstance(self._vocab_data_or_pkl, list):
            words2ids = self._build_words2ids_from_data(self._vocab_data_or_pkl)
        else:
            raise NotImplementedError()
        return words2ids

    def _pad_or_truncate_sentence(self, sentence: list):
        l = len(sentence)
        if l > self._max_len:
            sentence = sentence[:self._max_len]
        else:
            sentence = sentence + [self.padding] * (self._max_len - l)
        return sentence

    def _sentence_to_id(self, sentence: list):
        ids = list()
        oov_id = self.words2ids[self.out_of_vocab]
        for token in sentence:
            ids.append(self.words2ids.get(token, oov_id))
        return ids

    def sentences_to_ids(self, sentences: list):
        sentences_pad_id = list()
        for sentence in sentences:
            sentence_pad = self._pad_or_truncate_sentence(sentence)
            sentence_pad_id = self._sentence_to_id(sentence_pad)
            sentences_pad_id.append(sentence_pad_id)
        result = np.array(sentences_pad_id)
        return result

    def labels_to_ids(self, labels: list):
        oov_id = self.classes2ids[self.out_of_vocab]
        result = list()
        for cls in labels:
            idx = self.classes2ids.get(cls, oov_id)
            result.append(idx)
        result = np.array(result)
        return result


if __name__ == '__main__':
    pass
