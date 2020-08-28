#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import pandas as pd

from .base import TextClassifyBase


class CSVDataSet(object):
    def __init__(self, fpath):
        self._fpath = fpath
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self._init_data()
        return self._data

    def _init_data(self):
        data = pd.read_csv(self._fpath)
        return data

    def get_data_by_columns(self, columns):
        data = self.data.loc[:, columns]
        return data


class TextClassifyDataBuild(TextClassifyBase):
    def __init__(self, sentences, labels,
                 max_len, min_word_freq=1, max_vocab=None,
                 padding='<pad>', out_of_vocab='<oov>', ):
        """
        sentences 和 labels 一一对应.
        :param sentences: 已分词的 `Token 列表` 的列表.
        :param labels:  cls 类别的列表, 从中编码类别.
        """
        super(TextClassifyDataBuild, self).__init__(sentences, labels)
        self._max_len = max_len
        self._min_word_freq = min_word_freq
        self._max_vocab = max_vocab

        self.padding = padding
        self.out_of_vocab = out_of_vocab

        self.words2ids = None
        self.ids2words = None
        self.vocab_size = None

        self.n_classes = None
        self.classes2ids = None
        self.ids2classes = None

    def _build_sentence_vocab(self, token_sentences):
        counter = Counter()
        for token_sentence in token_sentences:
            counter.update(token_sentence)

        counter = Counter(dict(filter(lambda x: x[1] >= self._min_word_freq, counter.items())))
        if self._max_vocab is not None:
            counter = Counter(dict(counter.most_common(n=self._max_vocab)))
        words = list(counter.keys())
        words.append(self.padding)
        words.append(self.out_of_vocab)

        ids = list(np.arange(len(words)))
        words2ids = dict(zip(words, ids))
        ids2words = dict(zip(ids, words))

        self.words2ids = words2ids
        self.ids2words = ids2words
        self.vocab_size = len(self.words2ids)

    def sentences_to_ids(self, sentences: list):
        sentences_pad_id = list()
        for sentence in sentences:
            sentence_pad = self._pad_or_truncate_sentence(sentence)
            sentence_pad_id = self._sentence_to_id(sentence_pad)
            sentences_pad_id.append(sentence_pad_id)
        result = np.array(sentences_pad_id)
        return result

    def _pad_or_truncate_sentence(self, sentence: list):
        l = len(sentence)
        if l > self._max_len:
            sentence = sentence[:self._max_len]
        else:
            sentence = sentence + [self.padding] * (self._max_len - l)
        return sentence

    def _sentence_to_id(self, sentence: list):
        ids = list()
        oov_id = self.words2ids[self.padding]
        for token in sentence:
            ids.append(self.words2ids.get(token, oov_id))
        return ids

    def _build_class_vocab(self, classes):
        unique_classes = list(set(classes))
        unique_classes.append(self.out_of_vocab)
        self.n_classes = len(unique_classes)

        ids = list(np.arange(self.n_classes))

        self.classes2ids = dict(zip(unique_classes, ids))
        self.ids2classes = dict(zip(ids, unique_classes))

    def classes_to_one_hot(self, classes):
        ids = self.classes_to_ids(classes)
        result = self.one_hot(ids)
        return result

    def classes_to_ids(self, classes):
        oov_id = self.classes2ids[self.out_of_vocab]
        result = list()
        for cls in classes:
            idx = self.classes2ids.get(cls, oov_id)
            result.append(idx)
        return result

    def one_hot(self, idx):
        if isinstance(idx, list):
            result = list()
            for i in idx:
                one_hot = [0] * self.n_classes
                one_hot[i] = 1
                result.append(one_hot)
        else:
            result = [0] * self.n_classes
            result[idx] = 1
        result = np.array(result)
        return result


if __name__ == '__main__':
    pass
