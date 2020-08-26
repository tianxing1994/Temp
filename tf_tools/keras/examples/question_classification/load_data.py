#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import pandas as pd

from tf_tools.data.vocabulary import Vocabulary, Classes
from tf_tools.tokenizer.delimiter_tokenizer import DelimiterTokenizer


class Dataset(object):
    """
    Dataset: https://www.kaggle.com/ananthu017/question-classification
    """
    def __init__(self, fpath='./question_classification_dataset.csv'):
        self._fpath = fpath
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self._init_data()
        return self._data

    def _init_data(self):
        data = pd.read_csv(self._fpath)
        data = data.loc[:, ('Questions', 'Category0', 'Category1', 'Category2')]
        return data


class LoadData(object):
    def __init__(self, max_len):
        self._max_len = max_len
        self._data = Dataset().data
        self.vocabulary_obj = Vocabulary(
            max_len=self._max_len,
            tokenizer=DelimiterTokenizer(sep=' ')
        )
        self.classes_obj = Classes()

        self.sentences = None
        self.classes = None
        self._init_training_data()

    def _init_training_data(self):
        data = self._data.loc[:, ('Questions', 'Category0')]
        sentences, classes = list(), list()
        for i, row in data.iterrows():
            sentence = row['Questions']
            category = row['Category0']
            sentences.append(sentence)
            classes.append(category)

        # sentences = self._data['Questions'].tolist()
        self.vocabulary_obj.init_vocab_from_sentences(sentences)
        self.sentences = self.vocabulary_obj.sentences_to_ids(sentences)

        # classes = self._data['Category2'].tolist()
        self.classes_obj.init_from_classes(classes)
        self.classes = self.classes_obj.classes_to_one_hot(classes)

    def save_component(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        config = dict()

        vocab_config = self.vocabulary_obj.get_config()
        words2ids_pkl_path = self.vocabulary_obj.save_component(model_path)
        vocab_config['words2ids_pkl_path'] = words2ids_pkl_path
        config.update(vocab_config)

        classes_config = self.classes_obj.get_config()
        classes2ids_pkl_path = self.classes_obj.save_component(model_path)
        classes_config['classes2ids_pkl_path'] = classes2ids_pkl_path
        config.update(classes_config)

        return config


def demo1():
    load_data = Dataset()
    data = load_data.data

    vocabulary_obj = Vocabulary(
        max_len=15,
        tokenizer=DelimiterTokenizer(sep=' ')
    )

    classes_obj = Classes()

    sentences = data['Questions'].tolist()
    vocabulary_obj.init_vocab_from_sentences(sentences)
    sentences = vocabulary_obj.sentences_to_ids(sentences)
    print(sentences)

    classes = data['Category2'].tolist()
    classes_obj.init_from_classes(classes)
    classes = classes_obj.classes_to_one_hot(classes)
    print(classes)
    return


def demo2():
    load_data = LoadData(max_len=15)

    return


if __name__ == '__main__':
    # demo1()
    demo2()
