#!/usr/bin/python3
# -*- coding: utf-8 -*-
from tf_tools.data.csv_data import CSVDataSet
from tf_tools.data.nlp_encoder import TextClassifyEncoder
from tf_tools.tokenizer.delimiter_tokenizer import DelimiterTokenizer


class LoadData(object):
    def __init__(self, csv_path, sentence_col_name: str, labels_col_name: str,
                 max_len, min_word_freq=1, max_vocab=None,
                 padding='<pad>', out_of_vocab='<oov>'):
        self._csv_path = csv_path
        self._sentence_col_name = sentence_col_name
        self._labels_col_name = labels_col_name

        self._max_len = max_len
        self._min_word_freq = min_word_freq
        self._max_vocab = max_vocab
        self._padding = padding
        self._out_of_vocab = out_of_vocab

        self._data = None
        self._tokenizer = None
        self._tokenize_sentences = None
        self._tokenize_labels = None
        self.init_data()

        self._encoder = None
        self.init_component()

        self._id_sentences = None
        self._id_labels = None

    def init_data(self):
        self._data = CSVDataSet(fpath=self._csv_path).get_data_by_columns(
            columns=(self._sentence_col_name, self._labels_col_name)
        )
        sentences = self._data.loc[:, self._sentence_col_name].tolist()
        labels = self._data.loc[:, self._labels_col_name].tolist()
        self._tokenizer = DelimiterTokenizer(sep=' ')

        new_sentences = list()
        for sentence in sentences:
            w_list = self._tokenizer.tokenize(sentence)
            new_sentences.append(w_list)
        sentences = new_sentences
        self._tokenize_sentences = sentences
        self._tokenize_labels = labels

    def init_component(self):
        self._encoder = TextClassifyEncoder(
            vocab_data_or_pkl=self._tokenize_sentences,
            label_data_or_pkl=self._tokenize_labels,
            max_len=self._max_len,
            min_word_freq=self._min_word_freq,
            max_vocab=self._max_vocab,
            padding=self._padding,
            out_of_vocab=self._out_of_vocab
        )

    @property
    def id_sentences(self):
        if self._id_sentences is None:
            self._id_sentences = self._encoder.sentences_to_ids(
                sentences=self._tokenize_sentences
            )
        return self._id_sentences

    @property
    def id_labels(self):
        if self._id_labels is None:
            self._id_labels = self._encoder.labels_to_ids(
                labels=self._tokenize_labels
            )
        return self._id_labels
