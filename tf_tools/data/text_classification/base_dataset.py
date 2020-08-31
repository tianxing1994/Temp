#!/usr/bin/python3
# -*- coding: utf-8 -*-
from tf_tools.data.read_file.csv_data import CSVDataSet


class TwoColumnsCSVDataSet(CSVDataSet):
    def __init__(self, csv_path, sentence_col_name: str, labels_col_name: str):
        super(TwoColumnsCSVDataSet, self).__init__(fpath=csv_path)
        self._csv_path = csv_path
        self._sentence_col_name = sentence_col_name
        self._labels_col_name = labels_col_name

        self._sentences = None
        self._labels = None

    @property
    def sentences(self):
        if self._sentences is None:
            self._sentences = self.data.loc[:, self._sentence_col_name].tolist()
        return self._sentences

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.data.loc[:, self._labels_col_name].tolist()
        return self._labels


if __name__ == '__main__':
    pass
