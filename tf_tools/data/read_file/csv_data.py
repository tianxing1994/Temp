#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd


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
        data = pd.read_csv(self._fpath, encoding='utf-8')
        return data

    def get_data_by_columns(self, columns):
        data = self.data.loc[:, columns]
        return data
