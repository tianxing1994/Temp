#!/usr/bin/python3
# -*- coding: utf-8 -*-
from .base import BaseTokenizer


class DelimiterTokenizer(BaseTokenizer):
    """其于分隔符 `sep` 执行分词. """
    def __init__(self, sep=' ', name='delimiter_tokenizer'):
        super(DelimiterTokenizer, self).__init__(name)
        self._sep = sep

    def tokenize(self, text: str):
        return text.split(sep=self._sep)

    def get_config(self):
        ret = {
            'name': self.name,
            'sep': self._sep
        }
        return ret
