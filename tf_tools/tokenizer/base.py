#!/usr/bin/python3
# -*- coding: utf-8 -*-


class BaseTokenizer(object):
    def __init__(self, name='base_tokenizer'):
        self.name = name

    def tokenize(self, text: str) -> list:
        raise NotImplementedError('tokenize')

    def get_config(self):
        raise NotImplementedError('get_config')


if __name__ == '__main__':
    pass
