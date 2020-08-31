#!/usr/bin/python3
# -*- coding: utf-8 -*-
from bert_utils.extract_feature import BertVector


def demo1():
    bert = BertVector()

    while True:
        question = input('question: ')
        v = bert.encode([question])
        # print(str(v))
        print(type(v))
        print(v.shape)


def demo2():
    bert = BertVector()
    sentence = '你还好吗. '
    v = bert.encode([sentence])
    print(v)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
