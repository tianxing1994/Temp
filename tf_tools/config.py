#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
pwd = os.path.abspath(os.path.dirname(__file__))


class SummaryConfig(object):
    # tensorboard --logdir outputs
    outputs_path = os.path.join(pwd, 'outputs')


if __name__ == '__main__':
    pass
