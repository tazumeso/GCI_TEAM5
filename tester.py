#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
import chainer.functions as F


def out_test_result(nn, test_X, test_t):
    @chainer.training.make_extension()
    def test_result(trainer):
        loss = 0
        for i in range(0, len(test_t), 100):
            x = test_X[i : i + 100]
            t = test_t[i : i + 100]
            with chainer.no_backprop_mode():
                y = nn(x)
            l = F.sum(F.sigmoid_cross_entropy(y, t))
            loss += l
        print("test", loss * 100 / len(test_t))
    return test_result
