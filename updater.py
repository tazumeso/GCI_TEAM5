#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
from chainer import Variable
import chainer.functions as F

class MyUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.nn = kwargs.pop('models')
        super(MyUpdater, self).__init__(*args, **kwargs)

    def lossfun(self, nn, y, t):
        batchsize = len(y)
        # 計算してください。
        loss = F.sum(F.sigmoid_cross_entropy(y, t))# / batchsize
        chainer.report({'loss': loss}, nn)
        return loss

    def update_core(self):
        optimizer = self.get_optimizer("nn")

        batch = self.get_iterator("main").next()
        x, t = chainer.dataset.convert.concat_examples(batch)
        x = Variable(x)
        xp = chainer.cuda.get_array_module(x.data)

        nn = self.nn
        y = nn(x)

        optimizer.update(self.lossfun, nn, y, t)
