#!/usr/bin/env python
# -*- coding;utf-8 -*-

import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L


class MyChain(chainer.Chain):
    def __init__(self, wscale=0.02):
        super(MyChain, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(88, 64, initialW=w)
            self.l1 = L.Linear(64, 32, initialW=w)
            self.l2 = L.Linear(32, 16, initialW=w)
            self.l3 = L.Linear(16, 1, initialW=w)
            self.bn0 = L.BatchNormalization(64)
            self.bn1 = L.BatchNormalization(32)
            self.bn2 = L.BatchNormalization(16)
            
    def __call__(self, x, dr=0.8):
        batch_size = len(x)
        h = F.dropout(F.sigmoid(self.bn0(self.l0(x))), ratio=dr)
        h = F.dropout(F.sigmoid(self.bn1(self.l1(h))), ratio=dr)
        h = F.dropout(F.sigmoid(self.bn2(self.l2(h))), ratio=dr)
        return F.reshape(self.l3(h), (batch_size,))


class MyChain2(chainer.Chain):
    def __init__(self, wscale=0.02):
        super(MyChain2, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(23, 11, initialW=w)
            self.l1 = L.Linear(11, 6, initialW=w)
            self.l2 = L.Linear(6, 1, initialW=w)
            self.bn0 = L.BatchNormalization(11)
            self.bn1 = L.BatchNormalization(6)
            
    def __call__(self, x, dr=0.7):
        batch_size = len(x)
        h = F.dropout(F.relu(self.bn0(self.l0(x))), ratio=dr)
        h = F.dropout(F.relu(self.bn1(self.l1(h))), ratio=dr)
        return F.reshape(self.l2(h), (batch_size,))
