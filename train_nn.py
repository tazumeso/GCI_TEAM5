#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda
import chainer.functions as F

from net import MyChain, MyChain2
from updater import MyUpdater

import glob
import numpy as np

from tester import out_test_result 


def load_train(gpu=-1):
    if gpu >= 0:
        xp = cuda.cupy
    else:
        xp = np
    #train_data = xp.load("test_npy/train_Xxmini.npy").astype(xp.float32)
    train_data = xp.load("GCI_TEAM5/train_Xindex.npy").astype(xp.float32)
    train_label = xp.load("test_npy/train_tmini.npy").reshape(len(train_data)).astype(xp.int8)
    return chainer.datasets.tuple_dataset.TupleDataset(train_data, train_label)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=1,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    #nn = MyChain()
    nn = MyChain2()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        nn.to_gpu()

    #optimizer = chainer.optimizers.Adam(alpha=0.0004, beta1=0.5)
    optimizer = chainer.optimizers.SGD(lr=0.005)
    #optimizer = chainer.optimizers.AdaGrad()
    optimizer.setup(nn)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')

    train = load_train(args.gpu)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    
    #test_X = np.load("test_npy/test_Xxmini.npy").astype(np.float32)
    test_X = np.load("GCI_TEAM5/test_Xindex.npy").astype(np.float32)
    test_t = np.load("test_npy/test_tmini.npy").astype(np.int8)

    updater = MyUpdater(
        models=nn,
        iterator=train_iter,
        optimizer={'nn': optimizer},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'epoch')
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        nn, 'nn_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'nn/loss'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_test_result(
            nn, test_X, test_t),
        trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
    

if __name__ == '__main__':
    main()
