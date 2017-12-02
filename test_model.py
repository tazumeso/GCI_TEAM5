import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, serializers
from net import MyChain

nn = MyChain()
serializers.load_npz('result/nn_epoch_50.npz', nn)
test_X = np.load("test_npy/test_Xxmini.npy")
test_t = np.load("test_npy/test_tmini.npy")
loss = 0
for i in range(0, len(test_t), 100):
    x = test_X[i : i + 100]
    t = test_t[i : i + 100]
    with chainer.no_backprop_mode():
        y = nn(x)
    l = F.sum(F.sigmoid_cross_entropy(y, t))
    print(i, l)
    loss += l
print(loss * 100 / len(test_t))