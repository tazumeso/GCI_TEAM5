import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

dfx = pd.DataFrame()
for i in range(1, 15):
    dfi = train[train.period == "train{}".format(i)].iloc[:, 2:90]
    dfx = dfx.append((dfi - dfi.mean()) / dfi.std())
test_X = (test.iloc[:, 1:] - test.iloc[:, 1:].mean()) / test.iloc[:, 1:].std()
train_X = dfx

#sample_submission = pd.read_csv("./data/sample_submit.csv", header=None)

# use former 6 periods as train and the latter 6 as valid
train_X = train.values[:, 2:90].astype("float32")
train_y = train.target.values.astype("int32")

test_X = test.values[:, 1:89].astype("float32")

pred_test = np.zeros((test_X.shape[0], 10))

# run 10 times using different seed

for i in range(10):
    # shuffle
    np.random.seed(50+i)
    perm = np.arange(train_X.shape[0])
    np.random.shuffle(perm)
    train_X = train_X[perm]
    train_y = train_y[perm]
    # model
    # 88-128-1
    # add strong regularizer
    model = Sequential()
    model.add(Dense(128, input_shape=(88, ), activation='relu'))
    model.add(Dense(1, kernel_regularizer=keras.regularizers.l1_l2(0.025), bias_initializer=keras.initializers.Zeros(), kernel_initializer=keras.initializers.Zeros()))
    model.add(Activation('sigmoid'))
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])

    model.fit(train_X, train_y, epochs=10, batch_size=64, verbose=1)
    pred_test[:, i] = model.predict(test_X).reshape(test_X.shape[0],)

# output
pd.DataFrame({"id":test.data_id.values, "p":pred_test.mean(axis=1)}).to_csv("model_1104_3.csv", index=False, header=False)
