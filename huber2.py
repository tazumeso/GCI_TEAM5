# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:45:54 2017

@author: MEIP-users
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import linear_model
from sklearn.model_selection import train_test_split

# 学習データ
df = pd.read_csv("train.csv")
#　テストデータ
df_t = pd.read_csv("test.csv")

X_cols =['c12', 'c80',  'c48', 'c81','c27']


col=88+2
dfx = pd.DataFrame()
for i in range(1, 15):
    dfi = df[df.period == "train{}".format(i)].iloc[:, 2:col]
    dfx = dfx.append((dfi - dfi.mean()) / dfi.std())

dfx_t = (df_t.iloc[:, 1:] - df_t.iloc[:, 1:].mean()) / df_t.iloc[:, 1:].std()

# 投稿用データ
df_submit = pd.read_csv("sample_submit.csv", header=None)
df_submit.columns = ["data_id","target"]

# 学習に使用する列の指定
#X_cols = df_t.columns.tolist()[1:]
y_cols = ["target"]

X = dfx[X_cols].as_matrix().astype("float")
y = df[y_cols].as_matrix().astype("int").flatten()

# テストデータで予測
X_test = dfx_t[X_cols].as_matrix().astype("float")


hub = linear_model.HuberRegressor()
hub.fit(X, y)
y_pred=hub.predict(X_test)
#0.2以下の数値を0.2、0.8以上の数値を0.8
y_pred_1=np.where(y_pred>0.2, y_pred, 0.2)
y_pred_=np.where(y_pred_1<0.8, y_pred_1, 0.8)

"""
classifier = linear_model.LogisticRegression(C=1.0, penalty='l2')
classifier.fit( X, y ) 

y_pred=classifier.predict_proba(X_test)
#0.2以下の数値を0.2、0.8以上の数値を0.8
y_pred_1=np.where(y_pred>0.2, y_pred, 0.2)
y_pred_=np.where(y_pred_1<0.8, y_pred_1, 0.8)
"""

# サンプルデータを書換
df_submit["target"] = pd.DataFrame(y_pred_)
# CSV出力
df_submit.to_csv('submit_huber_5.csv', header=None, index=None)