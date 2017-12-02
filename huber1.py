# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:36:06 2017

@author: MEIP-users
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import linear_model
from sklearn.model_selection import train_test_split

def logloss(pred_y, ans_y):
  n = len(pred_y)
  l1 = ans_y * np.log(pred_y)
  l2 = (1 - ans_y) * np.log(1 - pred_y)
  return -(l1+l2).sum() / n

# 学習データ
df = pd.read_csv("train.csv")
#　テストデータ
df_t = pd.read_csv("test.csv")
"""
X_kesson=['c5', 'c6', 'c7', 'c8', 'c10', 'c13', 'c17', 'c18', 'c19',\
'c23', 'c28', 'c32', 'c34', 'c37', 'c41', 'c42', 'c45', 'c46',\
'c52', 'c54', 'c55', 'c58', 'c60', 'c62', 'c63', 'c65', 'c70',\
'c75', 'c82', 'c84', 'c85', 'c86']

df=df.drop(X_kesson, axis=1)
df_t=df_t.drop(X_kesson,axis=1)
"""
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

X = dfx.as_matrix().astype("float")
y = df[y_cols].as_matrix().astype("int").flatten()

# テストデータで予測
X_test = dfx_t.as_matrix().astype("float")

"""
hub = linear_model.HuberRegressor()
hub.fit(X, y)
y_pred=hub.predict(X_test)
#0.2以下の数値を0.2、0.8以上の数値を0.8
y_pred_1=np.where(y_pred>0.2, y_pred, 0.2)
y_pred_=np.where(y_pred_1<0.8, y_pred_1, 0.8)
"""

lasso = linear_model.Lasso(alpha=0.008)
lasso.fit(X,y)
y_pred=lasso.predict(X_test)



# サンプルデータを書換
df_submit["target"] = pd.DataFrame(y_pred)
# CSV出力
df_submit.to_csv('submit_lasso_2.csv', header=None, index=None)