# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:47:48 2017

@author: MEIP-users
"""

import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LogisticRegression


# 学習データ
df = pd.read_csv("train.csv")
#　テストデータ
df_t = pd.read_csv("test.csv")

X_cols = ['c12','c80','c48','c81']


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

X = dfx[df.period!='train8'][X_cols].as_matrix().astype("float").T
y = df[df.period!='train8'][y_cols].as_matrix().astype("int")

# テストデータで予測
X_test = dfx_t[X_cols].as_matrix().astype("float").T


# input dimension
M = 4
# prior on W
Sigma_w = 100.0 * np.eye(M) 
# num of data points
N=100

# inference
# learning rate
alpha = 1.0e-4 
# VI maximum iterations 
max_iter = 1000

mu, rho = LogisticRegression.VI(y, X, M, Sigma_w, alpha, max_iter)

sigma = np.log(1 + np.exp(rho))
z=0
for n in range(N):
    W=np.random.multivariate_normal(mu.flatten(), np.diag((sigma**2).flatten())).reshape(M,1)
    z_tmp = np.array([LogisticRegression.sigmoid(np.dot(W.T,(X_test[:, i].reshape(M,1)))) for i in range(X_test.shape[1])])
    z=(z*n+z_tmp)/(n+1)
    
#0.2以下の数値を0.2、0.8以上の数値を0.8
z_1=np.where(z>0.35, z, 0.35)
z_=np.where(z_1<0.65, z_1, 0.65)    

# サンプルデータを書換
df_submit["target"] = pd.DataFrame(z_)

# CSV出力
df_submit.to_csv('submit_bayes.csv', header=None, index=None)

#0.2以下の要素数
z[z < 0.35].shape
z[z > 0.65].shape