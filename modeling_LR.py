# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:28:05 2017

@author: MEIP-users
"""

import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LogisticRegression


# 学習データ
df_train = pd.read_csv("train.csv")

#　テストデータ
df_test = pd.read_csv("test.csv")

# 投稿用データ
df_submit = pd.read_csv("sample_submit.csv", header=None)
df_submit.columns = ["data_id","target"]

# 学習に使用する列の指定
X_cols = df_test.columns.tolist()[1:]
y_cols = ["target"]

# 入力と出力に分割
#X = df_train[X_cols].as_matrix().astype("float").T
#y = df_train[y_cols].as_matrix().astype("int")


X1 = df_train[df_train.period=='train1'][X_cols].as_matrix().astype("float").T
y1 = df_train[df_train.period=='train1'][y_cols].as_matrix().astype("int")
X2 = df_train[df_train.period=='train2'][X_cols].as_matrix().astype("float").T
y2 = df_train[df_train.period=='train2'][y_cols].as_matrix().astype("int")
X3 = df_train[df_train.period=='train3'][X_cols].as_matrix().astype("float").T
y3 = df_train[df_train.period=='train3'][y_cols].as_matrix().astype("int")
X4 = df_train[df_train.period=='train4'][X_cols].as_matrix().astype("float").T
y4 = df_train[df_train.period=='train4'][y_cols].as_matrix().astype("int")
X5 = df_train[df_train.period=='train5'][X_cols].as_matrix().astype("float").T
y5 = df_train[df_train.period=='train5'][y_cols].as_matrix().astype("int")
X6 = df_train[df_train.period=='train6'][X_cols].as_matrix().astype("float").T
y6 = df_train[df_train.period=='train6'][y_cols].as_matrix().astype("int")
X7 = df_train[df_train.period=='train7'][X_cols].as_matrix().astype("float").T
y7 = df_train[df_train.period=='train7'][y_cols].as_matrix().astype("int")
X8 = df_train[df_train.period=='train8'][X_cols].as_matrix().astype("float").T
y8 = df_train[df_train.period=='train8'][y_cols].as_matrix().astype("int")
X9 = df_train[df_train.period=='train9'][X_cols].as_matrix().astype("float").T
y9 = df_train[df_train.period=='train9'][y_cols].as_matrix().astype("int")
X10 = df_train[df_train.period=='train10'][X_cols].as_matrix().astype("float").T
y10 = df_train[df_train.period=='train10'][y_cols].as_matrix().astype("int")
X11 = df_train[df_train.period=='train11'][X_cols].as_matrix().astype("float").T
y11 = df_train[df_train.period=='train11'][y_cols].as_matrix().astype("int")
X12 = df_train[df_train.period=='train12'][X_cols].as_matrix().astype("float").T
y12 = df_train[df_train.period=='train12'][y_cols].as_matrix().astype("int")
X13 = df_train[df_train.period=='train13'][X_cols].as_matrix().astype("float").T
y13 = df_train[df_train.period=='train13'][y_cols].as_matrix().astype("int")
X14 = df_train[df_train.period=='train14'][X_cols].as_matrix().astype("float").T
y14 = df_train[df_train.period=='train14'][y_cols].as_matrix().astype("int")


# input dimension
M = 88 
# prior on W
Sigma_w = 100.0 * np.eye(M) 
# num of data points
#N = 560000 
#何回サンプルするか(理想はデータ数だけどメモリが・・・)
N=1000

# inference
# learning rate
alpha = 1.0e-4 
# VI maximum iterations 
#max_iter = 100000 
max_iter = 1000

# learn variational parameters (mu & rho)
#mu, rho = LogisticRegression.VI(y, X, M, Sigma_w, alpha, max_iter)

mu1, rho1 = LogisticRegression.VI(y1, X1, M, Sigma_w, alpha, max_iter)
mu2, rho2 = LogisticRegression.VI(y2, X2, M, Sigma_w, alpha, max_iter)
mu3, rho3 = LogisticRegression.VI(y3, X3, M, Sigma_w, alpha, max_iter)
mu4, rho4 = LogisticRegression.VI(y4, X4, M, Sigma_w, alpha, max_iter)
mu5, rho5 = LogisticRegression.VI(y5, X5, M, Sigma_w, alpha, max_iter)
mu6, rho6 = LogisticRegression.VI(y6, X6, M, Sigma_w, alpha, max_iter)
mu7, rho7 = LogisticRegression.VI(y7, X7, M, Sigma_w, alpha, max_iter)
mu8, rho8 = LogisticRegression.VI(y8, X8, M, Sigma_w, alpha, max_iter)
mu9, rho9 = LogisticRegression.VI(y9, X9, M, Sigma_w, alpha, max_iter)
mu10, rho10 = LogisticRegression.VI(y10, X10, M, Sigma_w, alpha, max_iter)
mu11, rho11 = LogisticRegression.VI(y11, X11, M, Sigma_w, alpha, max_iter)
mu12, rho12 = LogisticRegression.VI(y12, X12, M, Sigma_w, alpha, max_iter)
mu13, rho13 = LogisticRegression.VI(y13, X13, M, Sigma_w, alpha, max_iter)
mu14, rho14 = LogisticRegression.VI(y14, X14, M, Sigma_w, alpha, max_iter)

mu_mean=np.mean([mu1,mu2,mu3,mu4,mu5,mu6,mu7,mu8,mu9,mu10,mu11,mu12,mu13,mu14],axis=0)
sig_mean=np.mean([np.log(1 + np.exp(rho1))**2,np.log(1 + np.exp(rho2))**2,np.log(1 + np.exp(rho3))**2,np.log(1 + np.exp(rho4))**2\
,np.log(1 + np.exp(rho5))**2,np.log(1 + np.exp(rho6))**2,np.log(1 + np.exp(rho7))**2,np.log(1 + np.exp(rho8))**2\
,np.log(1 + np.exp(rho9))**2,np.log(1 + np.exp(rho10))**2,np.log(1 + np.exp(rho11))**2,np.log(1 + np.exp(rho12))**2\
,np.log(1 + np.exp(rho13))**2,np.log(1 + np.exp(rho14))**2],axis=0)

# テストデータで予測
X_test = df_test[X_cols].as_matrix().astype("float").T

z_list = []
#sigma = np.log(1 + np.exp(rho))
    
'''
for n in range(N):
    W=np.random.multivariate_normal(mu.flatten(), np.diag((sigma**2).flatten())).reshape(M,1)
    z_tmp = [LogisticRegression.sigmoid(np.dot(W.T,(X_test[:, i].reshape(M,1)))) for i in range(X_test.shape[1])]
    z_list.append(z_tmp)
'''

for n in range(N):
    W=np.random.multivariate_normal(mu_mean.flatten(), np.diag(sig_mean.flatten())).reshape(M,1)
    z_tmp = [LogisticRegression.sigmoid(np.dot(W.T,(X_test[:, i].reshape(M,1)))) for i in range(X_test.shape[1])]
    z_list.append(z_tmp)


z = np.mean(z_list,axis=0)

# サンプルデータを書換
df_submit["target"] = pd.DataFrame(z)

# CSV出力
df_submit.to_csv('submit.csv', header=None, index=None)
