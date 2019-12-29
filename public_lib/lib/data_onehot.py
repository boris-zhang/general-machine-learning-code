#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: data_process.py
Description: 数据处理函数包
Variables: None
Author: zhangzhiyong
Change Activity: First coding on 2018/5/8
---------------------------------------------------------------------------
'''

import sys
import datetime
import warnings
import itertools

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler


def splitComa(s):
    if type(s) == str:
        s = s.split(',')
        return list(x.strip() for x in s)
    else:
        return []

def proc_tuple(s):
    '''将[(a, b), (c, d)]处理为[a_b, c_d]形式'''
    return list(x[0] + '_' + x[1] for x in s)

def proc_space(s):
    '''去除list元素中的多余空格'''
    return list(x.strip() for x in s)

def comb_cols(col):
    result_lst = []
    if type(col) == str:
        cols_lst = col.split('|')
        cols_num = len(cols_lst)
        for col in range(cols_num):
            if col == 0:
                result_lst = []
            if col == 1:
                tmp1 = cols_lst[col - 1].split(',')
                tmp2 = cols_lst[col].split(',')
                a = list(itertools.product(proc_space(tmp1), proc_space(tmp2)))
                result_lst = proc_tuple(a)
            if col > 1:
                tmp3 = cols_lst[col].split(',')
                a = list(itertools.product(result_lst, proc_space(tmp3)))
                result_lst = proc_tuple(a)
        return result_lst
    else:
        return []


def oneEncoding(df, unohfeats=[], ohfeats=[], mulohfeats=[], combohfeats=[], stype='non'):
    feat_idx = []
    enc = LabelBinarizer(sparse_output=False)  # 字符串型类别变量只能用LabelBinarizer()
    cn = 0

    # 先拼接onehot字段
    for i, feat in enumerate(ohfeats):
        x_train = enc.fit_transform(df.iloc[:, feat].values.reshape(-1, 1))
        if i == 0:
            X_train = x_train
        else:
            # X_train = sparse.hstack((X_train, x_train))
            X_train = np.hstack((X_train, x_train))

        # 拼接索引标签
        ec = list(enc.classes_)
        if len(ec) == 1:
            feat_idx.append('%d:%s %d' % (feat, ec, cn))
            cn += 1
        elif len(ec) == 2:
            ec = ec[0]
            feat_idx.append('%d:%s %d' % (feat, ec, cn))
            cn += 1
        else:
            for j in range(len(ec)):
                feat_idx.append('%d:%s %d' % (feat, ec[j], cn))
                cn += 1
        # print('X_train: ', X_train.shape[1])
        # print('cn:      ', cn)

    # 拼接非onehot字段
    for i, feat in enumerate(unohfeats):
        x1 = df.iloc[:, feat].values.reshape(-1, 1)

        if stype == 'non':
            x_train = x1
        elif stype == 'std':
            scaler = StandardScaler().fit(x1)
            x_train = scaler.transform(x1)
        elif stype == 'mm':
            scaler = MaxAbsScaler().fit(x1)
            x_train = scaler.transform(x1)
        X_train = np.hstack((X_train, x_train))

        feat_idx.append('%d:%s %d' % (feat, 'unohfeat', cn))
        cn += 1

    print('		%d onehot encoding concat: done!' % len(feat_idx))
    return X_train, feat_idx