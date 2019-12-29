#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : segwords_classifier.py
Description : 为分词结果建立矢量化特征文件。
              1. 从关键词表中读入分词结果
              2. 使用scikit-learn提供的类完成向量化，
                 得到训练集和测试集两个文本的特征矩阵，矩阵类型为稀疏矩阵
              3. 根据矩阵进行分类
Created at  : 2019/03/29
---------------------------------------------------------------------------
"""
__author__ = 'zhang zhiyong'

import os
import pandas as pd
import numpy as np
import datetime
import argparse

import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import  roc_auc_score, classification_report
from sklearn.externals import joblib

from gensim.models import word2vec

import sys
sys.path.append("/home/dmer/models/pub/")
import mysql_conn as msc
import text_mining_lib as tml
import general_logging as gl


# only accept Chinese and English characters
LOGFILEPATH = 'segwords_vectorizer%s.log' % datetime.datetime.now().strftime('%Y-%m-%d')


def fetch_segwords(tablename):
    mysql = msc.MyPymysqlPool("dbMysql")
    sql = "SELECT t1.keyword_segmented,t2.seqno \
            FROM pzbase.ai_keywords_classification_train2 t1 \
            inner join pzbase.ai_keywords_classification_classdef t2 \
                    on t1.class_level1=t2.class_level1 and t1.class_level2=t2.class_level2 and t1.class_level3=t2.class_level3 and t2.valid_flag=1"
    rst1 = mysql.getAll(sql)

    sql = "SELECT keyword_segmented,seqno FROM pzbase.ai_keywords_classification_test2_gnb where length(keyword)!=char_length(keyword) and prediction_method is null"
    rst2 = mysql.getAll(sql)
    mysql.dispose()
    return rst1, rst2


def nativebayes_model_train(vec_trainX, trainy):
    mnb_tfidf = GaussianNB()
    mnb_tfidf.fit(vec_trainX, trainy)
    return mnb_tfidf


def write_predicted_class(tablename, seqno, classid, score):
    mysql = msc.MyPymysqlPool("dbMysql")
    sql = ''.join(['update ', tablename, " set prediction_method='bayes', prediction_score=", str(score), ", prediction_classid=", str(classid), " where seqno=", str(seqno)])
    mysql.update(sql)
    mysql.dispose()


if __name__=='__main__':

    print('the 1st step start!')
    rst1, rst2 = fetch_segwords('pzbase.ai_keywords_classification_test2_gnb')

    sentences = word2vec.PathLineSentences('./segwords_train2')
    w2vModel = word2vec.Word2Vec(sentences, sg=1, size=512, window=15, min_count=5, iter=5)

    dfrst1 = pd.DataFrame(list(list(x.values()) for x in rst1))
    trainX, trainy = dfrst1.iloc[:][0], dfrst1.iloc[:][1]

    from pandas import Series

    print('the 2nd step start!')
    similar_trainX = []
    for segword in trainX:
        similar_segword=[]
        for kw in segword.split():
            similar_segword.append(kw)
            try:
                simliar_kw = w2vModel.most_similar(kw, topn=15)
            except:
                pass
            else:
                for item in simliar_kw:
                    if item[1]>=0.8:
                        similar_segword.append(item[0])
        similar_trainX.append(" ".join(similar_segword))
    similar_trainX = Series(similar_trainX)

    print('the 3rd step start!')
    dfrst2 = pd.DataFrame(list(list(x.values()) for x in rst2))
    testX, testX_seqno = dfrst2.iloc[:][0], dfrst2.iloc[:][1]

    similar_testX = []
    for segword in testX:
        similar_segword=[]
        for kw in segword.split():
            similar_segword.append(kw)
            try:
                simliar_kw = w2vModel.most_similar(kw, topn=15)
            except:
                pass
            else:
                for item in simliar_kw:
                    if item[1]>=0.8:
                        similar_segword.append(item[0])
        similar_testX.append(" ".join(similar_segword))
    similar_testX = Series(similar_testX)


    vec = CountVectorizer()
    # vec = TfidfVectorizer()
    vec_trainX = vec.fit_transform(similar_trainX)
    vec_testX = vec.transform(similar_testX)

    print('the 4th step start!')
    model = nativebayes_model_train(vec_trainX.toarray(), trainy)
    mnb_predict_y = model.predict(vec_testX.toarray())
    mnb_predict_proba_y = model.predict_proba(vec_testX.toarray())


    print('the 5th step start!')
    i= 0
    for seqno in testX_seqno:
        pred_classid = mnb_predict_y[i]
        pred_score = np.max(mnb_predict_proba_y[i])
        write_predicted_class('pzbase.ai_keywords_classification_test2_gnb', seqno, pred_classid, pred_score)
        if i % 1000 == 0:
            print('finished seqno: ', seqno)
        i +=1