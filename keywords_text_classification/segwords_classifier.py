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
from sklearn.naive_bayes import MultinomialNB
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


def nativebayes_model_train(tfidf_vec_trainX, trainy):
    mnb_tfidf = MultinomialNB()
    mnb_tfidf.fit(tfidf_vec_trainX, trainy)
    return mnb_tfidf


def tfidf_vectorizer1(rst, test_size_ratio):
    # 采用TfidfVectorizer提取文本特征向量
    
    dfrst = pd.DataFrame(list(list(x.values()) for x in rst))
    X, y = dfrst.iloc[:][0], dfrst.iloc[:][1]
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=test_size_ratio, random_state=0)
    # print(trainX, testX, trainy, testy)

    tfidf_vec = TfidfVectorizer()
    tfidf_vec_trainX = tfidf_vec.fit_transform(trainX)
    tfidf_vec_testX = tfidf_vec.transform(testX)

    return tfidf_vec_trainX, tfidf_vec_testX, trainy, testy


def tfidf_vectorizer2(rst1, rst2):
    # 采用TfidfVectorizer提取文本特征向量
    print('tfidf_vectorizer2 start!')

    dfrst1 = pd.DataFrame(list(list(x.values()) for x in rst1))
    trainX, trainy = dfrst1.iloc[:][0], dfrst1.iloc[:][1]

    # dfrst2 = pd.DataFrame(list(list(x.values()) for x in rst2))
    # testX, testy = dfrst2.iloc[:][0], dfrst2.iloc[:][1]

    dfrst2 = pd.DataFrame(list(list(x.values()) for x in rst2))
    testX, seqno = dfrst2.iloc[:][0], dfrst2.iloc[:][1]

    tfidf_vec = TfidfVectorizer(max_df=0.2, min_df=10)
    tfidf_vec_trainX = tfidf_vec.fit_transform(trainX)
    tfidf_vec_testX = tfidf_vec.transform(testX)

    # print(tfidf_vec.get_feature_names())
    # print(tfidf_vec.vocabulary_)
    # print(trainX)
    # print(tfidf_vec_trainX)
    # print(tfidf_vec.get_feature_names())
    # print(tfidf_vec.vocabulary_)
    # print(tfidf_vec_testX)

    # exit()

    # return tfidf_vec_trainX, tfidf_vec_testX, trainy, testy
    return tfidf_vec_trainX, tfidf_vec_testX, trainy, seqno


def tf_vectorizer(rst1, rst2):
    # 采用TfidfVectorizer提取文本特征向量
    print('tf_vectorizer start!')

    dfrst1 = pd.DataFrame(list(list(x.values()) for x in rst1))
    trainX, trainy = dfrst1.iloc[:][0], dfrst1.iloc[:][1]
    dfrst2 = pd.DataFrame(list(list(x.values()) for x in rst2))
    testX, seqno = dfrst2.iloc[:][0], dfrst2.iloc[:][1]

    vec = CountVectorizer()
    vec_trainX = vec.fit_transform(trainX)
    vec_testX = vec.transform(testX)

    return vec_trainX, vec_testX, trainy, seqno


def word2vec_vectorizer(rst1, rst2, embedding_size=1024, in_window=20, in_min_count=5):
    sentences = word2vec.PathLineSentences('./segwords')
    w2vModel = word2vec.Word2Vec(sentences, sg=1, size=embedding_size, window=in_window, 
                                        min_count=in_min_count)
    return w2vModel
    # vector1 = w2vModel.wv['精油']
    # vector2 = w2vModel.wv['祖马龙']
    # print('精油, 祖马龙: ', vector1,  vector2)


def write_predicted_class(tablename, seqno, classid):
    mysql = msc.MyPymysqlPool("dbMysql")
    sql = ''.join(['update ', tablename, " set predict_method='bayes', predicted_classid=", str(classid), " where seqno=", str(seqno)])
    mysql.update(sql)
    mysql.dispose()


def fetch_segwords(tablename):
    mysql = msc.MyPymysqlPool("dbMysql")
    sql = "SELECT t1.keyword_segmented,t2.seqno \
            FROM pzbase.ai_keywords_classification_train t1 \
            inner join pzbase.ai_keywords_classification_classdef t2 \
                    on t1.class_level1=t2.class_level1 and t1.class_level2=t2.class_level2 and t1.class_level3=t2.class_level3 and t2.search_word_flag=1 \
            where t1.proc_flag=1 and t1.keyword not in (select keyword from pzbase.ai_keywords_classification_test where predict_method='bayes')"
    rst1 = mysql.getAll(sql)

    # sql = "SELECT t1.keyword_segmented,t2.seqno \
    #         FROM pzbase.ai_keywords_classification_train t1 \
    #         inner join pzbase.ai_keywords_classification_classdef t2 \
    #                 on t1.class_level1=t2.class_level1 and t1.class_level2=t2.class_level2 and t1.class_level3=t2.class_level3 and t2.search_word_flag=1 \
    #         where t1.proc_flag=1 and t1.keyword in (select keyword from pzbase.ai_keywords_classification_test where predict_method='')"
    # rst2 = mysql.getAll(sql)

    sql = "SELECT keyword_segmented,seqno FROM pzbase.ai_keywords_classification_test where predict_method='bayes'"
    rst2 = mysql.getAll(sql)
    mysql.dispose()

    loginfo = ' %d, %d segmented keywords and classIDs are fetched.' % (len(rst1), len(rst2))
    gl.write_log(logpath, 'info', loginfo)
    return rst1, rst2


def comand_line_set():
    args = argparse.ArgumentParser(description='word segmentation for keywords in train/test table', epilog='')
    # optional parameter
    args.add_argument("-p", type=str, dest="logpath", help="the log path", default='/home/dmer/logs/jd_keywords_classfication/' + LOGFILEPATH)
    args.add_argument("-t", type=str, dest="tablename", help="the train/test table name", default='pzbase.ai_keywords_classification_test')

    args = args.parse_args()
    args_dict = args.__dict__
    return args_dict


if __name__=='__main__':
    global logpath

    args_dict = comand_line_set()
    tablename = args_dict.get("tablename")
    logpath =  args_dict.get("logpath")

    gl.write_log(logpath, 'info', '\n\n')
    loginfo = 'segwords vectorization starting...'
    gl.write_log(logpath, 'info', loginfo)

    # get segmeted keywords
    rst1, rst2 = fetch_segwords(tablename)
    word2vec_vectorizer(rst1, rst2)
    exit()

    # tfidf_vec_trainX, tfidf_vec_testX, trainy, testy = tfidf_vectorizer1(rst1, 0.1)
    # vec_trainX, vec_testX, trainy, seqno_test = tfidf_vectorizer2(rst1, rst2)
    vec_trainX, vec_testX, trainy, seqno_test = tf_vectorizer(rst1, rst2)
    model = nativebayes_model_train(vec_trainX, trainy)

    # joblib.dump((tfidf_vec_trainX, tfidf_vec_testX, trainy, seqno_test), 'vec_data.pkl'.format(), compress=3)
    # joblib.dump(model, 'vec_model.pkl'.format(), compress=3)
    # tfidf_vec_trainX, tfidf_vec_testX, trainy, seqno_test = joblib.load('vec_data.pkl')
    # model = joblib.load('vec_model.pkl')

    # mnb_tfidf_predicty = model.predict(tfidf_vec_testX)
    # print("TfidVectorizer提取的特征学习模型准确率：", model.score(tfidf_vec_testX, testy))
    # print("更加详细的评估指标:\n", classification_report(mnb_tfidf_predicty, testy))


    # 预测并回写预测结果
    mnb_tfidf_predicty = model.predict(vec_testX)
    # print(vec_testX)
    # print(mnb_tfidf_predicty)
    # print(seqno_test)

    i= 0
    for seqno in seqno_test:
        classid = mnb_tfidf_predicty[i]
        write_predicted_class(tablename, seqno, classid)
        if i % 1000 == 0:
            print('finished seqno: ', seqno)
        i +=1


