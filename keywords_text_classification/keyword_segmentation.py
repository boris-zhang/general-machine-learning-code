#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : keyword_segmentation.py
Description : 对广告文案分词，支持多关键词精确检索和语义拓展查询。
              1. 从关键词表中读入文案标题
              2. 使用Jieba进行分词，使用了Jieba内置字典，用户字典，停用词字典
              3. 将分词结果写入关键词表
Created at  : 2019/03/29
---------------------------------------------------------------------------
"""
__author__ = 'zhang zhiyong'

import os
import pandas as pd
import datetime
import argparse
import jieba
import re

import sys
sys.path.append("/home/dmer/models/pub/")
import mysql_conn as msc
import text_mining_lib as tml
import general_logging as gl


# only accept Chinese and English characters
GOOD_WORDS = 'en&cn'
CUT_WORDS = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
LOGFILEPATH = 'keyword_segmentation_%s.log' % datetime.datetime.now().strftime('%Y-%m-%d')


def write_segmented_words(tablename, seqno, words):
    mysql = msc.MyPymysqlPool("dbMysql")
    sql = ''.join(['update ', tablename, " set proc_flag=1, keyword_segmented='", str(words), "' where seqno=", str(seqno)])
    # sql = ''.join(['update ', tablename, " set proc_flag=1, keyword_segmented_cutall='", str(words), "' where seqno=", str(seqno)])
    mysql.update(sql)
    mysql.dispose()


def segment_title(titles, tablename):
    i = 0
    for title in titles:
        seqno = title["seqno"]
        keyword = title["keyword"]
        segmented_words = tml.words_segment(keyword, stopwords, GOOD_WORDS, iscutall=False)
        # segmented_words = tml.words_segment(keyword, stopwords, GOOD_WORDS, iscutall=True)
        words = re.sub(CUT_WORDS, '', str(segmented_words))
        write_segmented_words(tablename, seqno, words)
        if i % 1000 == 0:
            loginfo = ' progress status: %d ' % i
            gl.write_log(logpath, 'info', loginfo)
        i +=1
    loginfo = ' Total %d keyword\'s segmented words have been writen.' % i
    gl.write_log(logpath, 'info', loginfo)


def get_titles(tablename):
    mysql = msc.MyPymysqlPool("dbMysql")
    # upper chars for general match, Boss=BOSS=boss
    # sql = "SELECT seqno, upper(keyword) as keyword FROM %s where proc_flag=0 order by seqno" % tablename
    sql = "SELECT seqno, upper(keyword) as keyword FROM %s order by seqno" % tablename
    rst = mysql.getAll(sql)     # [{'seqno': 97, 'keyword': '小说看到一半要花钱？这里让你免费看完大结局！'}, {'seqno': 98, 'keyword': '玄幻大神天蚕土豆亲授逆袭攻略'}]
    mysql.dispose()

    loginfo = ' %d titles are fetched.' % len(rst)
    gl.write_log(logpath, 'info', loginfo)
    return rst


def comand_line_set():
    args = argparse.ArgumentParser(description='word segmentation for keywords in train/test table', epilog='')
    # optional parameter
    args.add_argument("-p", type=str, dest="logpath", help="the log path", default='/home/dmer/logs/jd_keywords_classfication/' + LOGFILEPATH)
    args.add_argument("-t", type=str, dest="tablename", help="the train/test table name", default='pzbase.ai_keywords_classification_test2')
    args.add_argument("-d", type=str, dest="userdictpath", help="the path of user dictionary file", default='/home/dmer/models/ad_title_retrieval/dict/ik_taobao_dict300000_nojieba.txt')
    args.add_argument("-k", type=str, dest="compdictpath", help="the path of comp dictionary file", default='/home/dmer/models/ad_title_retrieval/dict/comp_dict.txt')
    args.add_argument("-s", type=str, dest="stopwordpath", help="the path of stop words file", default='/home/dmer/models/ad_title_retrieval/dict/stop_words.txt')

    args = args.parse_args()
    args_dict = args.__dict__
    return args_dict


if __name__=='__main__':
    global logpath

    args_dict = comand_line_set()
    tablename = args_dict.get("tablename")
    logpath =  args_dict.get("logpath")
    user_dict_path = args_dict.get("userdictpath")
    comp_dict_path = args_dict.get("compdictpath")
    stop_word_path = args_dict.get("stopwordpath")

    gl.write_log(logpath, 'info', '\n\n')
    loginfo = 'keyword segmentation starting...'
    gl.write_log(logpath, 'info', loginfo)

    # preload dicts to save running time
    tml.load_dicts(user_dict_path, logpath)
    tml.load_dicts(comp_dict_path, logpath)
    stopwords = tml.get_stopwords(stop_word_path, logpath)
 
    titles = get_titles(tablename)
    segment_title(titles, tablename)
