#!/usr/local/var/pyenv/versions/anaconda3-5.3.1/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : project_main.py
Description : 数学词典构建。
              1. 利用已有字典，将字典中的词在数学题目中进行检索，统计词频
              2. 利用词频进行分词
Created at  : 2019/12/22
---------------------------------------------------------------------------
"""
__author__ = 'zhang zhiyong'

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../pub/lib/")
import mysql_conn as msc

import file_content_filter as fc
import wordfreq_count as wc
import dict_build as db

# 只保留中文和运算符
# CUT_WORDS = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
CUT_WORDS = '[a-zA-Z0-9’!"#$&\',．＿①②③④⑤()()（）./:：;；?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

DATAPATH = '/Users/zhangzhiyong/Desktop/puxue_project/dataset/questions_proc'
DICTPATH = '/Users/zhangzhiyong/Desktop/puxue_project/question_database_building/pub/dict'
BASEDICT = 'BigDict-master/367w.dict.utf-8'
DESTDICT = '/dict_puxue_subject.txt'
QUESTION_FILE = 'question_stems_subject_cn.txt'
SUBJECT = 'math'
MAX_CHARNUM = 10

tablename_question_stem = 'staging.question_stem'


def read_question_stem(tablename):
    mysql = msc.MyPymysqlPool("dbMysql")
    sql = ''.join(['select distinct question_stem from ', tablename, ])     # 过滤重复题目
    rstData = mysql.getAll(sql)
    mysql.dispose()

    if rstData:
        list_rstData = list(list(x.values())[0] for x in rstData)
        return list_rstData


if __name__ == '__main__':

    fpath_cn = DATAPATH + '/' + QUESTION_FILE.replace('subject', SUBJECT)

    # 读取题干，过滤停用词
    question_stems_list = read_question_stem(tablename_question_stem)
    fc.word_filter(question_stems_list, fpath_cn, CUT_WORDS)

    # 按照字符数，对题干文件进行词频统计
    wordfreq_list = wc.word_freq_loop(fpath_cn, MAX_CHARNUM)

    # 将词频字典与词库字典匹配，输出有效词频
    fpath_dict_base = DICTPATH + '/' + BASEDICT
    fpath_dict_wordfreq = DICTPATH + '/' + DESTDICT.replace('subject', SUBJECT)
    db.dict_build(fpath_dict_base, fpath_dict_wordfreq, wordfreq_list, MAX_CHARNUM)
