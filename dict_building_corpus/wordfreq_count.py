#!/usr/local/var/pyenv/versions/anaconda3-5.3.1/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : wordfreq_count.py
Description : 统计文件词频，返回词频字典。
Created at  : 2019/12/22
---------------------------------------------------------------------------
"""
__author__ = 'zhang zhiyong'

import warnings
warnings.filterwarnings("ignore")


def word_freq(fpath, n):  # n 记录每次切片的一组中包含的字符数
    f1 = open(fpath, 'rb')
    line = f1.read()
    line = str(line, 'utf-8')
    f1.close()

    strcount = {}
    for i in list(range(len(line) - 1)):
        strcount[line[i:i+n]] = 0

    for i in list(range(len(line) - 1)):
        strcount[line[i:i+n]] += 1

    # 过滤掉包含空格和\n的词
    strcount_cut = {k:v for k,v in strcount.items() if k.find(' ') == -1 and k.find('\n') == -1}

    strcount_sorted = sorted(strcount_cut.items(), key=lambda d:d[1], reverse=True)
    return strcount_sorted


def word_freq_loop(fpath, maxnum):
    wordfreq_list = []
    for i in range(2, maxnum+1):    # 从两个字的词开始
        wordfreq = word_freq(fpath, i)
        wordfreq_list.append(wordfreq)
    return wordfreq_list
