#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : text_mining_lib.py
"""
import os
import jieba
import re
import glob
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, NUMERIC
from whoosh.qparser import QueryParser

import sys
sys.path.append("/home/dmer/models/pub/")
import general_logging as gl

def load_dicts(path, logpath):
    jieba.load_userdict(path)
    # 动态调整词频，让未登录词的词频自动靠前，这样可以优先匹配
    [jieba.suggest_freq(line.strip(), tune=True) for line in open(path, 'r', encoding='utf8')]

    loginfo = ' User dict %s has beed loaded.' % path
    gl.write_log(logpath, 'info', loginfo)


def get_stopwords(path, logpath):
    stopwords = []
    with open(path, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())

    loginfo = ' Stop words dict %s has beed loaded.' % path
    gl.write_log(logpath, 'info', loginfo)
    return stopwords


def words_segment(sentence, stopwords, goodwords, HMM=True):
    filter_words = None
    if goodwords == 'cn&en':
        filter_words = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z]")

    if HMM:
        cut_words = jieba.cut(sentence, HMM=True)
    else:
        cut_words = jieba.cut(sentence, HMM=False)

    segmented_words_list = []
    for word in cut_words:
        if filter_words:
            word = filter_words.sub("", word)
        if word.strip() != "" and word not in stopwords:
            segmented_words_list.append(word)
    return segmented_words_list


def query_index(words, max_number, index_searcher, query_parser, logpath):
    query_results = []
    for word in words:
        results = index_searcher.search(query_parser.parse(word), limit=max_number)
        query_results.append(results)

    return query_results

# 清理不在词汇表中的词语
def clear_word_from_vocab(word_list, vocab):
    """
    版权声明：本文为CSDN博主「Steven灬」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/weixin_40547993/article/details/89414317
    """
    new_word_list = []
    for word in word_list:
        if word in vocab:
            new_word_list.append(word)
    return new_word_list


def iterate_replacements(sentence, findstr, repstr):
    """
    迭代替换字符串
    :param input_data:
    :return:
    """
    while findstr in sentence:
        sentence = sentence.replace(findstr, repstr)
    return sentence
