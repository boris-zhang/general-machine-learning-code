#!/usr/local/var/pyenv/versions/anaconda3-5.3.1/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : file_content_filter.py
Description : 过滤文本文件中无效字符，以便进行词频统计。
              1. 过滤内容：标点符号、英文字母
              2. 内容替换为空格，减少词频统计复杂度
Created at  : 2019/12/22
---------------------------------------------------------------------------
"""
__author__ = 'zhang zhiyong'

import warnings
warnings.filterwarnings("ignore")
import re


def iterate_replacements(sentence, findstr, repstr):
    """
    迭代替换字符串
    :param input_data:
    :return:
    """
    while findstr in sentence:
        sentence = sentence.replace(findstr, repstr)
    return sentence


def iterate_replacements2(sentence, findstr, word):
    if findstr in sentence:
        for x in iterate_replacements(sentence.replace(findstr, word, 1)): yield x
    else:
        yield sentence


def word_filter(sentences, fpath, cutwords):

    with open(fpath, 'w') as f:
        for sentence in sentences:
            line_cut = re.sub(cutwords, ' ', sentence)
            line_cut = line_cut.replace('\xa0', ' ')
            line_cut = line_cut.replace('\t', ' ')
            line_cut = line_cut.replace('\u3000', ' ')
            words = iterate_replacements(line_cut, '  ', ' ')
            f.writelines(words.strip() + '\n')
