
#!/usr/local/var/pyenv/versions/anaconda3-5.3.1/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : dict_build.py
Description : 词典构建。
Created at  : 2019/12/22
---------------------------------------------------------------------------
"""
__author__ = 'zhang zhiyong'

import warnings
warnings.filterwarnings("ignore")


def dict_build(fpath_basedict, fpath_dict_wordfreq, wordfreq_sets, num):

    basewords_dict = {}
    with open(fpath_basedict, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            line = line.replace('\n', '').strip(' ')
            word = line.split(' ')[0]
            wordparts = line.split(' ')[-1]
            if not len(word) > num:
                basewords_dict[word] = wordparts

    with open(fpath_dict_wordfreq, 'w') as f2:
        basewords = basewords_dict.keys()
        for wordfreq_list in wordfreq_sets:
            for wordfreq in wordfreq_list:
                word = wordfreq[0]
                freq = wordfreq[1]
                if word in basewords:
                    f2.writelines(word + ' ' + str(freq) + ' ' + basewords_dict[word] + '\n')
