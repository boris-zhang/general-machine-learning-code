#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : gdt_adcopt_lib.py
Description : 创意优选公共服务
Author      : Guo Jiahao
Created at  : 2018/12/11
---------------------------------------------------------------------------
"""

import re
import hashlib
import warnings
import pandas as pd
import sys
import mysql_conn as msc
import general_logging as gl
warnings.filterwarnings("ignore")


def md5_convert(string):
    '''
    计算哈希值
    :param string: 字符串
    :return: 哈希值
    '''
    if string == '':
        return ''
    m = hashlib.md5()
    m.update(string.encode())
    return m.hexdigest()


def analysis_creative(src):
    '''
    寻找创意中出现的图片、视频和文案。
    :param src: 创意元素
    :return: 图片列表，视频列表，文案列表。
    '''
    # find images
    image_pat = '"image[0-9]*":"[0-9a-z:]+"'
    image_results = re.findall(image_pat, src, re.I)
    
    # find videos
    video_pat = '"video[0-9]*":"[0-9a-z:]+"'
    video_results = re.findall(video_pat, src, re.I)
    
    # find titles(开启非贪婪模式，'?'放在'+'后边，要求匹配越少越好。)
    title_pat = '"title[0-9]*":".+?"'
    title_results = re.findall(title_pat, src, re.I)
    
    return image_results, video_results, title_results


def combine_results(images, videos, titles):
    '''
    获取素材id和文案
    :param images: 图片列表
    :param videos: 视频列表
    :param titles: 文案列表
    :return: 图片和视频id，文案。
    '''
    img_and_vid_ls = []
    if len(images) > 0:
        for image in images:
            img_and_vid_ls.append(image.split(':')[-1].replace('"', ''))
    
    if len(videos) > 0:
        for video in videos:
            img_and_vid_ls.append(video.split(':')[-1].replace('"', ''))
    
    ttl_ls = []
    if len(titles) > 0:
        for title in titles:
            ttl_ls.append(title.split(':')[-1].replace('"', ''))
    
    img_and_vid_ls.sort()
    ttl_ls.sort()
    
    return '-'.join(img_and_vid_ls), '-'.join(ttl_ls)


def calculate_quanitle(string):
    '''
    计算ctr列表的分位数
    :param string:
    :return:
    '''
    ctr_list = string.split(',')
    ctr_list = [float(i) * 100000000 for i in ctr_list]
    ctr_series = pd.Series(ctr_list).astype('int')
    q_1 = ctr_series.quantile(0.25) / 100000000
    q_2 = ctr_series.quantile(0.75) / 100000000
    
    return q_1, q_2


def gen_quanitle(logpath, source_table, target_table, dt):
    mysql = msc.MyPymysqlPool("dbMysql")
    sql = "select hour,ind_first_industry_id,site_set,promoted_object_type,group_concat(ctr) as ctrs from {0} \
           where ind_first_industry_id is not null and site_set is not null and promoted_object_type is not null \
           and dt>='{1}' group by hour,ind_first_industry_name,site_set,promoted_object_type;".format(source_table, dt)
    try:
        rstData = mysql.getAll(sql)
    except BaseException as e:
        logerror = 'Error occurred when get data from {0} for ctr_level:{1}'.format(table, e)
        print(logerror)
        # gl.write_log(logpath, 'error', logerror)
        return False
    finally:
        mysql.dispose()
    
    if not rstData:
        logerror = 'Get nothing from {0} for ctr_level'.format(table)
        print(logerror)
        # gl.write_log(logpath, 'error', logerror)
        return False
    
    dataSet = pd.DataFrame(list(rstData))
    
    quanitle_lists = []
    for i in range(dataSet.shape[0]):
        hour = str(dataSet.iloc[i]['hour'])
        indst_id = str(dataSet.iloc[i]['ind_first_industry_id'])
        site = str(dataSet.iloc[i]['site_set'])
        prd = str(dataSet.iloc[i]['promoted_object_type'])
        ctr_q1, ctr_q2 = calculate_quanitle(str(dataSet.iloc[i]['ctrs']))
        quanitle_list = [hour, site, prd, indst_id, ctr_q1, ctr_q2]
        quanitle_lists.append(quanitle_list)
    
    # 插入前清空
    mysql = msc.MyPymysqlPool("dbMysql")
    sql = "truncate table {0};".format(target_table)
    try:
        mysql.delete(sql)
    except BaseException as e:
        logerror = 'Error occurred when truncate table {0}:{1}'.format(target_table, e)
        print(logerror)
        # gl.write_log(logpath, 'error', logerror)
        return False
    finally:
        mysql.dispose()
    
    # 将最新的分位数据插入维表
    mysql = msc.MyPymysqlPool("dbMysql")
    sql = "insert into {0}(hour,site_set,promoted_object_type,ind_first_industry_id,ctr_q1,ctr_q2) \
           VALUES (%s,%s,%s,%s,%s,%s);".format(target_table)
    try:
        mysql.insertMany(sql, quanitle_lists)
    except BaseException as e:
        logerror = 'Error occurred when insert data into {0}:{1}'.format(target_table, e)
        print(logerror)
        # gl.write_log(logpath, 'error', logerror)
        return False
    finally:
        mysql.dispose()
    
    return True