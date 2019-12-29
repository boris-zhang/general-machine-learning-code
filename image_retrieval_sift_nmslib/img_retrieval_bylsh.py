#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: img_retrieval_bylsh.py
Description:
Variables: None
Author: 
Change Activity: First coding on 2018/6/8
---------------------------------------------------------------------------
'''
import os
import math
import sys
import timeit
import numpy as np
import pandas as pd
import pickle as pkl

# import cv2
import nmslib

sys.path.append("/home/dmer/models/pub")
import mysql_conn as ms

pm_tr = 0.4
nm_t = 0.02
nb_t = 100000
mp_t = 5   # 匹配点下限，过滤低匹配量

def fetch_data(tabimg, tabfets, num_s, num_e):
    mysql = ms.MyPymysqlPool("dbMysql")
    sql = "SELECT t2.* \
            FROM %s t1 \
            inner join %s t2 on t1.img_id=t2.img_id \
            where t1.img_id between %s and %s" % (tabimg, tabfets, num_s, num_e)
    rst = mysql.getAll(sql)

    if not rst:
        mysql.dispose()     # 释放资源
        print('     No data!')
        return -1
    else:
        df = pd.DataFrame(list(list(x.values()) for x in rst)).fillna('0')
        print('     %d rows data have been fetched.' % len(df))
        mysql.dispose()
        return df

def insert_result(tab_rst, tab_image_q, tab_image_lib, imgid_q, mtid, mpkpnum, kpnum_l, kpnum_q):
    mysql = ms.MyPymysqlPool("dbMysql")
    sql = "insert into %s(img_id_query, img_id_lib, img_path_query, img_path_lib, \
                ratio_kpnum_match, kpnum_match, kpnum_img_query, kpnum_img_lib, op_dt) \
            select t1.img_id, t2.img_id, t1.img_path, t2.img_path, %f, %d, %d, %d, SYSDATE() \
            from %s t1 \
            inner join %s t2 on t2.img_id=%d \
            where t1.img_id=%d" % (tab_rst, mpkpnum/min(kpnum_l, kpnum_q), mpkpnum, kpnum_q, kpnum_l, 
                                        tab_image_q, tab_image_lib, mtid, imgid_q)
    rnum = mysql.insert(sql)
    mysql.dispose()     # 释放资源
    return rnum


if __name__ == "__main__":        
    
    stpos = int(sys.argv[1])
    endpos = int(sys.argv[2])
    batnum = [int(x) for x in sys.argv[3].split(',')]
    print('query id: %d--%d, index batnum:' % (stpos, endpos), batnum)
    # tab_image_q = 'imgbase.imglib_info'
    # tab_fets_q = 'imgbase.imglib_sift_features'
    tab_image_q = 'imgbase.imgquery_info'
    tab_fets_q = 'imgbase.imgquery_sift_features'
    tab_rst = 'imgbase.imgquery_result_info'
    tab_image_lib = 'imgbase.imglib_info'

    dfImgset = fetch_data(tab_image_q, tab_fets_q, stpos, endpos)
    imgid_query = dfImgset.iloc[:, 0].values
    imgpos_query = dfImgset.iloc[:, 1:2].values
    img_query = np.c_[dfImgset.iloc[:, 0].values, dfImgset.iloc[:, 3:].values]

    # 提取索引文件，批量检索
    for idxseq in batnum:
        print('\nload nmslib index nmslib_index%d.lsh...' % idxseq)        
        nmslib_index = nmslib.init(method='hnsw', space='cosinesimil')
        nmslib_index.loadIndex('/data/pkl_file/imgp/nmslib_index%d.lsh' % idxseq, print_progress=True)
    
        imgid_lib = pkl.load(open('/data/pkl_file/imgp/imgid%d.pkl' % idxseq, 'rb'))
        # imgpos_lib = pkl.load(open('/data/pkl_file/imgp/imgpos%d.pkl' % idxseq, 'rb'))
        # imgsift_lib = pkl.load(open('/data/pkl_file/imgp/imgsift%d.pkl' % idxseq, 'rb'))
        # img_lib = np.c_[imgid_lib, imgsift_lib]

        imgid_qSet = set(list(imgid_query)) # 提取被查询图id
        for imgid_q in imgid_qSet:
            # 提取检索图像的特征点
            print('   image retrieval, img_id: %d, table name: %s.' % (imgid_q, tab_fets_q))
            
            # 检索图库，每个点找k个近邻点
            img_kpfeats = [img_query[x, 1:] for x in range(len(img_query[:, 0])) if img_query[x, 0] == imgid_q]
            neighbours = nmslib_index.knnQueryBatch(img_kpfeats, k=nb_t, num_threads=4)
            kpnum_q = len(neighbours) #查询图的kp点数
            # print('       kpnum_q: ', kpnum_q)
            # exit()

            # 逐点找匹配点
            imgkpMatch = {}
            for k in range(kpnum_q):
                kp_nbid = list(neighbours[k][0])
                kp_nbdist = list(neighbours[k][1])
                # print('       kp_nbid: ', kp_nbid)
                # print('       kp_nbdist: ', np.min(kp_nbdist), np.max(kp_nbdist))
                # exit()

                # 提取匹配点中匹配多次的图像id和距离
                # 存入字典{3136: [21348.0, 45853.0, 46525.0], 4684: [21865.0, 30497.0], 1712: [22365.0, 29510.0]}
                imgkpDist = {}
                ii = 0
                # print(kp_nbid)
                for kpid in kp_nbid:
                    if not imgid_lib[kpid] in imgkpDist:
                        imgkpDist[imgid_lib[kpid]] = [kp_nbdist[ii]]
                    else:
                        imgkpDist[imgid_lib[kpid]].append(kp_nbdist[ii])
                    ii +=1

                imgkpDist = {k: v for k, v in imgkpDist.items() if len(v) > 1}  # 找到有2个近邻点的图像id
                # print('       imgkpDist: ', imgkpDist)
                
                # 获取与A 距离最近的点B（最近）和C（次近），只有当B/C 小于阈值时才被认为是匹配
                # 存入匹配图像id
                for imgid, kpdist in imgkpDist.items():  # (3136, [21348.0, 45853.0, 46525.0])
                    stkpdist = sorted(kpdist)
                    # print('       imgid: %d, stkpdist: %s, raito: %f, %s' % (imgid, stkpdist, stkpdist[0] / stkpdist[1], stkpdist[0] / stkpdist[1] < pm_tr))
                    if (stkpdist[0] / stkpdist[1] < pm_tr):
                        if not imgid in imgkpMatch:
                            imgkpMatch[imgid] = 1
                        else:
                            imgkpMatch[imgid] += 1
                        # print(imgid, imgkpMatch)
            # print('       imgkpDist: ', imgkpDist)
            # print('       imgkpMatch: ', imgkpMatch)

            # 有超过query图中 1/m_t 的匹配点，则认定为匹配图像，按照匹配数排序
            imgFetch = {k: (v, len([imgid_lib[x] for x in range(len(imgid_lib)) if imgid_lib[x]==k])) 
                        for k, v in imgkpMatch.items() if v > mp_t 
                            and v > nm_t * min(kpnum_q, len([imgid_lib[x] for x in range(len(imgid_lib)) if imgid_lib[x]==k]))}
            # imgFetch = sorted(imgFetch.items(), key = lambda x: x[1], reverse = True)
            # print('       imgFetch: ', imgFetch)
            istnum = 0
            for mtid, mtv in imgFetch.items():
                # 读取图像id路径，插入数据表
                # print('     query image id: %d, matched image id: %d, matched kp ratio: %f(%d of %d)' \
                #         % (imgid_q, mtid, mpkpnum/kpnum_q, mpkpnum, kpnum_q))
                rnum = insert_result(tab_rst, tab_image_q, tab_image_lib, imgid_q, mtid, mtv[0], mtv[1], kpnum_q)
                istnum +=rnum
            if istnum > 0:
                print('     %d matched images for query image id: %d have been inserted.' % (istnum, imgid_q))

            # exit()

