#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: lsh_nmslib.py
Description: 
			 目标: lsh_nmslib
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


def fetch_data(num_s, num_e):
	mysql = ms.MyPymysqlPool("dbMysql")
	sql = "SELECT * FROM imgbase.img_sift_features1 \
			where img_id between %s and %s" % (num_s, num_e)
	rst = mysql.getAll(sql)

	if not rst:
		mysql.dispose()		# 释放资源
		print('		No data!')
		return -1
	else:
		df = pd.DataFrame(list(list(x.values()) for x in rst)).fillna('0')
		print('		%d rows data have been fetched.' % len(df))
		mysql.dispose()
		return df


def lsh_crt(lshidx, batnum=10000):

	for i in range(30):
		dfImgset = fetch_data(i*batnum, (i+1)*batnum)
		imgid = dfImgset.iloc[:, 0].values
		imgpos = dfImgset.iloc[:, 1:2].values
		imgsift = dfImgset.iloc[:, 3:].values

		nmslib_index.addDataPointBatch(imgsift)
		nmslib_index.createIndex({'post': 2, 'delaunay_type': 1, 'M': 80}, print_progress=True)
		nmslib_index.saveIndex('/data/pkl_file/imgp/nmslib_index%d.lsh' % i)

		pkl.dump(imgid, open('/data/pkl_file/imgp/imgid%d.pkl' % i, 'wb'))
		pkl.dump(imgpos, open('/data/pkl_file/imgp/imgpos%d.pkl' % i, 'wb'))
		pkl.dump(imgsift, open('/data/pkl_file/imgp/imgsift%d.pkl' % i, 'wb'))


if __name__ == "__main__":

	# initialize a new index, using a HNSW index on Cosine Similarity
	nmslib_index = nmslib.init(method='hnsw', space='l2')
	batnum = 10000
	lsh_crt(nmslib_index, batnum)
	exit()

	for i in [0, 1, 2]:
		lshf = nmslib_index.loadIndex('/data/pkl_file/imgp/nmslib_index%d.lsh' % i)
		imgid = pkl.load(open('/data/pkl_file/imgp/imgid%d.pkl' % i, 'rb'))
		imgpos = pkl.load(open('/data/pkl_file/imgp/imgpos%d.pkl' % i, 'rb'))
		imgsift = pkl.load(open('/data/pkl_file/imgp/imgsift%d.pkl' % i, 'rb'))
		img_fetch = np.c_[imgid, imgsift]

	# query for the nearest neighbours of the first datapoint
	# ids, distances = nmslib_index.knnQuery(img_fetch[0], k=5)
	# print(ids)
	# print(distances)

	# # get all nearest neighbours for all the datapoint
	# # using a pool of 4 threads to compute
	# neighbours = index.knnQueryBatch(data, k=10, num_threads=4)

	# dfImgset5 = get_siftdata(1001, 1001)
	# img_fetch = np.c_[dfImgset5.iloc[:, 0].values, dfImgset5.iloc[:, 3:].values]

	# for ii in [1, 2, 3]: 
	# 	imgid = pkl.load(open('/data/image_file/pkl/imgid%d.pkl' % ii, 'rb'))
	# 	lshf = pkl.load(open('/data/image_file/pkl/lshf%d.pkl' % ii, 'rb'))
	# 	for i in range(15001, 16001):
	# 		# 提取检索图像的特征点
	# 		print('image retrieval for %d.' % i)
	# 		kpnum = len(img_fetch[:, 0])
	# 		imgsift = [img_fetch[x, 1:] for x in range(kpnum) if img_fetch[x, 0] == i]

	# 		# 检索图库，每个点找5个近邻点，过滤出有2个以上的被选图，识别是否匹配
	# 		# distances, indices = lshf.radius_neighbors(imgsift, radius=0.1)
	# 		distances, indices = lshf.kneighbors(imgsift, n_neighbors=3)
	# 		getKpset = indices.reshape(-1, 1)
	# 		getKpset = [getKpset[x][0] for x in range(len(getKpset))]
	# 		# print(getKpset)

	# 		# 统计匹配点对应的图像次数
	# 		imgCnt = {}
	# 		for kpid in getKpset:
	# 			if not imgid[kpid] in imgCnt:
	# 				imgCnt[imgid[kpid]] = 1
	# 			else:
	# 				imgCnt[imgid[kpid]] +=1
			
	# 		# 匹配图像阈值为10%的点匹配到
	# 		imgCnt = {k: v for k, v in imgCnt.items() if v > kpnum * 0.1}
	# 		imgFetch = sorted(imgCnt.items(), key = lambda x: x[1], reverse = True)
	# 		if len(imgFetch) > 0:
	# 			print('image', i, ': ', imgFetch)
	# 		else:
	# 			print('image', i, ': ', imgFetch)


