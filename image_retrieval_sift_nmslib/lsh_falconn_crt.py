#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: lsh_falconn_crt.py
Description: 
			 目标: 测试falconn索引行
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

import cv2
import falconn
import gc 
from sklearn.neighbors import LSHForest

sys.path.append("/home/dmer/models/pub")
import mysql_conn as ms


def get_siftdata(num_s, num_e):
	mysql = ms.MyPymysqlPool("dbMysql")
	sql = "SELECT * FROM imgbase.img_sift_features \
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


def aligned(a, alignment=32):
    if (a.ctypes.data % alignment) == 0:
        return a
    extra = alignment // a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) // a.itemsize
    aa = buf[ofs:ofs + a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert (aa.ctypes.data % alignment) == 0
    
    return aa

def pre_data_for_cp(data_db):
	data = []
	for res in data_db:
		data.append(np.array([(d,x) for d,x in res.items()])[3:,1])
	data = np.array(data)
	data = data.astype(np.float32)
	data -= np.mean(data, axis=0)
	data = aligned(data)
	return data




if __name__ == "__main__":

	lshf = LSHForest(random_state=42)

	# dfImgset1 = get_siftdata(0, 5000)
	# imgid1 = dfImgset1.iloc[:, 0].values
	# imgpos1 = dfImgset1.iloc[:, 1:2].values
	# imgsift1 = dfImgset1.iloc[:, 3:].values
	# lshf.fit(imgsift1)
	# pkl.dump(lshf, open('/data/image_file/pkl/lshf1.pkl', 'wb'))
	# pkl.dump(imgid1, open('/data/image_file/pkl/imgid1.pkl', 'wb'))

	# dfImgset2 = get_siftdata(5001, 10000)
	# imgid2 = dfImgset2.iloc[:, 0].values
	# imgpos2 = dfImgset2.iloc[:, 1:2].values
	# imgsift2 = dfImgset2.iloc[:, 3:].values
	# lshf.fit(imgsift2)
	# pkl.dump(lshf, open('/data/image_file/pkl/lshf2.pkl', 'wb'))
	# pkl.dump(imgid2, open('/data/image_file/pkl/imgid2.pkl', 'wb'))

	# dfImgset3 = get_siftdata(10001, 15000)
	# imgid3 = dfImgset3.iloc[:, 0].values
	# imgpos3 = dfImgset3.iloc[:, 1:2].values
	# imgsift3 = dfImgset3.iloc[:, 3:].values
	# lshf.fit(imgsift3)
	# pkl.dump(lshf, open('/data/image_file/pkl/lshf3.pkl', 'wb'))
	# pkl.dump(imgid3, open('/data/image_file/pkl/imgid3.pkl', 'wb'))


	# dfImgset4 = get_siftdata(15001, 16000)
	# imgid4 = dfImgset4.iloc[:, 0].values
	# imgpos4 = dfImgset4.iloc[:, 1:2].values
	# imgsift4 = dfImgset4.iloc[:, 3:].values
	# pkl.dump(imgid4, open('/data/image_file/pkl/imgid4.pkl', 'wb'))
	# pkl.dump(imgsift4, open('/data/image_file/pkl/imgsift4.pkl', 'wb'))
	imgid_fetch = pkl.load(open('/data/image_file/pkl/imgid4.pkl', 'rb'))
	imgsift_fetch = pkl.load(open('/data/image_file/pkl/imgsift4.pkl', 'rb'))
	img_fetch = np.c_[imgid_fetch, imgsift_fetch]

	# dfImgset5 = get_siftdata(1001, 1001)
	# img_fetch = np.c_[dfImgset5.iloc[:, 0].values, dfImgset5.iloc[:, 3:].values]

	for ii in [1, 2, 3]: 
		imgid = pkl.load(open('/data/image_file/pkl/imgid%d.pkl' % ii, 'rb'))
		lshf = pkl.load(open('/data/image_file/pkl/lshf%d.pkl' % ii, 'rb'))
		for i in range(15001, 16001):
			# 提取检索图像的特征点
			print('image retrieval for %d.' % i)
			kpnum = len(img_fetch[:, 0])
			imgsift = [img_fetch[x, 1:] for x in range(kpnum) if img_fetch[x, 0] == i]

			# 检索图库，每个点找5个近邻点，过滤出有2个以上的被选图，识别是否匹配
			# distances, indices = lshf.radius_neighbors(imgsift, radius=0.1)
			distances, indices = lshf.kneighbors(imgsift, n_neighbors=3)
			getKpset = indices.reshape(-1, 1)
			getKpset = [getKpset[x][0] for x in range(len(getKpset))]
			# print(getKpset)

			# 统计匹配点对应的图像次数
			imgCnt = {}
			for kpid in getKpset:
				if not imgid[kpid] in imgCnt:
					imgCnt[imgid[kpid]] = 1
				else:
					imgCnt[imgid[kpid]] +=1
			
			# 匹配图像阈值为10%的点匹配到
			imgCnt = {k: v for k, v in imgCnt.items() if v > kpnum * 0.1}
			imgFetch = sorted(imgCnt.items(), key = lambda x: x[1], reverse = True)
			if len(imgFetch) > 0:
				print('image', i, ': ', imgFetch)
			else:
				print('image', i, ': ', imgFetch)


