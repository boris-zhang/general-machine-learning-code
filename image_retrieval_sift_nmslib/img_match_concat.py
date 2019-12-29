#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: img_match_concat.py
Description:
Variables: None
Author: 
Change Activity: First coding on 2018/6/11
---------------------------------------------------------------------------
'''
import os
import math
import sys
import timeit
import numpy as np
import pandas as pd

import cv2
import pytesseract

sys.path.append("/home/dmer/models/pub")
import mysql_conn as ms

pm_tr = 0.4
nm_t = 0.02
mp_t = 5  #匹配点下限

def fetch_data(tab_rst, lmt_kpsup=0.05, lmt_kpup=1, op_tt='2000-01-01 00:00:00', mtksup=10):
	mysql = ms.MyPymysqlPool("dbMysql")
	sql = "SELECT img_path_query,img_path_lib,ratio_kpnum_match,kpnum_match,kpnum_img_query,kpnum_img_lib \
			FROM %s \
			where ratio_kpnum_match between %f and %f and op_dt>='%s' and kpnum_match>=%d" \
					% (tab_rst, lmt_kpsup, lmt_kpup, op_tt, mtksup)
	rst = mysql.getAll(sql)

	if not rst:
		mysql.dispose()	 # 释放资源
		print('	 No data!')
		return -1
	else:
		df = pd.DataFrame(list(list(x.values()) for x in rst)).fillna('0')
		print('	 %d rows data have been fetched.' % len(df))
		mysql.dispose()
		return df

def ocr_detector(fns, clear_text):
    image = cv2.imread(fns)
    # 背景色处理
    # image = image.point(lambda x: 0 if x < 143 else 255)
    text = pytesseract.image_to_string(image, lang='chi_sim')
    # print(text)
    for kw in clear_text:
        if kw in text:
            return True
    return False

if __name__ == "__main__":		
	
	line_ratio = 0.3
	lmt_kpsup = 0.05
	lmt_kpup = 1
	op_tt='2000-01-01 00:00:00'
	mtksup = 10
	if len(sys.argv) == 6:
		mtksup = int(sys.argv[5])
	if len(sys.argv) >= 5:
		op_tt = sys.argv[4]
	if len(sys.argv) >= 4:
		lmt_kpup = float(sys.argv[3])
	if len(sys.argv) >= 3:
		lmt_kpsup = float(sys.argv[2])
	if len(sys.argv) >= 2:
		line_ratio = float(sys.argv[1])
	if len(sys.argv) < 2 or len(sys.argv) > 6:
		print('bad input values: <line_ratio lmt_kpsup lmt_kpup optime uplimit>')
		exit(-1)
	
	tab_rst = 'imgbase.imgquery_result_info'
	rst_dir_good = '/data/pkl_file/imgp/match_result'
	rst_dir_bad = "/data/pkl_file/imgp/match_result/bad_queryimg"	# 错误图像OCR处理
	clear_text = ['垃圾', '空间不', '内存', '连接成功', '清理', '广告投放日']

	rstData = fetch_data(tab_rst, lmt_kpsup, lmt_kpup, op_tt, mtksup)
	Imgset = np.array(rstData.iloc[:, 0:6]).tolist()

	for mat in Imgset:
		path_query = mat[0]
		path_lib = mat[1]
		mat_ratio = mat[2]
		kpnum_m = mat[3]
		kpnum_q = mat[4]
		kpnum_l = mat[5]
		# print(path_query, path_lib, mat_ratio)
		fn_query = path_query.split('/')[-1]
		fn_lib = path_lib.split('/')[-1]

		# 转成jpeg格式，否则图像显示可能会有异常
		img_query = cv2.imread(path_query)
		img_encode = cv2.imencode('.jpg', img_query)[1]
		img1 = cv2.imdecode(img_encode, -1)

		img_lib = cv2.imread(path_lib)
		img_encode = cv2.imencode('.jpg', img_lib)[1]
		img2 = cv2.imdecode(img_encode, -1)

		sift = cv2.xfeatures2d.SIFT_create()
		kp1, des1 = sift.detectAndCompute(img1, None)
		kp2, des2 = sift.detectAndCompute(img2, None)

		# 用同样的参数，采用蛮力匹配算法验证识别结果
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1, des2, k=2)
		good = []
		for m, n in matches:
			if m.distance < pm_tr * n.distance:
				good.append([m])
		if len(good) >= min(len(kp1),len(kp2)) * nm_t and len(good) >= mp_t:   # 匹配识别阈值
			img_mat = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:int(line_ratio*len(good))], None, flags=2)
			
			# 通过OCR识别图像，如果是错误图像，放到bad目录
			isBfile = ocr_detector(path_query, clear_text)
			if isBfile==True:
				rst_dir = rst_dir_bad
			else:
				rst_dir = rst_dir_good

			path_mat = rst_dir + '/(%f, %d, %d, %d) ' % (mat_ratio, kpnum_m, kpnum_q, kpnum_l) + fn_query + '---' + fn_lib
			cv2.imencode('.jpg', img_mat)[1].tofile(path_mat)
			print('saved: %s.jpg' % path_mat)