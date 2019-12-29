#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: image_match_brute.py
Description: 
			 目标: 蛮力匹配
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
import cv2

sys.path.append("/home/dmer/models/pub")
import mysql_conn as ms

src_folder = "/data/image_query/news_video_company/website/comdir"
lib_folder = "/data/image_file/creative_raw/creative/download/20180601/image/jrtt"
rst_folder = "/data/image_file/siftmatch_rst"
pm_t = 0.6
nm_t = 3

if __name__ == '__main__':

   sift = cv2.xfeatures2d.SIFT_create()
   for fnsrc in os.listdir(src_folder):
        if fnsrc.split('.')[-1].upper() in ("JPG", "JPEG", "PNG", "BMP", "GIF"):
            fns = src_folder + '/' + fnsrc
            print('searching for: ', fns)
            img1 = cv2.imdecode(np.fromfile(fns, dtype=np.uint8), -1)
            kp1, des1 = sift.detectAndCompute(img1, None)

            for fnlib in os.listdir(lib_folder):
                if fnlib.split('.')[-1].upper() in ("JPG", "JPEG", "PNG", "BMP", "GIF"):
                    fng = lib_folder + '/e62041927c5033bf0c3ffe84de07e4d5.jpeg'
                    img2 = cv2.imdecode(np.fromfile(fng, dtype=np.uint8), -1)
                    kp2, des2 = sift.detectAndCompute(img2, None)
                    minkpnum = min(len(kp1), len(kp2))
                    if minkpnum > 20: 
	                    # 蛮力匹配算法,有两个参数，距离度量(L2(default),L1)，是否交叉匹配(默认false)
	                    bf = cv2.BFMatcher()
	                    matches = bf.knnMatch(des1, des2, k=2)
	                    # print(type(matches))
	                    # print(matches)
	                    # print(len(matches))
	                    # exit()
	                    if len(matches) * nm_t < minkpnum:
	                    	continue

	                    # cv2.drawMatchesKnn expects list of lists as matches.
	                    # opencv3.0有drawMatchesKnn函数
	                    # Apply ratio test
	                    # 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
	                    # 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
	                    good = []
	                    for m, n in matches:
	                        if m.distance < pm_t * n.distance:
	                            print('ddist: ', m.distance/n.distance)
	                            good.append([m])

	                    if len(good) * nm_t > minkpnum:   # 匹配识别阈值
	                        print('lengood: %d, minkpnum:%f ' % (len(good), min(len(kp1), len(kp2))))
	                        print('matched: ', fns, '---', fng)
	                        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:int(len(good))], None, flags=2)
	                        cv2.imencode('.jpeg', img3)[1].tofile(rst_folder + '/' + fnsrc + '---' + fnlib)
