#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: img_proprecess.py
Description: 
             目标: 图像预处理
Variables: None
Author: 
Change Activity: First coding on 2018/6/12
---------------------------------------------------------------------------
'''
import sys
import os
import cv2
import shutil

from PIL import Image
import pytesseract

def movefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath, fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile, dstfile)          #移动文件
        print("move %s -> %s" %(srcfile, dstfile))

def copyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


src_dir = '/data/pkl_file/imgp/match_result'
dst_dir = "/data/pkl_file/imgp/match_result/bad_img"
clear_text = ['垃圾', '空间不', '内存', '连接成功', '清理', '广告投放日']

def ocr_detector(fns, clear_text):

    image = Image.open(fns)

    # 背景色处理
    # image = image.point(lambda x: 0 if x < 143 else 255)

    text = pytesseract.image_to_string(image, lang='chi_sim')
    # print(text)
    
    for kw in clear_text:
        if kw in text:
            return kw, True