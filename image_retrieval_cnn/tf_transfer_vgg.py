#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : tf_cnn_transfer_test1.py
Description : 基于tensorflow的迁移学习图像检索测试。
              1. 。
Author      : Zhang Zhiyong
Created at  : 2018/12/03
---------------------------------------------------------------------------
"""
import warnings
warnings.filterwarnings("ignore")
from urllib.request import urlretrieve
from os.path import isfile, isdir

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.externals import joblib
import inspect
from tqdm import tqdm

import tensorflow as tf
from tensorflow_vgg import vgg16

import sys
sys.path.append("/home/dmer/models/pub/")
import mysql_conn as ms
import redis_lib as rds
from project_utils import const


# 配置常量
const.TRAIN_DATE = 'traindate'             #  
const.VGG_DIR = '/data/cnn_image/'


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def file_check(dir_name, file_name):
    if not isdir(dir_name):
        raise Exception("The directory %s exist!", dir_name)

    if not isfile(dir_name + file_name):
        raise Exception("The file %s/%s is not exists!" %(dir_name, file_name))
    else:
    	print("The file %s/%s is exists!" %(dir_name, file_name))


if __name__ == '__main__':

    file_check(const.VGG_DIR, "vgg16.npy")
    file_check(const.VGG_DIR, "flower_photos.tgz")
    vgg16_npy_path = const.VGG_DIR + "/vgg16.npy"

    data_dir = const.VGG_DIR + '/flower_photos/'
    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(data_dir + each)]

    # Set the batch size higher if you can fit in in your GPU memory
    batch_size = 10
    codes_list = []
    labels = []
    batch = []
     
    codes = None
     
    with tf.Session() as sess:
        vgg = vgg16.Vgg16(vgg16_npy_path)
        input_ = tf.placeholder(tf.float32, [224, 224, 3, None])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)
     
        for each in classes:
            print("Starting {} images".format(each))
            class_path = data_dir + each
            files = os.listdir(class_path)
            for ii, file in enumerate(files, 1):
                # Add images to the current batch
                # utils.load_image crops the input images for us, from the center
                img = utils.load_image(os.path.join(class_path, file))
                batch.append(img.reshape((1, 224, 224, 3)))
                labels.append(each)
                
                # Running the batch through the network to get the codes
                if ii % batch_size == 0 or ii == len(files):
                    images = np.concatenate(batch)
     
                    feed_dict = {input_: images}
                    codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
                    
                    # Here I'm building an array of the codes
                    if codes is None:
                        codes = codes_batch
                    else:
                        codes = np.concatenate((codes, codes_batch))
                    
                    # Reset to start building the next batch
                    batch = []
                    print('{} images processed'.format(ii))