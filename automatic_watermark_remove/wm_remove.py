#!/home/kddmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: wm_remove.py
Description: 
			 目标: 模拟google2017的方法自动去水印
Variables: None
Author: 
Change Activity: First coding on 2018/6/6
---------------------------------------------------------------------------
'''
import sys;
import datetime
import warnings
warnings.filterwarnings("ignore")
import copy
import os 
import cv2
import numpy as np
import pickle as pkl

sys.path.append("/home/kddmer/models/imgp/auto_wmr")
import preprocess as pp
import estimate_watermark as ew
import watermark_reconstruct as wr



'''
1. 用原图和水印图提取水印
2. 用高质量图提取水印
'''

raw_folder = '/home/kddmer/models/imgp/auto_wmr/images/fotolia_raw'
proc_folder = '/home/kddmer/models/imgp/auto_wmr/images/fotolia_proc'
# wmr_folder = '/home/kddmer/models/imgp/auto_wmr/images/fotolia_wmr'

if __name__ == "__main__":

	# 预处理图像为500*500
	# pp.preprocess(raw_folder, resize_folder)

	# 识别水印位置
	print('Estimating watermark start...')
	gx, gy, gxlist, gylist = ew.estimate_watermark(proc_folder)
	# est = ew.poisson_reconstruct(gx, gy)
	cropped_gx, cropped_gy = ew.crop_watermark(gx, gy)
	W_m = ew.poisson_reconstruct(cropped_gx, cropped_gy)

	# 水印提取和区域图像合成
	print('Solving images start...')
	fpath, dir, files = list(os.walk(proc_folder))[0]
	img = cv2.imread(os.sep.join([os.path.abspath(fpath), files[0]]))
	im, start, end = ew.watermark_detector(img, cropped_gx, cropped_gy)

	num_images = len(gxlist)
	# Wm = (255 * ew.PlotImage(W_m))
	Wm = W_m - W_m.min()
	J, img_paths = wr.get_cropped_images(proc_folder, num_images, start, end, cropped_gx.shape)

	# get threshold of W_m for alpha matte estimate
	alph_est = wr.estimate_normalized_alpha(J, Wm, num_images)
	alph = np.stack([alph_est, alph_est, alph_est], axis=2)
	C, est_Ik = wr.estimate_blend_factor(J, Wm, alph)

	alpha = alph.copy()
	for i in range(3):
		alpha[:,:,i] = C[i] * alpha[:, :, i]
	Wm = Wm + alpha * est_Ik
	W = copy.deepcopy(Wm)
	for i in range(3): 
		W[:,:,i] /= C[i]
	Jt = J
	Wk, Ik, W, alpha1 = wr.solve_images(Jt, W_m, alpha, W)

	# 保存数据
	pkl.dump(Ik, open('/home/kddmer/models/imgp/auto_wmr/images/pkl/Ik.pkl', 'wb'))
	pkl.dump(img_paths, open('/home/kddmer/models/imgp/auto_wmr/images/pkl/img_paths.pkl', 'wb'))
	pkl.dump(start, open('/home/kddmer/models/imgp/auto_wmr/images/pkl/start.pkl', 'wb'))
	pkl.dump(end, open('/home/kddmer/models/imgp/auto_wmr/images/pkl/end.pkl', 'wb'))
	# Ik = pkl.load(open('/home/kddmer/models/imgp/auto_wmr/images/pkl/Ik.pkl', 'rb'))
	# img_paths = pkl.load(open('/home/kddmer/models/imgp/auto_wmr/images/pkl/img_paths.pkl', 'rb'))
	# start = pkl.load(open('/home/kddmer/models/imgp/auto_wmr/images/pkl/start.pkl', 'rb'))
	# end = pkl.load(open('/home/kddmer/models/imgp/auto_wmr/images/pkl/end.pkl', 'rb'))

	# 图像恢复
	k = 0
	mg = 3 # 边缘经常有毛边，跳开这部分
	for file in img_paths:
		img = cv2.imread(file)
		iw = 0
		for i in range(start[0]+mg, start[0]+end[0]-mg):	# start end形如: (229, 164) (39, 172)
			jw = 0
			for j in range(start[1]+mg, start[1]+end[1]-mg):
				img[i][j] = Ik[k][iw+mg][jw+mg]
				jw +=1
			iw +=1
		filewmr = file.split('.')[0] + '_wmr2.' + file.split('.')[1]
		cv2.imwrite(filewmr, img)
		k +=1

	print('%s Reconstructing images done!' % iw)