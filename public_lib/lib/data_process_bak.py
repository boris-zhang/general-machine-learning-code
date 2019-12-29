#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: data_process.py
Description: 数据处理函数包
Variables: None
Author: zhangzhiyong
Change Activity: First coding on 2018/5/8
---------------------------------------------------------------------------
'''

import sys
import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

sys.path.append("/home/dmer/models/pub/")
import comb_gdt_pub as gpub
import mysql_conn as ms


# 数据插入，后续参数需要调整成通用的
def dataInsert(tbn, trans_dt, indst_id, indst_name, ttype, bevt, freqItems, freqItemSups):	
	mysql = ms.MyPymysqlPool("dbMysql")
	fstr = "\'" + trans_dt +  "\'," + str(indst_id) + ",\'" + indst_name + "\',\'" + bevt + "\'"
	# sqlInsert = "insert into " + tbn + "(dt,industry_id,industry_name,trans_type,adg_billing_event,\
	# 				item_num,adg_bid_amount_bin,adg_ts_workday1,adg_ts_workday2,adg_ts_workday3,\
	# 				adg_ts_workday4,adg_ts_workday5,adg_ts_workday6,adg_ts_workday7,adg_ts_weekend1,\
	# 				adg_ts_weekend2,adg_ts_weekend3,adg_ts_weekend4,adg_ts_weekend5,\
 #    				adg_site_set,adg_product_type,ad_adcreative_name,ad_adcreative_destination_url_SHA,\
 #    				adc_spl_image_name,adc_spl_image_thumb,adc_spl_image_image,\
 #    				ade_image,ade_title,ade_image2,clk_ratio_bin1,support) \
 #                    values(" + fstr + ",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
 #                    					%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

	sqlInsert = "insert into " + tbn + "(dt,industry_id,industry_name,adg_billing_event,\
					item_num,adg_bid_amount,adg_ts_workday1,adg_ts_workday2,adg_ts_workday3,\
					adg_ts_workday4,adg_ts_workday5,adg_ts_workday6,adg_ts_workday7,adg_ts_weekend1,\
					adg_ts_weekend2,adg_ts_weekend3,adg_ts_weekend4,adg_ts_weekend5,adg_site_set,\
					adg_product_type,adg_configured_status,ad_adcreative_name,ade_image_hash,\
					ade_title,ade_image2_hash,ad_space,adc_form_size,adc_form_detail,adc_description,\
					adc_drp1_4,adc_drp1_5,adc_drp1_6,adc_drp1_17,adc_drp1_23,adc_drp1_24,adc_drp1_31,\
					adc_drp1_32,adc_drp1_33,adc_drp1_36,adc_drp1_37,adc_drp1_38,adc_drp3_5,adc_drp3_6,\
					adc_drp3_11,adc_drp3_14,adc_drp3_16,adc_drp3_20,adc_drp3_25,adc_drp3_26,adg_crt_days,\
					atv_ratio_bin11_hisratio_h,tag_age,tag_gender,tag_business_interest,tag_gloc_location_types,\
					tag_gloc_regions,tag_network_type,tag_app_install_status,tag_abh_object_type,\
					tag_abh_object_id_list,tag_abh_time_window,tag_abh_act_id_list,tag_customized_audience,\
					tag_shopping_capability,tag_player_consupt,tag_paying_user_type,tag_media_category_union,\
					tag_boi_i_targeting_tags,atv_ratio_bin11,support) \
                    values(" + fstr + ",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
                    					%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
                    					%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
                    					%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
                    					%s,%s,%s,%s,%s,%s)"                    					
	i = 0
	vals = ()
	print('			%s concat sql...' % datetime.datetime.now())
	for itemSet in freqItems:
		while None in itemSet:		# fpgrowth返回有None {'te4_24', 'te5_20', None}
			itemSet.remove(None)
		if itemSet:
			val = gpub.items_concat_values(itemSet, freqItemSups[i])
			vals = vals + (val,)
		i +=1
	print('			%s insert data...' % datetime.datetime.now())
	rc = mysql.insertMany(sqlInsert, vals)
	mysql.dispose()		# 释放资源

	return rc

def splitComa(s):
	if type(s) == str:
		return s.split(', ')
	else:
		return []

def fetchDataGDT(trans_tab, dt, indst_id, bill_evt, limit=''):
	mysql = ms.MyPymysqlPool("dbMysql")
	sql = "SELECT \
			atv_flag \
			,id \
			,adg_ts_workday1 \
			,adg_ts_workday2 \
			,adg_ts_workday3 \
			,adg_ts_workday4 \
			,adg_ts_workday5 \
			,adg_ts_workday6 \
			,adg_ts_workday7 \
			,adg_ts_weekend1 \
			,adg_ts_weekend2 \
			,adg_ts_weekend3 \
			,adg_ts_weekend4 \
			,adg_ts_weekend5 \
			,adg_site_set \
			,adg_product_type \
			,adg_configured_status \
			,ad_adcreative_name \
			,ad_adcreative_name_game_name \
			,ad_adcreative_name_material_type \
			,ad_adcreative_name_material_effect \
			,ad_adcreative_name_device_type \
			,ad_adcreative_name_interest \
			,ad_adcreative_name_behavior \
			,ad_adcreative_name_paid \
			,ade_image_hash \
			,ade_title \
			,ade_image2_hash \
			,ade_image3_hash \
			,ad_space \
			,adc_form_size \
			,adc_form_detail \
			,adc_description \
			,adg_bid_amount \
			,log(adg_crt_days) \
			,atv_ratio_bin11_hisratio_h \
			,adc_drp1_1 \
			,adc_drp1_2 \
			,adc_drp1_3 \
			,adc_drp1_4 \
			,adc_drp1_5 \
			,adc_drp1_6 \
			,adc_drp1_7 \
			,adc_drp1_8 \
			,adc_drp1_9 \
			,adc_drp1_10 \
			,adc_drp1_11 \
			,adc_drp1_12 \
			,adc_drp1_13 \
			,adc_drp1_14 \
			,adc_drp1_15 \
			,adc_drp1_16 \
			,adc_drp1_17 \
			,adc_drp1_18 \
			,adc_drp1_19 \
			,adc_drp1_20 \
			,adc_drp1_21 \
			,adc_drp1_22 \
			,adc_drp1_23 \
			,adc_drp1_24 \
			,adc_drp1_25 \
			,adc_drp1_26 \
			,adc_drp1_27 \
			,adc_drp1_28 \
			,adc_drp1_29 \
			,adc_drp1_30 \
			,adc_drp1_31 \
			,adc_drp1_32 \
			,adc_drp1_33 \
			,adc_drp1_34 \
			,adc_drp1_35 \
			,adc_drp1_36 \
			,adc_drp1_37 \
			,adc_drp1_38 \
			,adc_drp2_1 \
			,adc_drp2_2 \
			,adc_drp2_3 \
			,adc_drp2_4 \
			,adc_drp2_5 \
			,adc_drp2_6 \
			,adc_drp2_7 \
			,adc_drp2_8 \
			,adc_drp2_9 \
			,adc_drp3_1 \
			,adc_drp3_2 \
			,adc_drp3_3 \
			,adc_drp3_4 \
			,adc_drp3_5 \
			,adc_drp3_6 \
			,adc_drp3_7 \
			,adc_drp3_8 \
			,adc_drp3_9 \
			,adc_drp3_10 \
			,adc_drp3_11 \
			,adc_drp3_12 \
			,adc_drp3_13 \
			,adc_drp3_14 \
			,adc_drp3_15 \
			,adc_drp3_16 \
			,adc_drp3_17 \
			,adc_drp3_18 \
			,adc_drp3_19 \
			,adc_drp3_20 \
			,adc_drp3_21 \
			,adc_drp3_22 \
			,adc_drp3_23 \
			,adc_drp3_24 \
			,adc_drp3_25 \
			,adc_drp3_26 \
			,tag_age \
			,tag_gender \
			,tag_education \
			,tag_relationship_status \
			,tag_living_status \
			,tag_business_interest \
			,tag_location \
			,tag_region \
			,tag_gloc_location_types \
			,tag_gloc_regions \
			,tag_gloc_business_districts \
			,tag_user_os \
			,tag_new_device \
			,tag_device_price \
			,tag_network_type \
			,tag_network_operator \
			,tag_dressing_index \
			,tag_uv_index \
			,tag_makeup_index \
			,tag_climate \
			,tag_temperature \
			,tag_app_install_status \
			,tag_abh_object_type \
			,tag_abh_object_id_list \
			,tag_abh_time_window \
			,tag_abh_act_id_list \
			,tag_customized_audience \
			,tag_shopping_capability \
			,tag_player_consupt \
			,tag_paying_user_type \
			,tag_residential_community_price \
			,tag_media_category_wechat \
			,tag_ad_placement_id \
			,tag_media_category_union \
			,tag_qzone_fans \
			,tag_online_scenario \
			,tag_custom_audience \
			,tag_boi_i_targeting_tags \
		    FROM %s \
			where dt in %s and ind_first_industry_id=%d and adg_billing_event='%s'  %s " \
			% (trans_tab, dt, indst_id, bill_evt, limit)
	# print(sql)
	rst = mysql.getAll(sql)

	if not rst:
		mysql.dispose()		# 释放资源
		print('		No data!')
		return -1
	else:
		# 训练模型不能有缺失值，补0
		df = pd.DataFrame(list(list(x.values()) for x in rst)).fillna('0')
		print('		%d rows data have been fetched.' % len(df))
		mysql.dispose()
		return df

def oneEncoding(ohfeats, unohfeats, mulohfeats, dfTrain, dfTest, stype='std'):
	# 先拼接onehot字段
	feat_idx = []
	enc = LabelBinarizer(sparse_output=True)	# 字符串型类别变量只能用LabelBinarizer()
	cn = 0
	for i, feat in enumerate(ohfeats):
		# print('dfTrain: ', dfTrain)
		x_train = enc.fit_transform(dfTrain.iloc[:,feat].values.reshape(-1, 1))
		x_test = enc.transform(dfTest.iloc[:,feat].values.reshape(-1, 1))
		if i == 0:
			X_train, X_test = x_train, x_test
		else:
			X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
			# X_train, X_test = np.hstack((X_train, x_train)), np.hstack((X_test, x_test))
		
		# 拼接索引标签
		ec = list(enc.classes_)
		for j in range(len(ec)):
			feat_idx.append('%d:%s %d' % (feat, ec[j], cn))
			cn +=1

	# 拼接非onehot字段
	for i, feat in enumerate(unohfeats):
		x1 = dfTrain.iloc[:,feat].values.reshape(-1, 1)
		x2 = dfTest.iloc[:,feat].values.reshape(-1, 1)

		if stype=='std':
			scaler = StandardScaler().fit(x1)
		elif stype=='mm':
			scaler = MaxAbsScaler().fit(x1)
		x_train = scaler.transform(x1)
		x_test = scaler.transform(x2)		
		X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
		# X_train, X_test = np.hstack((X_train, x_train)), np.hstack((X_test, x_test))

		feat_idx.append('%d:%s %d' % (feat, 'unohfeat', cn))
		cn +=1
		# print(dfTrain.iloc[:,feat].values)

	# 拼接多值的onehot字段，
	# 字段值形如'547135, 3547136, 3547137, 3547102, 3547096, 3547092, 3547091, 3547090'
	enc = MultiLabelBinarizer(sparse_output=True)
	for i, feat in enumerate(mulohfeats):
		x1 = dfTrain.iloc[:, feat].apply(splitComa)
		x2 = dfTest.iloc[:, feat].apply(splitComa)
		x_train = enc.fit_transform(x1)
		x_test = enc.transform(x2)
		X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
		# X_train, X_test = np.hstack((X_train, x_train)), np.hstack((X_test, x_test))

		ec = list(enc.classes_)
		for j in range(len(ec)):
			feat_idx.append('%d:%s %d' % (feat, ec[j], cn))
			cn +=1
		# print('x_testML: \n', x_test)
	# exit()

	print('		onehot encoding concat: done!')
	return X_train, X_test, feat_idx