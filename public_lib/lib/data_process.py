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
import itertools

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

sys.path.append("/home/dmer/models/pub/")
# import comb_gdt_pub as gpub
import mysql_conn as ms


# 数据插入，后续参数需要调整成通用的
# def dataInsert(tbn, trans_dt, indst_id, indst_name, ttype, bevt, freqItems, freqItemSups):
# 	mysql = ms.MyPymysqlPool("dbMysql")
# 	fstr = "\'" + trans_dt +  "\'," + str(indst_id) + ",\'" + indst_name + "\',\'" + bevt + "\'"
# 	# sqlInsert = "insert into " + tbn + "(dt,industry_id,industry_name,trans_type,adg_billing_event,\
# 	# 				item_num,adg_bid_amount_bin,adg_ts_workday1,adg_ts_workday2,adg_ts_workday3,\
# 	# 				adg_ts_workday4,adg_ts_workday5,adg_ts_workday6,adg_ts_workday7,adg_ts_weekend1,\
# 	# 				adg_ts_weekend2,adg_ts_weekend3,adg_ts_weekend4,adg_ts_weekend5,\
#  #    				adg_site_set,adg_product_type,ad_adcreative_name,ad_adcreative_destination_url_SHA,\
#  #    				adc_spl_image_name,adc_spl_image_thumb,adc_spl_image_image,\
#  #    				ade_image,ade_title,ade_image2,clk_ratio_bin1,support) \
#  #                    values(" + fstr + ",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
#  #                    					%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
#
# 	sqlInsert = "insert into " + tbn + "(dt,industry_id,industry_name,adg_billing_event,\
# 					item_num,ifnull(adg_bid_amount, ''), '|',adg_ts_workday1,adg_ts_workday2,adg_ts_workday3,\
# 					adg_ts_workday4,adg_ts_workday5,adg_ts_workday6,adg_ts_workday7,adg_ts_weekend1,\
# 					adg_ts_weekend2,adg_ts_weekend3,adg_ts_weekend4,adg_ts_weekend5,adg_site_set,\
# 					adg_product_type,adg_configured_status,ad_adcreative_name,ade_title,\
# 					ade_image_hash,ade_image2_hash,ad_space,adc_form_size,adc_form_detail,adc_description,\
# 					adc_drp1_4,adc_drp1_5,adc_drp1_6,adc_drp1_17,adc_drp1_23,adc_drp1_24,adc_drp1_31,\
# 					adc_drp1_32,adc_drp1_33,adc_drp1_36,adc_drp1_37,adc_drp1_38,adc_drp3_5,adc_drp3_6,\
# 					adc_drp3_11,adc_drp3_14,adc_drp3_16,adc_drp3_20,adc_drp3_25,adc_drp3_26,adg_crt_days,\
# 					atv_ratio_bin11_hisratio_h,tag_age,tag_gender,tag_business_interest,tag_gloc_location_types,\
# 					tag_gloc_regions,tag_network_type,tag_app_install_status,tag_abh_object_type,\
# 					tag_abh_object_id_list,tag_abh_time_window,tag_abh_act_id_list,tag_customized_audience,\
# 					tag_shopping_capability,tag_player_consupt,tag_paying_user_type,tag_media_category_union,\
# 					tag_boi_i_targeting_tags,atv_ratio_bin11,support) \
#                     values(" + fstr + ",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
#                     					%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
#                     					%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
#                     					%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\
#                     					%s,%s,%s,%s,%s,%s)"
# 	i = 0
# 	vals = ()
# 	print('			%s concat sql...' % datetime.datetime.now())
# 	for itemSet in freqItems:
# 		while None in itemSet:		# fpgrowth返回有None {'te4_24', 'te5_20', None}
# 			itemSet.remove(None)
# 		if itemSet:
# 			val = gpub.items_concat_values(itemSet, freqItemSups[i])
# 			vals = vals + (val,)
# 		i +=1
# 	print('			%s insert data...' % datetime.datetime.now())
# 	rc = mysql.insertMany(sqlInsert, vals)
# 	mysql.dispose()		# 释放资源
#
# 	return rc

def splitComa(s):
    if type(s) == str:
        s = s.split(',')
        return list(x.strip() for x in s)
    else:
        return []


def fetchDataGDT(trans_tab, dt, indst_id, bill_evt, limit=''):
    mysql = ms.MyPymysqlPool("dbMysql")
    # sql = "SELECT \
    # 		atv_flag \
    # 		,id \
    # 		,adg_ts_workday1 \
    # 		,adg_ts_workday2 \
    # 		,adg_ts_workday3 \
    # 		,adg_ts_workday4 \
    # 		,adg_ts_workday5 \
    # 		,adg_ts_workday6 \
    # 		,adg_ts_workday7 \
    # 		,adg_ts_weekend1 \
    # 		,adg_ts_weekend2 \
    # 		,adg_ts_weekend3 \
    # 		,adg_ts_weekend4 \
    # 		,adg_ts_weekend5 \
    # 		,adg_site_set \
    # 		,adg_product_type \
    # 		,adg_configured_status \
    # 		,ad_adcreative_name \
    # 		,ad_adcreative_name_game_name \
    # 		,ad_adcreative_name_material_type \
    # 		,ad_adcreative_name_material_effect \
    # 		,ad_adcreative_name_device_type \
    # 		,ad_adcreative_name_interest \
    # 		,ad_adcreative_name_behavior \
    # 		,ad_adcreative_name_paid \
    # 		,ade_image_hash \
    # 		,ade_title \
    # 		,ade_image2_hash \
    # 		,ade_image3_hash \
    # 		,ad_space \
    # 		,adc_form_size \
    # 		,adc_form_detail \
    # 		,adc_description \
    # 		,adg_bid_amount \
    # 		,log(adg_crt_days) \
    # 		,atv_ratio_bin11_hisratio_h \
    # 		,adc_drp1_1 \
    # 		,adc_drp1_2 \
    # 		,adc_drp1_3 \
    # 		,adc_drp1_4 \
    # 		,adc_drp1_5 \
    # 		,adc_drp1_6 \
    # 		,adc_drp1_7 \
    # 		,adc_drp1_8 \
    # 		,adc_drp1_9 \
    # 		,adc_drp1_10 \
    # 		,adc_drp1_11 \
    # 		,adc_drp1_12 \
    # 		,adc_drp1_13 \
    # 		,adc_drp1_14 \
    # 		,adc_drp1_15 \
    # 		,adc_drp1_16 \
    # 		,adc_drp1_17 \
    # 		,adc_drp1_18 \
    # 		,adc_drp1_19 \
    # 		,adc_drp1_20 \
    # 		,adc_drp1_21 \
    # 		,adc_drp1_22 \
    # 		,adc_drp1_23 \
    # 		,adc_drp1_24 \
    # 		,adc_drp1_25 \
    # 		,adc_drp1_26 \
    # 		,adc_drp1_27 \
    # 		,adc_drp1_28 \
    # 		,adc_drp1_29 \
    # 		,adc_drp1_30 \
    # 		,adc_drp1_31 \
    # 		,adc_drp1_32 \
    # 		,adc_drp1_33 \
    # 		,adc_drp1_34 \
    # 		,adc_drp1_35 \
    # 		,adc_drp1_36 \
    # 		,adc_drp1_37 \
    # 		,adc_drp1_38 \
    # 		,adc_drp2_1 \
    # 		,adc_drp2_2 \
    # 		,adc_drp2_3 \
    # 		,adc_drp2_4 \
    # 		,adc_drp2_5 \
    # 		,adc_drp2_6 \
    # 		,adc_drp2_7 \
    # 		,adc_drp2_8 \
    # 		,adc_drp2_9 \
    # 		,adc_drp3_1 \
    # 		,adc_drp3_2 \
    # 		,adc_drp3_3 \
    # 		,adc_drp3_4 \
    # 		,adc_drp3_5 \
    # 		,adc_drp3_6 \
    # 		,adc_drp3_7 \
    # 		,adc_drp3_8 \
    # 		,adc_drp3_9 \
    # 		,adc_drp3_10 \
    # 		,adc_drp3_11 \
    # 		,adc_drp3_12 \
    # 		,adc_drp3_13 \
    # 		,adc_drp3_14 \
    # 		,adc_drp3_15 \
    # 		,adc_drp3_16 \
    # 		,adc_drp3_17 \
    # 		,adc_drp3_18 \
    # 		,adc_drp3_19 \
    # 		,adc_drp3_20 \
    # 		,adc_drp3_21 \
    # 		,adc_drp3_22 \
    # 		,adc_drp3_23 \
    # 		,adc_drp3_24 \
    # 		,adc_drp3_25 \
    # 		,adc_drp3_26 \
    # 		,tag_age \
    # 		,tag_gender \
    # 		,tag_education \
    # 		,tag_relationship_status \
    # 		,tag_living_status \
    # 		,tag_business_interest \
    # 		,tag_location \
    # 		,tag_region \
    # 		,tag_gloc_location_types \
    # 		,tag_gloc_regions \
    # 		,tag_gloc_business_districts \
    # 		,tag_user_os \
    # 		,tag_new_device \
    # 		,tag_device_price \
    # 		,tag_network_type \
    # 		,tag_network_operator \
    # 		,tag_dressing_index \
    # 		,tag_uv_index \
    # 		,tag_makeup_index \
    # 		,tag_climate \
    # 		,tag_temperature \
    # 		,tag_app_install_status \
    # 		,tag_abh_object_type \
    # 		,tag_abh_object_id_list \
    # 		,tag_abh_time_window \
    # 		,tag_abh_act_id_list \
    # 		,tag_customized_audience \
    # 		,tag_shopping_capability \
    # 		,tag_player_consupt \
    # 		,tag_paying_user_type \
    # 		,tag_residential_community_price \
    # 		,tag_media_category_wechat \
    # 		,tag_ad_placement_id \
    # 		,tag_media_category_union \
    # 		,tag_qzone_fans \
    # 		,tag_online_scenario \
    # 		,tag_custom_audience \
    # 		,tag_boi_i_targeting_tags \
    # 		,concat(adg_bid_amount,ad_adcreative_name) \
    # 		,concat(adg_bid_amount,ad_adcreative_name,atv_ratio_bin11_hisratio_h) \
    # 		,concat(adg_bid_amount,atv_ratio_bin11_hisratio_h) \
    # 		,concat(adg_bid_amount,tag_gender) \
    # 		,concat(adg_bid_amount,tag_gloc_regions) \
    # 		,concat(adg_bid_amount,tag_gloc_regions,ad_space) \
    # 		,concat(adg_bid_amount,tag_gloc_regions,ad_space,adc_drp1_24) \
    # 		,concat(adg_bid_amount,tag_gloc_regions,adc_drp1_24) \
    # 		,concat(adg_bid_amount,adg_ts_workday1) \
    # 		,concat(adg_bid_amount,adg_ts_workday1,ad_adcreative_name) \
    # 		,concat(adg_bid_amount,adg_ts_workday1,tag_gender) \
    # 		,concat(adg_bid_amount,adg_ts_workday1,tag_gloc_regions) \
    # 		,concat(adg_bid_amount,adg_ts_workday1,tag_gloc_regions,ad_space) \
    # 		,concat(adg_bid_amount,adg_ts_workday1,tag_gloc_regions,ad_space,adc_drp1_24) \
    # 		,concat(adg_bid_amount,adg_ts_workday1,tag_gloc_regions,adc_drp1_24) \
    # 		,concat(ad_adcreative_name,atv_ratio_bin11_hisratio_h) \
    # 		,concat(ad_adcreative_name,tag_gender) \
    # 		,concat(ad_adcreative_name,tag_gender,atv_ratio_bin11_hisratio_h) \
    # 		,concat(tag_gloc_regions,ad_space) \
    # 		,concat(tag_gloc_regions,ad_space,adc_drp1_24) \
    # 		,concat(tag_gloc_regions,adc_drp1_24) \
    # 		,concat(adg_ts_workday1,ad_adcreative_name) \
    # 		,concat(adg_ts_workday1,ad_adcreative_name,atv_ratio_bin11_hisratio_h) \
    # 		,concat(adg_ts_workday1,ad_adcreative_name,tag_gender) \
    # 		,concat(adg_ts_workday1,atv_ratio_bin11_hisratio_h) \
    # 		,concat(adg_ts_workday1,tag_gender) \
    # 		,concat(adg_ts_workday1,tag_gloc_regions) \
    # 		,concat(adg_ts_workday1,tag_gloc_regions,ad_space) \
    # 		,concat(adg_ts_workday1,tag_gloc_regions,ad_space,adc_drp1_24) \
    # 		,concat(adg_ts_workday1,tag_gloc_regions,adc_drp1_24) \
    # 	    FROM %s \
    # 		where dt in %s and ind_first_industry_id=%d and adg_billing_event='%s' \
    # 		and id <= 100000000 %s " \
    # 		% (trans_tab, dt, indst_id, bill_evt, limit)

    sql = "SELECT \
            atv_flag \
			,id \
			,hour \
            ,day_week \
            ,day_type \
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
			,ad_adcreative_name \
			,ade_image_hash \
			,ade_title \
			,ade_title_len \
			,ad_space \
			,adc_form_size \
			,adc_form_detail \
			,adc_description \
			,adg_bid_amount \
			,adg_atv_days \
            ,ad_atv_days \
            ,adgcrt_imp_hours \
            ,adgcrt_last_atv_interval \
			,atv_ratio_bin11_hisratio_h \
            ,atv_ratio_hisavg \
			,adc_drp1_4 \
			,adc_drp1_17 \
			,adc_drp1_23 \
			,adc_drp1_24 \
			,adc_drp1_31 \
			,adc_drp1_32 \
			,adc_drp1_33 \
			,adc_drp3_5 \
			,adc_drp3_16 \
			,adc_drp3_26 \
			,tag_age_0_15 \
			,tag_age_15_20 \
			,tag_age_20_25 \
			,tag_age_25_30 \
			,tag_age_30_35 \
			,tag_age_35_40 \
			,tag_age_40_45 \
			,tag_age_45_50 \
			,tag_age_50 \
			,tag_gender \
			,tag_business_interest \
			,tag_gloc_location_types \
			,tag_gloc_regions \
			,tag_network_type \
			,tag_app_install_status \
			,tag_abh_object_type \
			,tag_abh_object_id_list \
			,tag_abh_time_window \
			,tag_abh_act_id_list \
			,tag_customized_audience \
			,tag_player_consupt \
			,tag_paying_user_type \
            ,concat(day_week,'\|',hour) \
            ,concat(day_type,'\|',hour) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(ad_adcreative_name,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(ad_adcreative_name,''),'\|',ifnull(atv_ratio_bin11_hisratio_h,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(atv_ratio_bin11_hisratio_h,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(tag_gender,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(tag_gloc_regions,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(tag_gloc_regions,''),'\|',ifnull(ad_space,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(tag_gloc_regions,''),'\|',ifnull(ad_space,''),'\|',ifnull(adc_drp1_24,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(tag_gloc_regions,''),'\|',ifnull(adc_drp1_24,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(adg_ts_workday1,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(adg_ts_workday1,''),'\|',ifnull(ad_adcreative_name,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(adg_ts_workday1,''),'\|',ifnull(tag_gender,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(adg_ts_workday1,''),'\|',ifnull(tag_gloc_regions,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(adg_ts_workday1,''),'\|',ifnull(ifnull(tag_gloc_regions,''),''),'\|',ifnull(ad_space,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(adg_ts_workday1,''),'\|',ifnull(tag_gloc_regions,''),'\|',ifnull(ad_space,''),'\|',ifnull(adc_drp1_24,'')) \
			,concat(ifnull(adg_bid_amount,''),'\|',ifnull(adg_ts_workday1,''),'\|',ifnull(tag_gloc_regions,''),'\|',ifnull(adc_drp1_24,'')) \
			,concat(ifnull(ad_adcreative_name,''),'\|',ifnull(atv_ratio_bin11_hisratio_h,'')) \
			,concat(ifnull(ad_adcreative_name,''),'\|',ifnull(tag_gender,'')) \
			,concat(ifnull(ad_adcreative_name,''),'\|',ifnull(tag_gender,''),'\|',ifnull(atv_ratio_bin11_hisratio_h,'')) \
			,concat(ifnull(tag_gloc_regions,''),'\|',ifnull(ad_space,'')) \
			,concat(ifnull(tag_gloc_regions,''),'\|',ifnull(ad_space,''),ifnull(adc_drp1_24,'')) \
			,concat(ifnull(tag_gloc_regions,''),'\|',ifnull(adc_drp1_24,'')) \
			,concat(ifnull(adg_ts_workday1,''),'\|',ifnull(ad_adcreative_name,'')) \
			,concat(ifnull(adg_ts_workday1,''),'\|',ifnull(ad_adcreative_name,''),'\|',ifnull(atv_ratio_bin11_hisratio_h,'')) \
			,concat(ifnull(adg_ts_workday1,''),'\|',ifnull(ad_adcreative_name,''),'\|',ifnull(tag_gender,'')) \
			,concat(ifnull(adg_ts_workday1,''),'\|',ifnull(atv_ratio_bin11_hisratio_h,'')) \
			,concat(ifnull(adg_ts_workday1,''),'\|',ifnull(tag_gender,'')) \
			,concat(ifnull(adg_ts_workday1,''),'\|',ifnull(tag_gloc_regions,'')) \
			,concat(ifnull(adg_ts_workday1,''),'\|',ifnull(tag_gloc_regions,''),'\|',ifnull(ad_space,'')) \
			,concat(ifnull(adg_ts_workday1,''),'\|',ifnull(tag_gloc_regions,''),'\|',ifnull(ad_space,''),'\|',ifnull(adc_drp1_24,'')) \
			,concat(ifnull(adg_ts_workday1,''),'\|',ifnull(tag_gloc_regions,''),'\|',ifnull(adc_drp1_24,'')) \
			FROM %s \
			where %s and ind_first_industry_id=%d and adg_billing_event='%s' \
			and adg_site_set='tss_SITE_SET_MOBILE_INNER' and id <= 100000000  %s " \
          % (trans_tab, dt, indst_id, bill_evt, limit)

    # print(sql)
    rst = mysql.getAll(sql)

    if not rst:
        mysql.dispose()  # 释放资源
        print('		No data!')
        return -1
    else:
        # 训练模型不能有缺失值，补0
        df = pd.DataFrame(list(list(x.values()) for x in rst)).fillna('0')
        print('		%d rows data have been fetched.' % len(df))
        mysql.dispose()
        return df


def proc_tuple(s):
    '''将[(a, b), (c, d)]处理为[a_b, c_d]形式'''
    return list(x[0] + '_' + x[1] for x in s)


def proc_space(s):
    '''去除list元素中的多余空格'''
    return list(x.strip() for x in s)


def comb_cols(col):
    result_lst = []
    if type(col) == str:
        cols_lst = col.split('|')
        cols_num = len(cols_lst)
        for col in range(cols_num):
            if col == 0:
                result_lst = []
            if col == 1:
                tmp1 = cols_lst[col - 1].split(',')
                tmp2 = cols_lst[col].split(',')
                a = list(itertools.product(proc_space(tmp1), proc_space(tmp2)))
                result_lst = proc_tuple(a)
            if col > 1:
                tmp3 = cols_lst[col].split(',')
                a = list(itertools.product(result_lst, proc_space(tmp3)))
                result_lst = proc_tuple(a)
        return result_lst
    else:
        return []


def oneEncoding(ohfeats, unohfeats, mulohfeats, combohfeats, dfTrain, dfTest, stype='std'):
    # 先拼接onehot字段
    feat_idx = []
    enc = LabelBinarizer(sparse_output=True)  # 字符串型类别变量只能用LabelBinarizer()
    cn = 0
    for i, feat in enumerate(ohfeats):
        x_train = enc.fit_transform(dfTrain.iloc[:, feat].values.reshape(-1, 1))
        x_test = enc.transform(dfTest.iloc[:, feat].values.reshape(-1, 1))
        if i == 0:
            X_train, X_test = x_train, x_test
        else:
            X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
        # X_train, X_test = np.hstack((X_train, x_train)), np.hstack((X_test, x_test))

        # 拼接索引标签
        ec = list(enc.classes_)
        if len(ec) == 1:
            feat_idx.append('%d:%s %d' % (feat, ec, cn))
            cn += 1
        elif len(ec) == 2:    # LabelBinarizer对只有2个不同取值的字段只返回1维特征，即第0维
            ec = ec[0]
            feat_idx.append('%d:%s %d' % (feat, ec, cn))
            cn += 1
        else:
            for j in range(len(ec)):
                feat_idx.append('%d:%s %d' % (feat, ec[j], cn))
                cn += 1
        print('X_train: ', X_train.shape[1])
        print('x_test: ', x_test.shape[1])
        print('cn:      ', cn)
    # print('X_train: ', X_train.shape[1])
    # print('cn: ', cn)
    # print(feat_idx)
    # exit(0)

    # 拼接非onehot字段
    for i, feat in enumerate(unohfeats):
        x1 = dfTrain.iloc[:, feat].values.reshape(-1, 1)
        x2 = dfTest.iloc[:, feat].values.reshape(-1, 1)

        if stype == 'std':
            scaler = StandardScaler().fit(x1)
        elif stype == 'mm':
            scaler = MaxAbsScaler().fit(x1)
        x_train = scaler.transform(x1)
        x_test = scaler.transform(x2)
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
        # X_train, X_test = np.hstack((X_train, x_train)), np.hstack((X_test, x_test))

        feat_idx.append('%d:%s %d' % (feat, 'unohfeat', cn))
        cn += 1
        print('X_train: ', X_train.shape[1])
        print('cn:      ', cn)
    # print(dfTrain.iloc[:,feat].values)


    # 拼接多值的onehot字段，
    # 字段值形如'547135, 3547136, 3547137, 3547102, 3547096, 3547092, 3547091, 3547090'
    Total_df = pd.concat([dfTrain, dfTest], keys=['train', 'test'])
    encm = MultiLabelBinarizer(sparse_output=True)
    for i, feat in enumerate(mulohfeats):
        # print(feat)
        x1 = Total_df.iloc[:, feat].apply(splitComa)
        x2 = dfTest.iloc[:, feat].apply(splitComa)
        x_train = encm.fit_transform(x1)
        x_train = sparse.coo_matrix(x_train.toarray()[:dfTrain.shape[0],:])
        x_test = encm.transform(x2)
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
        # X_train, X_test = np.hstack((X_train, x_train)), np.hstack((X_test, x_test))

        ec = list(encm.classes_)
        for j in range(len(ec)):
            feat_idx.append('%d:%s %d' % (feat, ec[j], cn))
            cn += 1
            # print('x_testML: \n', x_test)
        print('X_train: ', X_train.shape[1])
        print('cn:      ', cn)
    # exit()

    for i, feat in enumerate(combohfeats):
        x1 = Total_df.iloc[:, feat].apply(comb_cols)
        x2 = dfTest.iloc[:, feat].apply(comb_cols)
        x_train = encm.fit_transform(x1)
        x_train = sparse.coo_matrix(x_train.toarray()[:dfTrain.shape[0],:])
        x_test = encm.transform(x2)
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
        # X_train, X_test = np.hstack((X_train, x_train)), np.hstack((X_test, x_test))
        # print(dfTrain.iloc[:, feat])
        # print(dfTrain.iloc[:, feat].apply(comb_cols).apply(splitComa))

        ec = list(encm.classes_)
        for j in range(len(ec)):
            feat_idx.append('%d:%s %d' % (feat, ec[j], cn))
            cn += 1
        print('X_train: ', X_train.shape[1])
        print('cn:      ', cn)

    print('		%d onehot encoding concat: done!' % len(feat_idx))
    return X_train, X_test, feat_idx